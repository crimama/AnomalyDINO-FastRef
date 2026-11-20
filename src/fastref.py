import contextlib
from dataclasses import dataclass
import math

import numpy as np
import torch

_VALID_METRICS = {"cosine", "euclidean"}
_VALID_SCORE_MODES = {"reconstruction", "nn"}
_VALID_OT_SCALES = {"row_normalized", "unscaled"}
_VALID_CORESET_MODES = {"greedy_kcenter", "approx_kcenter", "random"}


@dataclass
class FastRefConfig:
    """Hyperparameters and variant flags for FastRef reference refinement."""

    alpha: float = 0.5
    lam: float = 0.1
    num_iters: int = 2
    sinkhorn_iters: int = 10
    sinkhorn_eps: float = 0.1
    metric: str = "cosine"
    score_mode: str = "reconstruction"
    ot_scale: str = "row_normalized"
    coreset_mode: str = "greedy_kcenter"
    ridge: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.alpha) and 0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not (math.isfinite(self.lam) and self.lam >= 0.0):
            raise ValueError(f"lam must be finite and >= 0, got {self.lam}")
        if self.num_iters < 0:
            raise ValueError(f"num_iters must be >= 0, got {self.num_iters}")
        if self.sinkhorn_iters < 1:
            raise ValueError(f"sinkhorn_iters must be >= 1, got {self.sinkhorn_iters}")
        if not (math.isfinite(self.sinkhorn_eps) and self.sinkhorn_eps > 0.0):
            raise ValueError(f"sinkhorn_eps must be finite and > 0, got {self.sinkhorn_eps}")
        if not (math.isfinite(self.ridge) and self.ridge >= 0.0):
            raise ValueError(f"ridge must be finite and >= 0, got {self.ridge}")
        if self.metric not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {sorted(_VALID_METRICS)}, got {self.metric!r}")
        if self.score_mode not in _VALID_SCORE_MODES:
            raise ValueError(
                f"score_mode must be one of {sorted(_VALID_SCORE_MODES)}, got {self.score_mode!r}"
            )
        if self.ot_scale not in _VALID_OT_SCALES:
            raise ValueError(
                f"ot_scale must be one of {sorted(_VALID_OT_SCALES)}, got {self.ot_scale!r}"
            )
        if self.coreset_mode not in _VALID_CORESET_MODES:
            raise ValueError(
                f"coreset_mode must be one of {sorted(_VALID_CORESET_MODES)}, got {self.coreset_mode!r}"
            )


@contextlib.contextmanager
def _blas_thread_cap(max_threads: int = 4):
    """Cap BLAS threads inside tight dgemv loops to avoid multi-core barrier thrashing."""
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        yield
        return

    controller = ThreadpoolController()
    thread_counts = [
        lib["num_threads"]
        for lib in controller.info()
        if lib.get("user_api") == "blas" and lib.get("num_threads")
    ]
    if not thread_counts or max(thread_counts) <= max_threads:
        yield
        return

    cap = min(max_threads, min(thread_counts))
    with controller.limit(limits=cap, user_api="blas"):
        yield


def coreset_subsample(
    features: np.ndarray,
    ratio: float,
    mode: str = "greedy_kcenter",
    seed: int = 0,
    proj_dim: int = 128,
) -> np.ndarray:
    """Subsample rows of a 2D feature matrix using k-center or random coreset selection."""
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"features must be a non-empty 2D array, got shape {features.shape}")
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if mode not in _VALID_CORESET_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_CORESET_MODES)}, got {mode!r}")
    if proj_dim < 1:
        raise ValueError(f"proj_dim must be >= 1, got {proj_dim}")

    n_total, dim = features.shape
    n_select = max(1, min(n_total, int(round(ratio * n_total))))
    if n_select == n_total:
        return features.copy()

    rng = np.random.default_rng(seed)
    if mode == "random":
        indices = rng.choice(n_total, size=n_select, replace=False)
        return features[indices].copy()

    work = np.ascontiguousarray(features, dtype=np.float64)
    if mode == "approx_kcenter" and dim > proj_dim:
        proj = rng.normal(loc=0.0, scale=1.0 / math.sqrt(proj_dim), size=(dim, proj_dim))
        work = work @ proj

    start = int(rng.integers(0, n_total))
    selected = [start]
    work_norm_sq = np.einsum("ij,ij->i", work, work)
    with _blas_thread_cap(max_threads=4):
        min_sq_dists = np.maximum(0.0, work_norm_sq + work_norm_sq[start] - 2.0 * (work @ work[start]))
        min_sq_dists[start] = -1.0  # Sentinel prevents re-selecting chosen indices on duplicate points.

        for _ in range(1, n_select):
            idx = int(np.argmax(min_sq_dists))
            selected.append(idx)
            sq_dists = np.maximum(0.0, work_norm_sq + work_norm_sq[idx] - 2.0 * (work @ work[idx]))
            np.minimum(min_sq_dists, sq_dists, out=min_sq_dists)
            min_sq_dists[idx] = -1.0

    return features[selected]


def sinkhorn_transport(
    cost_matrix: torch.Tensor,
    eps: float = 0.1,
    num_iters: int = 10,
) -> torch.Tensor:
    """Compute entropic-regularized optimal transport plan in log domain with uniform marginals."""
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        raise ValueError(f"cost_matrix must be a non-empty 2D tensor, got shape {tuple(cost_matrix.shape)}")
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    if num_iters < 1:
        raise ValueError(f"num_iters must be >= 1, got {num_iters}")

    m, n = cost_matrix.shape
    device = cost_matrix.device
    log_a = torch.full((m,), -math.log(m), dtype=torch.float64, device=device)
    log_b = torch.full((n,), -math.log(n), dtype=torch.float64, device=device)
    log_k = -cost_matrix.to(dtype=torch.float64) / eps
    u = torch.zeros(m, dtype=torch.float64, device=device)
    v = torch.zeros(n, dtype=torch.float64, device=device)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)

    return torch.exp(u.unsqueeze(1) + v.unsqueeze(0) + log_k)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize rows of a 2D tensor with a lower norm clamp for zero-vector safety."""
    norms = torch.linalg.norm(x, dim=1, keepdim=True).clamp(min=eps)
    return x / norms


def _pairwise_cost(x: torch.Tensor, y: torch.Tensor, metric: str) -> torch.Tensor:
    """Compute pairwise non-negative cosine or squared-Euclidean cost between 2D tensors."""
    if metric == "cosine":
        x_norm = _l2_normalize(x)
        y_norm = _l2_normalize(y)
        return (0.5 * (1.0 - x_norm @ y_norm.T)).clamp(0.0, 1.0)
    if metric == "euclidean":
        return torch.cdist(x, y, p=2).pow(2)
    raise ValueError(f"Unsupported metric: {metric!r}")


class FastRefRefiner:
    """Closed-form test-time prototype refiner using Sinkhorn OT alignment."""

    def __init__(self, config: FastRefConfig, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.M_s: torch.Tensor | None = None
        self.M_pinv: torch.Tensor | None = None
        self.P_proj: torch.Tensor | None = None

    def _require_fitted(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.M_s is None or self.M_pinv is None or self.P_proj is None:
            raise RuntimeError("FastRefRefiner must be fitted before calling refine/score.")
        return self.M_s, self.M_pinv, self.P_proj

    def _prepare_query(self, query_features: np.ndarray | torch.Tensor, c: int) -> torch.Tensor:
        if isinstance(query_features, torch.Tensor):
            f_q = query_features.detach().to(device=self.device, dtype=torch.float64)
        else:
            f_q = torch.as_tensor(np.asarray(query_features), dtype=torch.float64, device=self.device)
        if f_q.ndim != 2 or f_q.shape[1] != c:
            raise ValueError(
                f"query_features must be a 2D array with {c} columns, got shape {tuple(f_q.shape)}"
            )
        if self.config.metric == "cosine" and f_q.shape[0] > 0:
            f_q = _l2_normalize(f_q)
        return f_q

    def fit(self, support_features: np.ndarray | torch.Tensor) -> "FastRefRefiner":
        """Build coreset prototypes and precompute closed-form projection operators."""
        if isinstance(support_features, torch.Tensor):
            sup_np = support_features.detach().cpu().numpy()
        else:
            sup_np = np.asarray(support_features)

        if sup_np.ndim != 2 or sup_np.shape[0] == 0 or sup_np.shape[1] == 0:
            raise ValueError(
                f"support_features must be a non-empty 2D array, got shape {sup_np.shape}"
            )

        sup_work = sup_np.astype(np.float64, copy=True)
        if self.config.metric == "cosine":
            norms = np.maximum(np.linalg.norm(sup_work, axis=1, keepdims=True), 1e-12)
            sup_work = sup_work / norms

        prototypes = coreset_subsample(
            sup_work,
            ratio=self.config.alpha,
            mode=self.config.coreset_mode,
            seed=self.config.seed,
        )

        m_s = torch.as_tensor(prototypes, dtype=torch.float64, device=self.device)
        gram = m_s @ m_s.T
        if self.config.ridge == 0.0:
            gram_inv = torch.linalg.pinv(gram, hermitian=True)
        else:
            eye = torch.eye(gram.shape[0], dtype=torch.float64, device=self.device)
            gram_inv = torch.linalg.inv(gram + self.config.ridge * eye)

        self.M_s = m_s
        self.M_pinv = m_s.T @ gram_inv
        self.P_proj = gram @ gram_inv
        return self

    def _refine_prepared(
        self, f_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        m_s, m_pinv, p_proj = self._require_fitted()
        m = f_q.shape[0]
        n, c = m_s.shape
        if m == 0:
            return (
                torch.zeros((0, c), dtype=torch.float64, device=self.device),
                torch.zeros((0, n), dtype=torch.float64, device=self.device),
                None,
            )

        w_0 = f_q @ m_pinv
        if self.config.lam == 0.0 or self.config.num_iters == 0:
            return w_0 @ m_s, w_0, None

        w = w_0
        t_plan: torch.Tensor | None = None
        for _ in range(self.config.num_iters):
            m_w = w @ m_s
            cost = _pairwise_cost(m_w, m_s, self.config.metric)
            t_plan = sinkhorn_transport(
                cost, eps=self.config.sinkhorn_eps, num_iters=self.config.sinkhorn_iters
            )
            t_eff = (m * t_plan) if self.config.ot_scale == "row_normalized" else t_plan
            denom = 1.0 + self.config.lam * t_eff.sum(dim=1, keepdim=True)
            w = (w_0 + self.config.lam * (t_eff @ p_proj)) / denom

        return w @ m_s, w, t_plan

    def refine(
        self, query_features: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Iteratively refine prototypes for a single query image's patch features."""
        m_s, _, _ = self._require_fitted()
        f_q = self._prepare_query(query_features, m_s.shape[1])
        return self._refine_prepared(f_q)

    def score(self, query_features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute per-patch anomaly scores (1D float32 array) for query features."""
        m_s, _, _ = self._require_fitted()
        f_q = self._prepare_query(query_features, m_s.shape[1])
        if f_q.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        refined, _, _ = self._refine_prepared(f_q)
        if self.config.score_mode == "reconstruction":
            if self.config.metric == "cosine":
                refined_norm = _l2_normalize(refined)
                scores = (0.5 * (1.0 - (f_q * refined_norm).sum(dim=1))).clamp(0.0, 1.0)
            else:
                scores = ((f_q - refined) ** 2).sum(dim=1)
        elif self.config.score_mode == "nn":
            scores = _pairwise_cost(f_q, refined, self.config.metric).min(dim=1).values
        else:
            raise ValueError(f"Unsupported score_mode: {self.config.score_mode!r}")

        return scores.detach().cpu().numpy().astype(np.float32)
