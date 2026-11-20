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
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.lam < 0.0:
            raise ValueError(f"lam must be >= 0, got {self.lam}")
        if self.num_iters < 0:
            raise ValueError(f"num_iters must be >= 0, got {self.num_iters}")
        if self.sinkhorn_iters < 1:
            raise ValueError(f"sinkhorn_iters must be >= 1, got {self.sinkhorn_iters}")
        if self.sinkhorn_eps <= 0.0:
            raise ValueError(f"sinkhorn_eps must be > 0, got {self.sinkhorn_eps}")
        if self.ridge < 0.0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")
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

    work = features.astype(np.float64, copy=False)
    if mode == "approx_kcenter" and dim > proj_dim:
        proj = rng.normal(loc=0.0, scale=1.0 / math.sqrt(proj_dim), size=(dim, proj_dim))
        work = work @ proj

    start = int(rng.integers(0, n_total))
    selected = [start]
    min_sq_dists = np.sum((work - work[start]) ** 2, axis=1)
    min_sq_dists[start] = -1.0  # Sentinel prevents re-selecting chosen indices on duplicate points.

    for _ in range(1, n_select):
        idx = int(np.argmax(min_sq_dists))
        selected.append(idx)
        sq_dists = np.sum((work - work[idx]) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)
        min_sq_dists[idx] = -1.0

    return features[selected].copy()


def sinkhorn_transport(
    cost_matrix: torch.Tensor,
    eps: float = 0.1,
    num_iters: int = 10,
) -> torch.Tensor:
    """Compute entropic-regularized optimal transport plan in log domain with uniform marginals."""
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        raise ValueError(f"cost_matrix must be a non-empty 2D tensor, got shape {tuple(cost_matrix.shape)}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
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
