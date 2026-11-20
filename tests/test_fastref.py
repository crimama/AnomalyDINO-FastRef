import time

import numpy as np
import pytest
import torch

from src.fastref import FastRefConfig, FastRefRefiner, coreset_subsample, sinkhorn_transport


def test_config_defaults_match_cvpr2026():
    cfg = FastRefConfig()
    assert cfg.alpha == 0.5
    assert cfg.lam == 0.1
    assert cfg.num_iters == 2
    assert cfg.sinkhorn_iters == 10
    assert cfg.sinkhorn_eps == 0.1
    assert cfg.metric == "cosine"
    assert cfg.score_mode == "reconstruction"
    assert cfg.ot_scale == "row_normalized"
    assert cfg.coreset_mode == "greedy_kcenter"
    assert cfg.ridge == 0.0
    assert cfg.seed == 0


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0.0},
    {"alpha": -0.1},
    {"alpha": 1.5},
    {"lam": -0.01},
    {"num_iters": -1},
    {"sinkhorn_iters": 0},
    {"sinkhorn_eps": 0.0},
    {"sinkhorn_eps": -0.1},
    {"metric": "manhattan"},
    {"score_mode": "max"},
    {"ot_scale": "col_normalized"},
    {"coreset_mode": "kmeans"},
    {"ridge": -1e-4},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        FastRefConfig(**kwargs)


@pytest.mark.parametrize("mode", ["greedy_kcenter", "approx_kcenter", "random"])
def test_coreset_subsample_shape_and_determinism(mode):
    rng = np.random.default_rng(42)
    feats = rng.normal(size=(80, 16)).astype(np.float32)
    out1 = coreset_subsample(feats, ratio=0.25, mode=mode, seed=7)
    out2 = coreset_subsample(feats, ratio=0.25, mode=mode, seed=7)
    assert out1.shape == (20, 16)
    np.testing.assert_array_equal(out1, out2)


def test_coreset_passthrough_when_ratio_is_one():
    feats = np.arange(30, dtype=np.float32).reshape(10, 3)
    out = coreset_subsample(feats, ratio=1.0, mode="greedy_kcenter", seed=0)
    np.testing.assert_array_equal(out, feats)
    assert out is not feats  # returns a copy, not an alias


def test_coreset_greedy_kcenter_covers_outliers():
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=0.0, scale=0.05, size=(50, 4))
    outliers = np.array([[10.0, 0, 0, 0], [0, 10.0, 0, 0], [0, 0, 10.0, 0]])
    feats = np.vstack([cluster, outliers]).astype(np.float32)
    selected = coreset_subsample(feats, ratio=4 / len(feats), mode="greedy_kcenter", seed=0)
    assert selected.shape == (4, 4)
    # Greedy k-center must pick all 3 far outliers
    for pt in outliers:
        assert np.min(np.linalg.norm(selected - pt, axis=1)) < 1e-5


def test_coreset_approx_kcenter_projects_high_dim_and_preserves_original_dim():
    rng = np.random.default_rng(123)
    feats = rng.normal(size=(40, 256)).astype(np.float32)
    out = coreset_subsample(feats, ratio=0.25, mode="approx_kcenter", seed=3, proj_dim=32)
    assert out.shape == (10, 256)
    assert out.dtype == np.float32


@pytest.mark.parametrize("bad_kwargs", [
    {"features": np.zeros((10,), dtype=np.float32), "ratio": 0.5},
    {"features": np.zeros((0, 4), dtype=np.float32), "ratio": 0.5},
    {"features": np.zeros((10, 0), dtype=np.float32), "ratio": 0.5},
    {"features": np.zeros((10, 4), dtype=np.float32), "ratio": 0.0},
    {"features": np.zeros((10, 4), dtype=np.float32), "ratio": 1.2},
    {"features": np.zeros((10, 4), dtype=np.float32), "ratio": 0.5, "mode": "kmeans"},
])
def test_coreset_rejects_invalid_inputs(bad_kwargs):
    with pytest.raises(ValueError):
        coreset_subsample(**bad_kwargs)


def test_sinkhorn_marginals_positivity_and_cost_preference():
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.1, 0.9, 1.8]],
        dtype=torch.float64,
    )
    # 100 iterations needed for eps=0.1 on this 4x3 cost matrix to reach atol=1e-5 on row marginals
    # (50 iterations leaves a ~8.6e-5 row residual).
    T = sinkhorn_transport(cost, eps=0.1, num_iters=100)
    assert T.shape == (4, 3)
    assert T.dtype == torch.float64
    assert torch.all(T > 0)
    assert torch.allclose(T.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(T.sum(dim=1), torch.full((4,), 1.0 / 4, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(T.sum(dim=0), torch.full((3,), 1.0 / 3, dtype=torch.float64), atol=1e-5)
    assert T[0, 0] > T[0, 2]
    assert T[2, 2] > T[2, 0]


@pytest.mark.parametrize("shape,eps,scale", [((1, 1), 0.1, 1.0), ((16, 8), 1e-3, 100.0)])
def test_sinkhorn_numerical_stability_and_edge_shapes(shape, eps, scale):
    torch.manual_seed(0)
    cost = torch.rand(shape, dtype=torch.float64) * scale
    T = sinkhorn_transport(cost, eps=eps, num_iters=10)
    assert T.shape == shape
    assert torch.isfinite(T).all()
    assert torch.all(T >= 0)
    assert torch.allclose(T.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-5)


@pytest.mark.parametrize("cost,eps,num_iters", [
    (torch.zeros(4), 0.1, 10),
    (torch.zeros((0, 3)), 0.1, 10),
    (torch.zeros((3, 0)), 0.1, 10),
    (torch.zeros((3, 3)), 0.0, 10),
    (torch.zeros((3, 3)), -0.1, 10),
    (torch.zeros((3, 3)), 0.1, 0),
])
def test_sinkhorn_rejects_invalid_inputs(cost, eps, num_iters):
    with pytest.raises(ValueError):
        sinkhorn_transport(cost, eps=eps, num_iters=num_iters)


def test_refiner_raises_if_called_before_fit():
    refiner = FastRefRefiner(FastRefConfig(), device="cpu")
    q = np.ones((4, 8), dtype=np.float32)
    with pytest.raises(RuntimeError, match="fitted"):
        refiner.refine(q)
    with pytest.raises(RuntimeError, match="fitted"):
        refiner.score(q)


@pytest.mark.parametrize("n_sup,dim", [(12, 32), (32, 12)])  # n < c (full-rank) and n > c (rank-deficient)
def test_closed_form_w_update_satisfies_kkt_zero_gradient(n_sup, dim):
    rng = np.random.default_rng(123)
    sup = rng.normal(size=(n_sup, dim)).astype(np.float64)
    qry = rng.normal(size=(10, dim)).astype(np.float64)
    cfg = FastRefConfig(
        alpha=1.0, lam=0.3, num_iters=1, sinkhorn_iters=10,
        sinkhorn_eps=0.5, metric="euclidean", ot_scale="row_normalized", seed=0,
    )
    refiner = FastRefRefiner(cfg, device="cpu").fit(sup)
    refined, W_star, T_star = refiner.refine(qry)
    assert T_star is not None
    # Verify Appendix B stationary condition: dL/dW = 0 at W_star for fixed T_eff
    M = torch.as_tensor(sup, dtype=torch.float64)
    F = torch.as_tensor(qry, dtype=torch.float64)
    T_eff = qry.shape[0] * T_star
    denom = (1.0 + cfg.lam * T_eff.sum(dim=1, keepdim=True))
    grad = 2.0 * (denom * (W_star @ M) - F - cfg.lam * (T_eff @ M)) @ M.T
    assert torch.max(torch.abs(grad)).item() < 1e-9
    assert torch.allclose(refined, W_star @ M, atol=1e-10)


def test_euclidean_objective_decreases_monotonically_across_iterations():
    rng = np.random.default_rng(7)
    sup = rng.normal(size=(16, 24)).astype(np.float64)
    qry = rng.normal(size=(12, 24)).astype(np.float64)
    M = torch.as_tensor(sup, dtype=torch.float64)
    F = torch.as_tensor(qry, dtype=torch.float64)

    def eval_obj(W, T, lam):
        rec = ((F - W @ M) ** 2).sum()
        ot_cost = torch.cdist(W @ M, M, p=2).pow(2)
        return (rec + lam * (qry.shape[0] * T * ot_cost).sum()).item()

    objs = []
    for iters in (1, 2, 3, 4):
        cfg = FastRefConfig(
            alpha=1.0, lam=0.3, num_iters=iters, sinkhorn_iters=50,
            sinkhorn_eps=0.5, metric="euclidean", ot_scale="row_normalized",
        )
        refiner = FastRefRefiner(cfg, device="cpu").fit(sup)
        _, W, T = refiner.refine(qry)
        objs.append(eval_obj(W, T, cfg.lam))
    for prev, curr in zip(objs, objs[1:]):
        assert curr <= prev + 1e-6


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
@pytest.mark.parametrize("n_sup,dim", [(24, 64), (48, 16)])  # n < c and n > c (dinov2_vits14 regime)
@pytest.mark.parametrize("score_mode", ["reconstruction", "nn"])
def test_anomaly_separation_and_ot_suppression(metric, n_sup, dim, score_mode):
    rng = np.random.default_rng(99)
    # Normal features on first two orthogonal thirds; anomalies on the third orthogonal third
    d1 = dim // 3
    d2 = 2 * dim // 3
    base1 = np.zeros(dim); base1[:d1] = 1.0
    base2 = np.zeros(dim); base2[d1:d2] = 1.0
    anomaly_dir = np.zeros(dim); anomaly_dir[d2:] = 1.0
    sup = np.vstack([
        base1 + 0.05 * rng.normal(size=(n_sup // 2, dim)),
        base2 + 0.05 * rng.normal(size=(n_sup - n_sup // 2, dim)),
    ]).astype(np.float32)
    normal_q = base1 + 0.05 * rng.normal(size=(10, dim))
    anomaly_q = anomaly_dir + 0.05 * rng.normal(size=(2, dim))
    qry = np.vstack([normal_q, anomaly_q]).astype(np.float32)

    cfg = FastRefConfig(
        alpha=0.75, lam=0.1, num_iters=2, sinkhorn_iters=10,
        sinkhorn_eps=0.1, metric=metric, score_mode=score_mode, ot_scale="row_normalized",
    )
    scores = FastRefRefiner(cfg, device="cpu").fit(sup).score(qry)
    assert scores.shape == (12,)
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    assert scores[-2:].min() > scores[:10].max()

    cfg_no_ot = FastRefConfig(
        alpha=0.75, lam=0.0, num_iters=2, sinkhorn_iters=10,
        sinkhorn_eps=0.1, metric=metric, score_mode=score_mode, ot_scale="row_normalized",
    )
    scores_no_ot = FastRefRefiner(cfg_no_ot, device="cpu").fit(sup).score(qry)
    assert np.all(scores[-2:] > scores_no_ot[-2:])
    if n_sup > dim:
        assert scores_no_ot[-2:].max() < 1e-5


def test_ablation_lambda_zero_and_num_iters_zero_return_w0():
    rng = np.random.default_rng(11)
    sup = rng.normal(size=(10, 20)).astype(np.float32)
    qry = rng.normal(size=(6, 20)).astype(np.float32)
    r_lam0 = FastRefRefiner(FastRefConfig(alpha=1.0, lam=0.0, num_iters=2), device="cpu").fit(sup)
    r_it0 = FastRefRefiner(FastRefConfig(alpha=1.0, lam=0.1, num_iters=0), device="cpu").fit(sup)
    ref1, W1, T1 = r_lam0.refine(qry)
    ref2, W2, T2 = r_it0.refine(qry)
    assert T1 is None and T2 is None
    assert torch.allclose(W1, W2)
    assert torch.allclose(ref1, ref2)


def test_ot_scale_unscaled_and_ridge_produce_finite_scores():
    rng = np.random.default_rng(5)
    sup = rng.normal(size=(30, 12)).astype(np.float32)
    qry = rng.normal(size=(10, 12)).astype(np.float32)
    for ot_scale, ridge in [("unscaled", 0.0), ("row_normalized", 1e-2)]:
        cfg = FastRefConfig(alpha=0.5, lam=0.1, num_iters=2, ot_scale=ot_scale, ridge=ridge)
        scores = FastRefRefiner(cfg, device="cpu").fit(sup).score(qry)
        assert scores.shape == (10,)
        assert np.isfinite(scores).all()


def test_edge_cases_empty_single_duplicate_zero_norm_and_immutability():
    # Duplicate (rank-1) support features + zero-norm query + caller immutability
    sup = np.ones((20, 8), dtype=np.float32) * 3.0
    sup_orig = sup.copy()
    refiner = FastRefRefiner(FastRefConfig(alpha=0.5, metric="cosine"), device="cpu").fit(sup)
    np.testing.assert_array_equal(sup, sup_orig)  # caller support array not mutated

    # Empty query (m == 0)
    empty_q = np.zeros((0, 8), dtype=np.float32)
    ref_empty, w_empty, t_empty = refiner.refine(empty_q)
    assert ref_empty.shape == (0, 8) and ref_empty.dtype == torch.float64
    assert w_empty.shape == (0, 10) and w_empty.dtype == torch.float64
    assert t_empty is None
    empty_scores = refiner.score(empty_q)
    assert empty_scores.shape == (0,) and empty_scores.dtype == np.float32

    # Single query (m == 1) and zero-norm row
    q_zero = np.zeros((1, 8), dtype=np.float32)
    q_zero_orig = q_zero.copy()
    s_zero = refiner.score(q_zero)
    assert s_zero.shape == (1,) and np.isfinite(s_zero).all()
    np.testing.assert_array_equal(q_zero, q_zero_orig)  # caller query array not mutated

    # Single support prototype (n == 1)
    r_one = FastRefRefiner(FastRefConfig(alpha=1.0, metric="cosine"), device="cpu").fit(np.ones((1, 8), dtype=np.float32))
    s_one = r_one.score(np.ones((3, 8), dtype=np.float32))
    assert s_one.shape == (3,) and np.isfinite(s_one).all()


def test_refiner_accepts_torch_tensors_and_rejects_invalid_shapes():
    sup_t = torch.ones((6, 4), dtype=torch.float32)
    sup_orig = sup_t.clone()
    qry_t = torch.ones((3, 4), dtype=torch.float32) * 2.0
    qry_orig = qry_t.clone()

    refiner = FastRefRefiner(FastRefConfig(alpha=0.5), device="cpu").fit(sup_t)
    assert torch.equal(sup_t, sup_orig)
    scores = refiner.score(qry_t)
    assert torch.equal(qry_t, qry_orig)
    assert scores.shape == (3,) and scores.dtype == np.float32

    with pytest.raises(ValueError):
        FastRefRefiner(FastRefConfig(), device="cpu").fit(torch.ones(4))
    with pytest.raises(ValueError):
        refiner.refine(np.ones((2, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        refiner.score(np.ones((4,), dtype=np.float32))


@pytest.mark.parametrize("field", ["alpha", "lam", "sinkhorn_eps", "ridge"])
@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_values(field, bad_val):
    with pytest.raises(ValueError):
        FastRefConfig(**{field: bad_val})


@pytest.mark.parametrize("bad_eps", [float("nan"), float("inf"), float("-inf")])
def test_sinkhorn_rejects_non_finite_eps(bad_eps):
    with pytest.raises(ValueError):
        sinkhorn_transport(torch.zeros((3, 3)), eps=bad_eps, num_iters=10)


def _reference_greedy_kcenter(features: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    n_total = features.shape[0]
    n_select = max(1, min(n_total, int(round(ratio * n_total))))
    work = features.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_total))
    selected = [start]
    selected_set = {start}
    for _ in range(1, n_select):
        best_idx = -1
        best_dist = -1.0
        for i in range(n_total):
            if i in selected_set:
                continue
            d = min(float(np.sum((work[i] - work[s]) ** 2)) for s in selected)
            if d > best_dist:
                best_dist = d
                best_idx = i
        selected.append(best_idx)
        selected_set.add(best_idx)
    return features[selected]


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_coreset_greedy_kcenter_matches_exact_bruteforce(seed):
    rng = np.random.default_rng(seed + 100)
    feats = rng.normal(size=(60, 16)).astype(np.float32)
    out = coreset_subsample(feats, ratio=0.35, mode="greedy_kcenter", seed=seed)
    expected = _reference_greedy_kcenter(feats, ratio=0.35, seed=seed)
    np.testing.assert_array_equal(out, expected)
    assert not np.shares_memory(out, feats)


def test_coreset_greedy_kcenter_handles_duplicates_without_reselection(monkeypatch):
    # All identical rows: must select n_select times without repeating indices
    ident = np.ones((12, 6), dtype=np.float32) * 5.0
    chosen_indices = []
    orig_argmax = np.argmax

    def spy_argmax(arr, *args, **kwargs):
        idx = int(orig_argmax(arr, *args, **kwargs))
        chosen_indices.append(idx)
        return idx

    monkeypatch.setattr(np, "argmax", spy_argmax)
    out_ident = coreset_subsample(ident, ratio=0.5, mode="greedy_kcenter", seed=0)
    assert out_ident.shape == (6, 6)
    assert np.isfinite(out_ident).all()
    assert len(chosen_indices) == 5  # n_select - 1 loop iterations
    assert len(set(chosen_indices)) == 5
    start_idx = int(np.random.default_rng(0).integers(0, len(ident)))
    assert start_idx not in chosen_indices

    # Large-magnitude duplicates + small-offset distinct rows: cancellation must not beat -1.0 sentinel
    chosen_indices.clear()
    base = np.array([1e8, 0.0, 0.0, 0.0], dtype=np.float64)
    feats = np.vstack([
        np.tile(base, (5, 1)),
        np.array([[1e8, 10.0, 0.0, 0.0], [1e8, 0.0, 10.0, 0.0]], dtype=np.float64),
    ])
    out_large = coreset_subsample(feats, ratio=4 / len(feats), mode="greedy_kcenter", seed=0)
    assert out_large.shape == (4, 4)
    assert len(set(chosen_indices)) == 3
    assert any(np.allclose(row, [1e8, 10.0, 0.0, 0.0]) for row in out_large)
    assert any(np.allclose(row, [1e8, 0.0, 10.0, 0.0]) for row in out_large)


def test_coreset_greedy_kcenter_4shot_cpu_speed():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(4096, 384)).astype(np.float32)
    t0 = time.perf_counter()
    out = coreset_subsample(feats, ratio=0.5, mode="greedy_kcenter", seed=0)
    elapsed = time.perf_counter() - t0
    assert out.shape == (2048, 384)
    assert elapsed < 5.0, f"4-shot coreset took {elapsed:.2f}s (expected < 5.0s)"
