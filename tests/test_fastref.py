import numpy as np
import pytest
import torch

from src.fastref import FastRefConfig, coreset_subsample, sinkhorn_transport


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
