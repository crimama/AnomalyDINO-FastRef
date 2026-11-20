import subprocess
import sys

import pytest

import run_anomalydino
import run_anomalydino_batched


def parse(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return module.parse_args()


def test_defaults(monkeypatch):
    args = parse(run_anomalydino, monkeypatch)
    assert args.shots == [1]
    assert args.preprocess == "agnostic"
    assert args.mask_ref_images is False
    assert args.faiss_on_cpu is False


@pytest.mark.parametrize("argv, expected", [([], False), (["--mask_ref_images"], True), (["--no-mask_ref_images"], False)])
def test_mask_ref_images_flag(monkeypatch, argv, expected):
    assert parse(run_anomalydino, monkeypatch, *argv).mask_ref_images is expected


def test_multiple_shots(monkeypatch):
    assert parse(run_anomalydino, monkeypatch, "--shots", "1", "2", "4").shots == [1, 2, 4]


@pytest.mark.parametrize("module, argv", [
    (run_anomalydino, ["--preprocess", "agnostc"]),
    (run_anomalydino, ["--knn_metric", "cosine"]),
    (run_anomalydino, ["--mask_ref_images", "False"]),  # boolean flags do not take values
    (run_anomalydino_batched, ["--model_size", "xl"]),
])
def test_invalid_arguments_are_rejected(monkeypatch, module, argv):
    with pytest.raises(SystemExit):
        parse(module, monkeypatch, *argv)


def test_batched_args(monkeypatch):
    args = parse(run_anomalydino_batched, monkeypatch, "--model_size", "S", "--no-save_examples")
    assert args.model_size == "s"
    assert args.save_examples is False


def test_layer_argument(monkeypatch):
    assert parse(run_anomalydino, monkeypatch).layer is None
    assert parse(run_anomalydino, monkeypatch, "--layer", "9", "10").layer == [9, 10]
    assert parse(run_anomalydino, monkeypatch, "--layer", "-2").layer == [-2]
    assert parse(run_anomalydino_batched, monkeypatch, "--layer", "9").layer == [9]


@pytest.mark.parametrize("module", [run_anomalydino, run_anomalydino_batched])
def test_num_threads_argument(monkeypatch, module):
    assert parse(module, monkeypatch).num_threads == 4
    assert parse(module, monkeypatch, "--num_threads", "0").num_threads == 0


def test_fastref_arg_defaults(monkeypatch):
    args = parse(run_anomalydino, monkeypatch)
    assert args.use_fastref is False
    assert args.fastref_alpha == 0.5
    assert args.fastref_lambda == 0.1
    assert args.fastref_iters == 2
    assert args.fastref_ot_iters == 10
    assert args.fastref_ot_eps == 0.1
    assert args.fastref_score_mode == "reconstruction"
    assert args.fastref_ot_scale == "row_normalized"
    assert args.fastref_coreset_mode == "greedy_kcenter"
    assert args.fastref_ridge == 0.0


@pytest.mark.parametrize("argv, expected", [([], False), (["--use_fastref"], True), (["--no-use_fastref"], False)])
def test_use_fastref_flag(monkeypatch, argv, expected):
    assert parse(run_anomalydino, monkeypatch, *argv).use_fastref is expected


@pytest.mark.parametrize("argv", [
    ["--fastref_score_mode", "invalid"],
    ["--fastref_ot_scale", "invalid"],
    ["--fastref_coreset_mode", "invalid"],
])
def test_invalid_fastref_arguments_are_rejected(monkeypatch, argv):
    with pytest.raises(SystemExit):
        parse(run_anomalydino, monkeypatch, *argv)


def test_build_fastref_config_syncs_metric_and_params(monkeypatch):
    args = parse(
        run_anomalydino, monkeypatch,
        "--use_fastref", "--knn_metric", "L2",
        "--fastref_alpha", "0.3", "--fastref_lambda", "0.2",
        "--fastref_iters", "3", "--fastref_ot_iters", "20",
        "--fastref_ot_eps", "0.05", "--fastref_score_mode", "nn",
        "--fastref_ot_scale", "unscaled", "--fastref_coreset_mode", "approx_kcenter",
        "--fastref_ridge", "0.01",
    )
    cfg = run_anomalydino.build_fastref_config(args, seed=2)
    assert cfg.metric == "euclidean"
    assert cfg.alpha == 0.3
    assert cfg.lam == 0.2
    assert cfg.num_iters == 3
    assert cfg.sinkhorn_iters == 20
    assert cfg.sinkhorn_eps == 0.05
    assert cfg.score_mode == "nn"
    assert cfg.ot_scale == "unscaled"
    assert cfg.coreset_mode == "approx_kcenter"
    assert cfg.ridge == 0.01
    assert cfg.seed == 2

    args_cos = parse(run_anomalydino, monkeypatch, "--use_fastref")
    assert run_anomalydino.build_fastref_config(args_cos, seed=0).metric == "cosine"


def test_build_results_dir_isolates_fastref_from_baseline(monkeypatch):
    base_args = parse(run_anomalydino, monkeypatch, "--preprocess", "force_no_mask_no_rotation")
    fr_args = parse(run_anomalydino, monkeypatch, "--preprocess", "force_no_mask_no_rotation", "--use_fastref")
    fr_tagged = parse(run_anomalydino, monkeypatch, "--preprocess", "force_no_mask_no_rotation", "--use_fastref", "--tag", "exp1")

    base_dir = run_anomalydino.build_results_dir(base_args, shot=1)
    fr_dir = run_anomalydino.build_results_dir(fr_args, shot=1)
    fr_tagged_dir = run_anomalydino.build_results_dir(fr_tagged, shot=1)

    assert base_dir == "results_MVTec/dinov2_vits14_448/1-shot_preprocess=force_no_mask_no_rotation"
    assert fr_dir != base_dir
    assert fr_dir.endswith("_fastref")
    assert fr_tagged_dir == fr_dir + "_exp1"


def test_resolve_device():
    assert run_anomalydino.resolve_device("cpu") == "cpu"
    assert run_anomalydino.resolve_device("cuda:0") == "cuda"
    assert run_anomalydino.resolve_device("cuda") == "cuda"


@pytest.mark.parametrize("argv", [
    ["--use_fastref", "--fastref_alpha", "2.0"],
    ["--use_fastref", "--fastref_lambda", "nan"],
    ["--use_fastref", "--fastref_lambda", "inf"],
    ["--use_fastref", "--fastref_ot_eps", "nan"],
    ["--use_fastref", "--fastref_ot_eps", "0.0"],
    ["--use_fastref", "--fastref_ridge", "-0.1"],
    ["--use_fastref", "--fastref_iters", "-1"],
    ["--use_fastref", "--fastref_ot_iters", "0"],
])
def test_validate_args_rejects_invalid_fastref_values(monkeypatch, argv):
    args = parse(run_anomalydino, monkeypatch, *argv)
    with pytest.raises(ValueError):
        run_anomalydino.validate_args(args)


@pytest.mark.parametrize("argv", [
    ["--use_fastref"],
    [],
    ["--fastref_alpha", "2.0", "--fastref_lambda", "nan"],
])
def test_validate_args_passes_when_valid_or_fastref_disabled(monkeypatch, argv):
    args = parse(run_anomalydino, monkeypatch, *argv)
    run_anomalydino.validate_args(args)


def test_cli_main_validates_fastref_before_model_load_or_dirs(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            run_anomalydino.__file__,
            "--use_fastref",
            "--fastref_alpha",
            "2.0",
            "--data_root",
            str(tmp_path / "missing_data"),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr
    assert "alpha" in proc.stderr
    assert list(tmp_path.iterdir()) == []
