import json
import subprocess
from pathlib import Path

import pytest

from scripts.summarize_fastref_mvtec import (
    CVPR2026_MVTEC_REF,
    format_markdown_table,
    main,
    summarize_fastref_comparison,
)


def _write_metrics(run_dir: Path, seed: int, im_auroc: float, px_auroc: float | None = None, separate_segm: bool = False):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean_classification_au_roc": im_auroc,
        "mean_classification_ap": im_auroc,
        "mean_classification_f1": im_auroc,
    }
    if px_auroc is not None and not separate_segm:
        payload["mean_segmentation_au_roc"] = px_auroc
        payload["mean_au_pro"] = px_auroc - 0.05
        payload["mean_segmentation_f1"] = 0.6
    (run_dir / f"metrics_seed={seed}.json").write_text(json.dumps(payload))
    if px_auroc is not None and separate_segm:
        segm = {
            "mean_segmentation_au_roc": px_auroc,
            "mean_au_pro": px_auroc - 0.05,
            "mean_segmentation_f1": 0.6,
        }
        (run_dir / f"segm_metrics_seed={seed}.json").write_text(json.dumps(segm))


def test_cvpr2026_reference_values():
    assert CVPR2026_MVTEC_REF[1] == {"baseline_im": 96.4, "baseline_px": 95.8, "fastref_im": 97.0, "fastref_px": 96.5}
    assert CVPR2026_MVTEC_REF[2] == {"baseline_im": 96.7, "baseline_px": 96.0, "fastref_im": 97.2, "fastref_px": 96.7}
    assert CVPR2026_MVTEC_REF[4] == {"baseline_im": 97.1, "baseline_px": 96.4, "fastref_im": 97.4, "fastref_px": 96.8}


def test_summarize_fastref_comparison_and_markdown(tmp_path):
    model_root = tmp_path / "dinov2_vits14_448"
    # 1-shot: inline segm metrics
    _write_metrics(model_root / "1-shot_preprocess=force_no_mask_no_rotation", 0, 0.960, 0.955)
    _write_metrics(model_root / "1-shot_preprocess=force_no_mask_no_rotation", 1, 0.968, 0.961)
    _write_metrics(model_root / "1-shot_preprocess=force_no_mask_no_rotation_fastref", 0, 0.968, 0.963)
    _write_metrics(model_root / "1-shot_preprocess=force_no_mask_no_rotation_fastref", 1, 0.972, 0.967)
    # 2-shot: separate segm_metrics_seed file
    _write_metrics(model_root / "2-shot_preprocess=force_no_mask_no_rotation", 0, 0.967, 0.960, separate_segm=True)
    _write_metrics(model_root / "2-shot_preprocess=force_no_mask_no_rotation_fastref", 0, 0.972, 0.967, separate_segm=True)

    summary = summarize_fastref_comparison(str(tmp_path), shots=(1, 2, 4))
    s1 = summary["shots"][1]
    assert s1["baseline"]["im_auroc_mean"] == pytest.approx(96.4)
    assert s1["fastref"]["im_auroc_mean"] == pytest.approx(97.0)
    assert s1["delta_im_auroc"] == pytest.approx(0.6)
    assert s1["delta_px_auroc"] == pytest.approx(0.7)

    s2 = summary["shots"][2]
    assert s2["fastref"]["px_auroc_mean"] == pytest.approx(96.7)
    assert summary["shots"][4]["baseline"]["im_auroc_mean"] is None  # missing shot handled cleanly

    md = format_markdown_table(summary)
    assert "1-shot" in md and "97.00" in md and "96.50" in md and "+0.60" in md


def test_reproduce_script_exists_and_valid_bash():
    script = Path("scripts/reproduce_fastref_mvtec.sh")
    assert script.is_file()
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    text = script.read_text()
    assert "--use_fastref" in text
    assert "force_no_mask_no_rotation" in text
    assert "--fastref_alpha" in text and "--fastref_lambda" in text


def test_summarize_tag_and_cli_json_out(tmp_path, capsys):
    model_root = tmp_path / "dinov2_vits14_448"
    _write_metrics(model_root / "1-shot_preprocess=force_no_mask_no_rotation_mytag", 0, 0.950, 0.940)
    _write_metrics(model_root / "1-shot_preprocess=force_no_mask_no_rotation_fastref_mytag", 0, 0.970, 0.960)

    json_out = tmp_path / "summary.json"
    rc = main([
        "--results_root", str(tmp_path),
        "--shots", "1",
        "--tag", "mytag",
        "--json_out", str(json_out),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "1-shot" in captured and "+2.00" in captured

    loaded = json.loads(json_out.read_text())
    s1 = loaded["shots"]["1"]
    assert s1["delta_im_auroc"] == pytest.approx(2.0)
    assert s1["delta_px_auroc"] == pytest.approx(2.0)
