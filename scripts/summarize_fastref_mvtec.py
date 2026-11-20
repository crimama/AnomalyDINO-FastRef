"""Summarize MVTec-AD FastRef vs. baseline AnomalyDINO runs against CVPR 2026 Table 2."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

CVPR2026_MVTEC_REF: dict[int, dict[str, float]] = {
    1: {"baseline_im": 96.4, "baseline_px": 95.8, "fastref_im": 97.0, "fastref_px": 96.5},
    2: {"baseline_im": 96.7, "baseline_px": 96.0, "fastref_im": 97.2, "fastref_px": 96.7},
    4: {"baseline_im": 97.1, "baseline_px": 96.4, "fastref_im": 97.4, "fastref_px": 96.8},
}

_SEED_FILE_RE = re.compile(r"^metrics_seed=(\d+)\.json$")


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _collect_run_metrics(run_dir: Path) -> dict[str, Any]:
    im_aurocs: list[float] = []
    px_aurocs: list[float] = []
    au_pros: list[float] = []

    if run_dir.is_dir():
        seed_files: list[tuple[int, Path]] = []
        for path in run_dir.iterdir():
            match = _SEED_FILE_RE.match(path.name)
            if match and path.is_file():
                seed_files.append((int(match.group(1)), path))
        seed_files.sort(key=lambda item: item[0])

        for seed, metrics_path in seed_files:
            payload = json.loads(metrics_path.read_text())
            im_val = payload.get("mean_classification_au_roc")
            if im_val is not None:
                im_aurocs.append(float(im_val) * 100.0)

            px_val = payload.get("mean_segmentation_au_roc")
            pro_val = payload.get("mean_au_pro")
            if px_val is None or pro_val is None:
                segm_path = run_dir / f"segm_metrics_seed={seed}.json"
                if segm_path.is_file():
                    segm_payload = json.loads(segm_path.read_text())
                    if px_val is None:
                        px_val = segm_payload.get("mean_segmentation_au_roc")
                    if pro_val is None:
                        pro_val = segm_payload.get("mean_au_pro")

            if px_val is not None:
                px_aurocs.append(float(px_val) * 100.0)
            if pro_val is not None:
                au_pros.append(float(pro_val) * 100.0)

        num_seeds = len(seed_files)
    else:
        num_seeds = 0

    im_mean, im_std = _mean_std(im_aurocs)
    px_mean, px_std = _mean_std(px_aurocs)
    pro_mean, pro_std = _mean_std(au_pros)
    return {
        "run_dir": str(run_dir),
        "num_seeds": num_seeds,
        "im_auroc_mean": im_mean,
        "im_auroc_std": im_std,
        "px_auroc_mean": px_mean,
        "px_auroc_std": px_std,
        "au_pro_mean": pro_mean,
        "au_pro_std": pro_std,
    }


def summarize_fastref_comparison(
    results_root: str,
    model_dir: str = "dinov2_vits14_448",
    preprocess: str = "force_no_mask_no_rotation",
    shots: tuple[int, ...] = (1, 2, 4),
    tag: str | None = None,
) -> dict[str, Any]:
    """Aggregate baseline and FastRef metrics across shots and compare to CVPR 2026 Table 2."""
    root = Path(results_root) / model_dir
    tag_suffix = f"_{tag}" if tag else ""
    shot_summaries: dict[int, dict[str, Any]] = {}

    for shot in shots:
        base_name = f"{shot}-shot_preprocess={preprocess}"
        baseline_dir = root / f"{base_name}{tag_suffix}"
        fastref_dir = root / f"{base_name}_fastref{tag_suffix}"

        baseline = _collect_run_metrics(baseline_dir)
        fastref = _collect_run_metrics(fastref_dir)

        delta_im = (
            fastref["im_auroc_mean"] - baseline["im_auroc_mean"]
            if fastref["im_auroc_mean"] is not None and baseline["im_auroc_mean"] is not None
            else None
        )
        delta_px = (
            fastref["px_auroc_mean"] - baseline["px_auroc_mean"]
            if fastref["px_auroc_mean"] is not None and baseline["px_auroc_mean"] is not None
            else None
        )

        shot_summaries[shot] = {
            "baseline": baseline,
            "fastref": fastref,
            "delta_im_auroc": delta_im,
            "delta_px_auroc": delta_px,
            "cvpr2026_ref": CVPR2026_MVTEC_REF.get(shot),
        }

    return {
        "results_root": results_root,
        "model_dir": model_dir,
        "preprocess": preprocess,
        "tag": tag,
        "shots": shot_summaries,
    }


def _fmt_metric(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def format_markdown_table(summary: dict[str, Any]) -> str:
    """Format a summary dictionary as a GitHub-flavored Markdown comparison table."""
    lines = [
        "| Shot | Base Im | Base Px | FastRef Im | FastRef Px | Δ Im | Δ Px | CVPR26 Base (Im/Px) | CVPR26 FastRef (Im/Px) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for shot, entry in summary["shots"].items():
        base = entry["baseline"]
        fast = entry["fastref"]
        ref = entry.get("cvpr2026_ref")
        ref_base = f"{ref['baseline_im']:.2f} / {ref['baseline_px']:.2f}" if ref else "-"
        ref_fast = f"{ref['fastref_im']:.2f} / {ref['fastref_px']:.2f}" if ref else "-"
        lines.append(
            f"| {shot}-shot "
            f"| {_fmt_metric(base['im_auroc_mean'])} "
            f"| {_fmt_metric(base['px_auroc_mean'])} "
            f"| {_fmt_metric(fast['im_auroc_mean'])} "
            f"| {_fmt_metric(fast['px_auroc_mean'])} "
            f"| {_fmt_metric(entry['delta_im_auroc'], signed=True)} "
            f"| {_fmt_metric(entry['delta_px_auroc'], signed=True)} "
            f"| {ref_base} "
            f"| {ref_fast} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize MVTec-AD FastRef vs. AnomalyDINO baseline results."
    )
    parser.add_argument("--results_root", type=str, default="results_MVTec")
    parser.add_argument("--model_dir", type=str, default="dinov2_vits14_448")
    parser.add_argument("--preprocess", type=str, default="force_no_mask_no_rotation")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args(argv)

    summary = summarize_fastref_comparison(
        results_root=args.results_root,
        model_dir=args.model_dir,
        preprocess=args.preprocess,
        shots=tuple(args.shots),
        tag=args.tag or None,
    )
    print(format_markdown_table(summary))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
