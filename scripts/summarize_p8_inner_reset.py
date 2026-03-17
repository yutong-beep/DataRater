#!/usr/bin/env python3
"""
Summarize the p8 inner-reset experiment suite into JSON and Markdown.

This compares:
1. original random_init
2. carryover
3. checkpoint_bank

The script is tolerant to partially finished suites: missing runs are reported
as pending instead of failing the whole summary.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize p8 inner-reset results")
    p.add_argument("--random_results", required=True, help="results.json for random_init run")
    p.add_argument("--carryover_results", required=True, help="results.json for carryover run")
    p.add_argument("--bank_results", required=True, help="results.json for checkpoint_bank run")
    p.add_argument("--output_dir", required=True, help="Directory to write summary.json / summary.md")
    p.add_argument("--label", default="p8_inner_reset_suite", help="Short label for the summary")
    return p.parse_args()


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _extract_random_mse(results: Dict[str, Any]) -> Optional[float]:
    retrained_random = results.get("retrained_random")
    if isinstance(retrained_random, dict):
        metrics = retrained_random.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("mse") is not None:
            return float(metrics["mse"])

    random_baseline = results.get("random_baseline")
    if isinstance(random_baseline, dict):
        metrics = random_baseline.get("best_metrics", {})
        if isinstance(metrics, dict) and metrics.get("mse") is not None:
            return float(metrics["mse"])

    return None


def _extract_row(label: str, results_path: str) -> Dict[str, Any]:
    results = _load_json_if_exists(results_path)
    if results is None:
        return {
            "label": label,
            "status": "pending",
            "results_path": results_path,
        }

    baseline_mse = _safe_float(results["baseline"]["best_metrics"]["mse"])
    retrained_mse = _safe_float(results["retrained"]["best_metrics"]["mse"])
    random_mse = _extract_random_mse(results)
    keep_ratio = results.get("retrained_datarater", {}).get("keep_ratio")
    keep_act = results.get("filtering", {}).get("keep_act")

    row = {
        "label": label,
        "status": "done",
        "results_path": results_path,
        "run_dir": os.path.dirname(results_path),
        "baseline_mse": baseline_mse,
        "retrained_mse": retrained_mse,
        "random_mse": random_mse,
        "delta_vs_baseline": None if baseline_mse is None or retrained_mse is None else retrained_mse - baseline_mse,
        "delta_vs_random": None if random_mse is None or retrained_mse is None else retrained_mse - random_mse,
        "keep_ratio": keep_ratio,
        "keep_act": keep_act,
    }
    return row


def _fmt(v: Optional[float]) -> str:
    return "pending" if v is None else f"{v:.4f}"


def _render_md(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# {summary['label']}")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at_utc']}")
    lines.append("")
    lines.append("| Run | Status | Baseline MSE | Retrained MSE | Random MSE | Delta vs Baseline | Delta vs Random |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["runs"]:
        lines.append(
            f"| {row['label']} | {row['status']} | {_fmt(row.get('baseline_mse'))} | "
            f"{_fmt(row.get('retrained_mse'))} | {_fmt(row.get('random_mse'))} | "
            f"{_fmt(row.get('delta_vs_baseline'))} | {_fmt(row.get('delta_vs_random'))} |"
        )
    lines.append("")
    lines.append("## Notes")
    for row in summary["runs"]:
        lines.append(f"- `{row['label']}`: `{row['results_path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = [
        _extract_row("random_init", args.random_results),
        _extract_row("carryover", args.carryover_results),
        _extract_row("checkpoint_bank", args.bank_results),
    ]

    summary = {
        "label": args.label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
    }

    json_path = os.path.join(args.output_dir, "summary.json")
    md_path = os.path.join(args.output_dir, "summary.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(md_path, "w") as f:
        f.write(_render_md(summary))

    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
