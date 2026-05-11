"""Random/sample QA runner for ResPlan wall segmentation.

Examples:
    python test_wall_segmentation_samples.py --random 10 --seed 42
    python test_wall_segmentation_samples.py --indices 0 1 2 10 100
    python test_wall_segmentation_samples.py --random 10 --no-save-plots
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import zipfile
from typing import Any, Dict, List

import resplan_utils as ru


def load_plans(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            with zf.open("ResPlan.pkl") as f:
                return pickle.load(f)

    with open(path, "rb") as f:
        return pickle.load(f)


def choose_indices(total: int, args: argparse.Namespace) -> List[int]:
    if args.indices:
        return args.indices

    count = min(args.random, total)
    rng = random.Random(args.seed)
    return sorted(rng.sample(range(total), count))


def print_report(idx: int, report: Dict[str, Any]) -> None:
    opening_count = report.get("opening_count", 0)
    fallback_opening_ids = report.get("fallback_opening_ids", [])
    paired_count = report.get("paired_count", "-")
    unpaired_count = report.get("unpaired_count", "-")
    dangling_count = report.get("dangling_wall_count", 0)
    attached_opening_count = report.get("attached_opening_count")
    openings_text = (
        f"{attached_opening_count}/{opening_count}"
        if attached_opening_count is not None
        else str(opening_count)
    )
    print(
        f"{idx:>5} | "
        f"segments={report['segment_count']:<4} "
        f"paired={paired_count!s:<4} "
        f"unpaired={unpaired_count!s:<4} "
        f"dangling={dangling_count:<4} "
        f"splits={report['split_intersection_part_count']:<4} "
        f"openings={openings_text:<7} "
        f"fallback={len(fallback_opening_ids):<3} "
        f"unsplit={len(report['unsplit_intersections']):<3} "
        f"short={len(report['too_short_segments']):<3} "
        f"ok={report['ok']}"
    )


def save_debug_plot(plan: Dict[str, Any],
                    segments: List[Dict[str, Any]],
                    out_dir: str,
                    idx: int) -> str:
    if ru.plt is None:
        raise ImportError("Saving plots requires matplotlib.")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wall_segments_debug_{idx}.png")
    fig, ax = ru.plt.subplots(figsize=(9, 9))
    ru.plot_wall_segmentation_debug(
        plan,
        segments,
        ax=ax,
        title=f"Wall segmentation debug {idx}",
    )
    fig.savefig(out_path, dpi=200)
    ru.plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ResPlan.zip",
                        help="Path to ResPlan.zip or ResPlan.pkl.")
    parser.add_argument("--random", type=int, default=10,
                        help="Number of random plans to sample when --indices is not set.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible sampling.")
    parser.add_argument("--indices", type=int, nargs="*",
                        help="Explicit plan indices to test.")
    parser.add_argument("--save-plots", action="store_true", default=True,
                        help="Save debug visualization images. Enabled by default.")
    parser.add_argument("--no-save-plots", action="store_false", dest="save_plots",
                        help="Disable saving debug visualization images.")
    parser.add_argument("--out-dir", default="assets/wall_segmentation_checks",
                        help="Directory for debug images.")
    args = parser.parse_args()

    plans = load_plans(args.data)
    indices = choose_indices(len(plans), args)

    print(f"Loaded {len(plans)} plans from {args.data}")
    print(f"Testing indices: {indices}")
    print(
        "index | segments paired unpaired dangling splits openings fallback unsplit short ok"
    )

    failed = []
    for idx in indices:
        plan = plans[idx]
        segments = ru.split_wall_segments(plan)
        report = ru.validate_wall_segmentation(plan, segments)
        print_report(idx, report)

        if not report["ok"] or report.get("fallback_opening_ids", []):
            failed.append(idx)

        if args.save_plots:
            out_path = save_debug_plot(plan, segments, args.out_dir, idx)
            print(f"      plot: {out_path}")

    if failed:
        print(f"\nPlans needing QA: {failed}")
    else:
        print("\nNo automatic QA warnings for this sample.")


if __name__ == "__main__":
    main()
