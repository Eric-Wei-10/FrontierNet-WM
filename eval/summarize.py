#!/usr/bin/env python3
"""
Aggregate exploration statistics across multiple runs of the same scene.

Reads all exploration_state_with_volume.json files matching a glob pattern,
extracts the final mapped_vol from each, and reports mean ± std coverage
relative to a reference voxel grid.

Usage:
    python eval/summarize.py \
        --json_glob "output/detr_multi_10000_876_pt*_run*/exploration_state_with_volume.json" \
        --voxel_grid eval_data/voxel_grid/000876-voxel_grid.ply \
        --scene 876
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


def load_total_volume(ply_path: str) -> Tuple[float, float, int]:
    vg = o3d.io.read_voxel_grid(ply_path)
    voxel_size = float(vg.voxel_size)
    n_vox = len(vg.get_voxels())
    return voxel_size**3 * n_vox, voxel_size, n_vox


def final_mapped_vol(json_path: Path) -> float:
    entries = [json.loads(l) for l in json_path.read_text().splitlines() if l.strip()]
    if not entries:
        return 0.0
    return float(entries[-1].get("mapped_vol") or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_glob", required=True,
                    help="Glob for exploration_state_with_volume.json files")
    ap.add_argument("--voxel_grid", required=True)
    ap.add_argument("--scene", default="", help="Scene label for display")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-run breakdown")
    ap.add_argument("--save", default=None,
                    help="Write metrics JSON to this path (parent dirs created automatically)")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.json_glob))
    if not files:
        print(f"No files matched: {args.json_glob}", file=sys.stderr)
        sys.exit(1)

    total_vol, voxel_size, n_vox = load_total_volume(args.voxel_grid)
    ratios: List[float] = []
    runs: List[dict] = []

    for f in files:
        vol = final_mapped_vol(f)
        ratio = vol / total_vol if total_vol > 0 else 0.0
        ratios.append(ratio)
        runs.append({"name": f.parent.name, "coverage": round(ratio, 6), "mapped_vol_m3": round(vol, 4)})
        if args.verbose:
            print(f"  {f.parent.name}: {ratio:.2%}  ({vol:.3f} / {total_vol:.3f} m³)")

    if not ratios:
        print("No valid runs found.")
        sys.exit(1)

    label = f"Scene {args.scene}" if args.scene else "Summary"
    print(f"\n{'='*52}")
    print(f" {label}  |  n={len(ratios)}  |  voxel_size={voxel_size:.3f}m  |  total={total_vol:.2f}m³")
    print(f"{'='*52}")
    print(f"  Final coverage : {np.mean(ratios):.2%} ± {np.std(ratios):.2%}")
    print(f"  Min / Max      : {min(ratios):.2%} / {max(ratios):.2%}")
    print(f"  Median         : {np.median(ratios):.2%}")
    print(f"{'='*52}\n")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "scene": args.scene,
            "n_runs": len(ratios),
            "coverage_mean": round(float(np.mean(ratios)), 6),
            "coverage_std": round(float(np.std(ratios)), 6),
            "coverage_median": round(float(np.median(ratios)), 6),
            "coverage_min": round(float(min(ratios)), 6),
            "coverage_max": round(float(max(ratios)), 6),
            "total_vol_m3": round(total_vol, 4),
            "voxel_size_m": voxel_size,
            "n_voxels": n_vox,
            "runs": runs,
        }
        out.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
