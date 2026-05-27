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
import glob as _glob
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


def load_entries(json_path: Path):
    try:
        return [json.loads(l) for l in json_path.read_text().splitlines() if l.strip()]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  WARNING: skipping {json_path} (corrupt JSON: {e})", file=sys.stderr)
        return None


def vol_at_fraction(entries: list, frac: float) -> float:
    """mapped_vol at frac of the way through the entries (0.0 = start, 1.0 = end)."""
    if not entries:
        return 0.0
    idx = int(frac * (len(entries) - 1))
    # Walk backward in case this exact entry lacks mapped_vol
    for i in range(idx, -1, -1):
        v = entries[i].get("mapped_vol")
        if v is not None:
            return float(v)
    return 0.0


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
    ap.add_argument("--success_threshold", type=float, default=0.4,
                    help="Coverage threshold for a run to count as successful (default: 0.4)")
    args = ap.parse_args()

    files = sorted(Path(p) for p in _glob.glob(args.json_glob))
    if not files:
        print(f"No files matched: {args.json_glob}", file=sys.stderr)
        sys.exit(1)

    total_vol, voxel_size, n_vox = load_total_volume(args.voxel_grid)
    ratios: List[float] = []
    ratios25: List[float] = []
    ratios50: List[float] = []
    runs: List[dict] = []

    for f in files:
        entries = load_entries(f)
        if entries is None:
            continue
        vol = float(entries[-1].get("mapped_vol") or 0.0)
        v25 = vol_at_fraction(entries, 0.25)
        v50 = vol_at_fraction(entries, 0.50)
        ratio    = vol / total_vol if total_vol > 0 else 0.0
        ratio25  = v25 / total_vol if total_vol > 0 else 0.0
        ratio50  = v50 / total_vol if total_vol > 0 else 0.0
        success  = ratio >= args.success_threshold
        ratios.append(ratio)
        ratios25.append(ratio25)
        ratios50.append(ratio50)
        runs.append({
            "name": f.parent.name,
            "coverage": round(ratio, 6),
            "coverage@25": round(ratio25, 6),
            "coverage@50": round(ratio50, 6),
            "mapped_vol_m3": round(vol, 4),
            "success": success,
        })
        if args.verbose:
            tag = "OK" if success else "FAIL"
            print(f"  [{tag}] {f.parent.name}: {ratio:.2%}  "
                  f"(@25%={ratio25:.2%}  @50%={ratio50:.2%}  "
                  f"vol={vol:.1f}/{total_vol:.1f} m³)")

    if not ratios:
        print("No valid runs found.")
        sys.exit(1)

    thr = args.success_threshold
    n_total = len(ratios)
    succ_mask = [r >= thr for r in ratios]
    successful   = [r   for r, s in zip(ratios,   succ_mask) if s]
    successful25 = [r25 for r25, s in zip(ratios25, succ_mask) if s]
    successful50 = [r50 for r50, s in zip(ratios50, succ_mask) if s]
    n_success = len(successful)
    success_rate = n_success / n_total

    def _stats(vals):
        if not vals:
            return None, None, None, None, None
        return (round(float(np.mean(vals)),   6),
                round(float(np.std(vals)),    6),
                round(float(np.median(vals)), 6),
                round(float(min(vals)),       6),
                round(float(max(vals)),       6))

    label = f"Scene {args.scene}" if args.scene else "Summary"
    print(f"\n{'='*60}")
    print(f" {label}  |  n={n_total}  |  voxel_size={voxel_size:.3f}m  |  total={total_vol:.2f}m³")
    print(f"{'='*60}")
    print(f"  Success rate   : {success_rate:.2%}  ({n_success}/{n_total}, threshold={thr:.0%})")
    if successful:
        print(f"  vox@25 : {np.mean(successful25):.2%} ± {np.std(successful25):.2%}")
        print(f"  vox@50 : {np.mean(successful50):.2%} ± {np.std(successful50):.2%}")
        print(f"  vox@100 : {np.mean(successful):.2%} ± {np.std(successful):.2%}")
        print(f"  Min / Max      : {min(successful):.2%} / {max(successful):.2%}")
        print(f"  Median         : {np.median(successful):.2%}")
    else:
        print(f"  Coverage (succ): N/A  (no successful runs)")
    print(f"{'='*60}\n")

    if args.save:
        mean, std, med, mn, mx = _stats(successful)
        m25, s25, *_ = _stats(successful25)
        m50, s50, *_ = _stats(successful50)
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "scene": args.scene,
            "n_runs": n_total,
            "success_threshold": thr,
            "success_rate": round(success_rate, 6),
            "n_success": n_success,
            "coverage_mean": mean,
            "coverage_std": std,
            "coverage_median": med,
            "coverage_min": mn,
            "coverage_max": mx,
            "vox25_mean": m25,
            "vox25_std": s25,
            "vox50_mean": m50,
            "vox50_std": s50,
            "total_vol_m3": round(total_vol, 4),
            "voxel_size_m": voxel_size,
            "n_voxels": n_vox,
            "runs": runs,
        }
        out.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
