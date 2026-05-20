"""
inspect_gmm_samples.py

Run the Bayesian GMM on the first --n (default 10) frontier samples in
--data_dir and print, for each active cluster:

    - image-plane centre  (u, v)   [pixels]
    - z  (depth of 3-D centre)     [metres]
    - w  (component weight)
    - 3×3 covariance matrix        [m²]

Usage:
    python inspect_gmm_samples.py [--data_dir ./data] [--n 10] [--seed 42]
"""

import argparse
import os

import cv2
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

# ── Camera constants (must match fit_gmm.py) ─────────────────────────────────
CAM_H = 544
CAM_W = 720
CAM_F = 300.0
CX    = CAM_W / 2.0 - 0.5
CY    = CAM_H / 2.0 - 0.5

K_MAX         = 10
N_GMM_SAMPLES = 5000
MIN_VAL       = 0.01
WEIGHT_THRESH = 0.02


def _get_intrinsics(H, W):
    fx = CAM_F * (W / CAM_W)
    fy = CAM_F * (H / CAM_H)
    cx = CX    * (W / CAM_W)
    cy = CY    * (H / CAM_H)
    return fx, fy, cx, cy


def project_to_2d(means_3d, fx, fy, cx, cy):
    x, y, z = means_3d[:, 0], means_3d[:, 1], means_3d[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.stack([u, v], axis=1)


def sample_coords_3d(frontier, depth, n_samples, min_val, fx, fy, cx, cy):
    vs, us = np.where(frontier > min_val)
    if len(us) < 2:
        return None
    d_vals = depth[vs, us].astype(np.float64)
    valid  = d_vals > 0
    vs, us, d_vals = vs[valid], us[valid], d_vals[valid]
    if len(us) < 2:
        return None
    weights = frontier[vs, us].astype(np.float64)
    weights /= weights.sum()
    idx = np.random.choice(len(us), size=n_samples, replace=True, p=weights)
    u_s = us[idx].astype(np.float64) + np.random.uniform(-0.5, 0.5, n_samples)
    v_s = vs[idx].astype(np.float64) + np.random.uniform(-0.5, 0.5, n_samples)
    d_s = d_vals[idx] + np.random.uniform(-0.01, 0.01, n_samples)
    d_s = np.maximum(d_s, 1e-3)
    x = (u_s - cx) * d_s / fx
    y = (v_s - cy) * d_s / fy
    return np.stack([x, y, d_s], axis=1)


def process_sample(stem, frontier, depth48, seed):
    H, W = frontier.shape
    fx, fy, cx, cy = _get_intrinsics(H, W)

    if depth48.shape != (H, W):
        depth48 = cv2.resize(depth48, (W, H), interpolation=cv2.INTER_LINEAR)

    coords = sample_coords_3d(frontier, depth48, N_GMM_SAMPLES,
                               MIN_VAL, fx, fy, cx, cy)
    if coords is None:
        return None

    gmm = BayesianGaussianMixture(
        n_components=K_MAX,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0 / K_MAX,
        random_state=seed,
        max_iter=500,
    )
    gmm.fit(coords)

    active_idx = np.where(gmm.weights_ > WEIGHT_THRESH)[0]
    centres_3d = gmm.means_[active_idx]                       # (K, 3)
    weights    = gmm.weights_[active_idx]                     # (K,)
    covs       = gmm.covariances_[active_idx]                 # (K, 3, 3)
    centres_2d = project_to_2d(centres_3d, fx, fy, cx, cy)   # (K, 2)

    return centres_2d, centres_3d, weights, covs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--out_dir",  default=os.path.join(os.path.dirname(__file__), "gmm_output"))
    parser.add_argument("--n",    type=int, default=10, help="Number of samples to inspect")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)

    frontier_files = sorted(
        f for f in os.listdir(args.data_dir) if f.endswith("_frontier.npy")
    )[:args.n]

    print(f"Inspecting {len(frontier_files)} sample(s) from {args.data_dir}\n")

    for fname in frontier_files:
        stem     = fname.replace("_frontier.npy", "")
        fpath    = os.path.join(args.data_dir, fname)
        dpath    = os.path.join(args.data_dir, stem + "_depth48.npy")

        frontier = np.load(fpath)
        if frontier.ndim == 3:
            frontier = frontier.squeeze(0)

        if not os.path.exists(dpath):
            print(f"[{stem}]  no depth48 file — skipping")
            continue
        depth48 = np.load(dpath)

        result = process_sample(stem, frontier, depth48, args.seed)

        sep = "─" * 64
        print(sep)
        print(f"Sample: {stem}")
        print(sep)

        if result is None:
            print("  Not enough active pixels — skipped.\n")
            continue

        centres_2d, centres_3d, weights, covs = result
        K = len(weights)
        print(f"  Active clusters K = {K}\n")

        for i, (uv, xyz, w, cov) in enumerate(
                zip(centres_2d, centres_3d, weights, covs)):
            print(f"  Cluster #{i}")
            print(f"    Image-plane  (u, v)  = ({uv[0]:7.1f}, {uv[1]:7.1f}) px")
            print(f"    z  (depth)           = {xyz[2]:.3f} m")
            print(f"    w  (weight)          = {w:.4f}")
            print("    Covariance (3×3, camera-space) [m²]:")
            for row in cov:
                print("      " + "  ".join(f"{v:+.6f}" for v in row))
            print()

        # Save per-sample results
        out_path = os.path.join(args.out_dir, stem + "_gmm.npz")
        np.savez(
            out_path,
            centres_2d  = centres_2d,   # (K, 2)  u,v  [pixels]
            centres_3d  = centres_3d,   # (K, 3)  x,y,z [m]
            z           = centres_3d[:, 2],  # (K,)  depth [m]
            weights     = weights,      # (K,)
            covariances = covs,         # (K, 3, 3)
        )
        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
