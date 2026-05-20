"""
get_depth.py

Run UniK3D depth estimation on a single image and save:
  <out_dir>/<stem>_depth.npy   — raw depth map (float32, metres)
  <out_dir>/<stem>_depth.png   — side-by-side RGB / depth colormap

Usage:
    python get_depth.py --stem Actmap_MH3D_00000_8_1 \
        [--rgb_root /cluster/project/cvg/students/shangwu/GEN3C/assets/diffusion/dataset_all] \
        [--out_dir  ./depth_output]
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAM_H = 544
CAM_W = 720
CAM_F = 300.0
CX    = CAM_W / 2.0 - 0.5
CY    = CAM_H / 2.0 - 0.5


def find_rgb(stem: str, rgb_root: str) -> str | None:
    for shard in sorted(os.listdir(rgb_root)):
        shard_dir = os.path.join(rgb_root, shard)
        if not os.path.isdir(shard_dir):
            continue
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(shard_dir, stem + ext)
            if os.path.exists(p):
                return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem",     required=True, help="e.g. Actmap_MH3D_00000_8_1")
    parser.add_argument("--rgb_root", default="/cluster/project/cvg/students/shangwu/GEN3C/assets/diffusion/dataset_all")
    parser.add_argument("--out_dir",  default="./depth_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Locate RGB
    rgb_path = find_rgb(args.stem, args.rgb_root)
    if rgb_path is None:
        raise FileNotFoundError(f"RGB image not found for stem '{args.stem}' under {args.rgb_root}")
    print(f"Found RGB: {rgb_path}")

    rgb_bgr = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # Intrinsics scaled to image resolution
    fx = CAM_F * (W / CAM_W)
    fy = CAM_F * (H / CAM_H)
    cx = CX    * (W / CAM_W)
    cy = CY    * (H / CAM_H)
    K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Load UniK3D
    import torch
    from unik3d.models import UniK3D
    from unik3d.utils.camera import OPENCV  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device)
    model.eval()
    print(f"UniK3D loaded on {device}")

    # Depth estimation
    rgb_tensor = (
        torch.from_numpy(rgb.astype(np.float32))
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    cam_params = torch.tensor(
        [K[0,0], K[1,1], K[0,2], K[1,2]] + [0.0]*12,
        dtype=torch.float32,
    ).to(device)
    camera = eval("OPENCV")(params=cam_params)
    with torch.no_grad():
        pred = model.infer(rgb_tensor, camera)
    depth = pred["depth"].squeeze().cpu().numpy().astype(np.float32)

    # Save .npy
    npy_out = os.path.join(args.out_dir, args.stem + "_depth.npy")
    np.save(npy_out, depth)
    print(f"Saved depth array: {npy_out}  shape={depth.shape}  range=[{depth.min():.2f}, {depth.max():.2f}] m")

    # Save colormap PNG
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=11)
    axes[0].axis("off")
    im = axes[1].imshow(depth, cmap="plasma")
    axes[1].set_title("Depth (UniK3D)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="metres")
    plt.suptitle(args.stem, fontsize=10)
    plt.tight_layout()
    png_out = os.path.join(args.out_dir, args.stem + "_depth.png")
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved depth vis:   {png_out}")


if __name__ == "__main__":
    main()
