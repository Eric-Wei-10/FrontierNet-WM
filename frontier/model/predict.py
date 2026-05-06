import numpy as np
import torch
import logging
import cv2
from PIL import Image

import sys

_DPT_REPO = "/cluster/project/cvg/students/shangwu/dpt_distillation_repo"
_UNET_REPO = "/cluster/project/cvg/students/shangwu/Pytorch-UNet"


def load_model(
    path,
    num_classes=11,
    use_depth=False,
    model_type="dpt",
    vit_depth=8,
    feature_layers=None,
    num_queries=10,
    detr_aux_depth=False,
):
    """
    Load a frontier-detection model.

    Args:
        path:        Path to checkpoint file.
        num_classes: Number of segmentation classes (not used for detr_unet).
        use_depth:   Whether the model uses a depth channel (UNet only).
        model_type:    "dpt"       — ViT+DPT, RGB-only, df_seg head
                       "unet"      — ResNet34+UNet, supports RGB-D, df_seg head
                       "detr_unet" — ResNet34+UNet, RGB-only, DETR sparse-set head
        num_queries:   Number of DETR slot queries (detr_unet only).
        detr_aux_depth: Whether the checkpoint was trained with an auxiliary dense
                        depth head (detr_unet only).  Must match the training flag
                        so the state-dict keys align correctly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {model_type} model from: {path}")
    logging.info(f"Using device {device}")

    if model_type == "dpt":
        if _DPT_REPO not in sys.path:
            sys.path.insert(0, _DPT_REPO)
        from model import DepthEstimationModel
        model = DepthEstimationModel(
            head_mode="df_seg",
            num_classes=num_classes,
            vit_depth=vit_depth,
            feature_layers=feature_layers,
        )

    elif model_type == "unet":
        if _UNET_REPO not in sys.path:
            sys.path.insert(0, _UNET_REPO)
        from unet import TwoHeadUnet
        in_channels = 4 if use_depth else 3
        model = TwoHeadUnet(
            classes=num_classes,
            in_channels=in_channels,
            head_config="df_seg",
        )

    elif model_type == "detr_unet":
        if _UNET_REPO not in sys.path:
            sys.path.insert(0, _UNET_REPO)
        from unet import TwoHeadUnet
        model = TwoHeadUnet(
            classes=1,
            in_channels=3,
            head_config="detr",
            num_queries=num_queries,
            detr_aux_depth=detr_aux_depth,
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected 'dpt', 'unet', or 'detr_unet'."
        )

    model.eval()
    model.to(device=device)

    checkpoint = torch.load(path, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    logging.info("Model loaded! Num classes: {}".format(num_classes))
    return model


def predict_from_img(
    net,
    rgb_img,
    depth_img,
    device,
    scale_factor=0.5,   # kept for API compatibility, not used
    use_depth=True,
    input_img_size=[224, 224],
    model_type="dpt",
    max_depth=10.0,
):
    """
    Preprocess an RGB (+ optional depth) image and run the model.

    Preprocessing matches the training pipeline (data.py):
      - Direct resize to (input_h, input_w) with BILINEAR interpolation
      - Normalize RGB to [0, 1] by dividing by 255
      - For UNet with depth: depth is clipped to [0, max_depth] and normalized to [0, 1],
        then concatenated as a 4th channel
      - No center crop, no ImageNet mean/std normalization

    Args:
        net:           The loaded model.
        rgb_img:       HxWx3 uint8 numpy array or PIL Image.
        depth_img:     HxW float32 numpy array (metric depth in metres). Used only for
                       model_type="unet" with use_depth=True.
        device:        torch.device to run inference on.
        scale_factor:  Kept for API compatibility; not used.
        use_depth:     Whether to pass depth as a 4th channel (UNet only).
        input_img_size: (height, width) expected by the model.
        model_type:    "dpt" or "unet".
        max_depth:     Depth clipping value for normalization (UNet only).

    Returns:
        (df_pred, cls_pred) tuple of tensors: [B,1,H,W] and [B,n_classes,H,W].
    """
    input_h, input_w = input_img_size

    # --- RGB preprocessing ---
    if isinstance(rgb_img, np.ndarray):
        rgb_pil = Image.fromarray(rgb_img.astype(np.uint8) if rgb_img.dtype != np.uint8 else rgb_img)
    else:
        rgb_pil = rgb_img
    rgb_pil = rgb_pil.convert("RGB").resize((input_w, input_h), Image.Resampling.BILINEAR)

    rgb_np = np.asarray(rgb_pil, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
    rgb_np = rgb_np.transpose(2, 0, 1)                       # [3, H, W]

    # --- Depth preprocessing (UNet only) ---
    if model_type == "unet" and use_depth and depth_img is not None:
        depth_resized = cv2.resize(
            depth_img.astype(np.float32), (input_w, input_h),
            interpolation=cv2.INTER_LINEAR,
        )
        depth_norm = np.clip(depth_resized, 0.0, max_depth) / max_depth  # [H, W], [0, 1]
        input_np = np.concatenate([rgb_np, depth_norm[np.newaxis]], axis=0)  # [4, H, W]
    else:
        input_np = rgb_np  # [3, H, W]

    input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = net(input_tensor)

    return output


def predict_detr_from_img(
    net,
    rgb_img,
    device,
    input_img_size=(544, 720),
):
    """
    Run a DETR-style frontier model (TwoHeadUnet with head_config='detr') on an RGB image.

    The model predicts N query slots; each slot carries:
        uv     — normalised pixel coord in [0, 1] (x=col, y=row)
        z      — metric depth [m]
        conf   — slot-active probability [0, 1]
        weight — GMM component weight (proxy for information gain)
        occ    — occlusion probability [0, 1]

    Args:
        net:            Loaded TwoHeadUnet(head_config='detr') model.
        rgb_img:        (H, W, 3) uint8 numpy array or PIL Image.
        device:         torch.device for inference.
        input_img_size: (height, width) expected by the model.

    Returns:
        Tuple (uv_np, z_np, conf_np, weight_np, occ_np) where
            uv_np     : (N, 2) float32 — normalised [0, 1] pixel coords (u=col, v=row)
            z_np      : (N,)   float32 — metric depth per slot [m]
            conf_np   : (N,)   float32 — slot confidence [0, 1]
            weight_np : (N,)   float32 — GMM component weight (info-gain proxy)
            occ_np    : (N,)   float32 — occlusion probability [0, 1]
    """
    input_h, input_w = input_img_size

    if isinstance(rgb_img, np.ndarray):
        rgb_pil = Image.fromarray(
            rgb_img.astype(np.uint8) if rgb_img.dtype != np.uint8 else rgb_img
        )
    else:
        rgb_pil = rgb_img

    rgb_pil = rgb_pil.convert("RGB").resize((input_w, input_h), Image.Resampling.BILINEAR)
    rgb_np = np.asarray(rgb_pil, dtype=np.float32) / 255.0   # [H, W, 3]
    rgb_np = rgb_np.transpose(2, 0, 1)                        # [3, H, W]

    input_tensor = torch.from_numpy(rgb_np).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        uv, z, conf, weight, occ, depth_pred = net(input_tensor)

    uv_np     = uv[0].cpu().numpy()                  # (N, 2)
    z_np      = z[0].squeeze(-1).cpu().numpy()       # (N,)
    conf_np   = conf[0].squeeze(-1).cpu().numpy()    # (N,)
    weight_np = weight[0].squeeze(-1).cpu().numpy()  # (N,)
    occ_np    = occ[0].squeeze(-1).cpu().numpy()     # (N,)

    # Auxiliary dense depth prediction (None when detr_aux_depth is disabled)
    depth_np = (
        depth_pred[0, 0].cpu().numpy().astype(np.float32)
        if depth_pred is not None
        else None
    )

    return uv_np, z_np, conf_np, weight_np, occ_np, depth_np
