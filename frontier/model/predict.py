import numpy as np
import torch
import logging
import cv2
from PIL import Image

import sys

_DPT_REPO = "/cluster/project/cvg/students/shangwu/dpt_distillation_repo"
_UNET_REPO = "/cluster/project/cvg/students/shangwu/Pytorch-UNet"
_FACTORY_REPO = "/cluster/project/cvg/students/shangwu/DETR-Factory-PyTorch/src"

# Model types that use the sparse DETR detection path
DETR_MODEL_TYPES = {"unet_detr", "detr", "cond_detr"}

# ImageNet statistics — must match FrontierLoader (frontier_loader.py)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_model(
    path,
    num_classes=11,
    use_depth=False,
    model_type="dpt",
    vit_depth=8,
    feature_layers=None,
    num_queries=10,
    detr_aux_depth=False,
    factory_d_model=256,
    factory_n_tokens=374,
    factory_n_layers=6,
    factory_n_heads=8,
    factory_dropout=0.1,
):
    """
    Load a frontier-detection model.

    Args:
        path:        Path to checkpoint file.
        num_classes: Number of segmentation classes (not used for sparse DETR models).
        use_depth:   Whether the model uses a depth channel (unet only).
        model_type:    "dpt"       — ViT+DPT, RGB-only, df_seg head
                       "unet"      — ResNet34+UNet, RGB-D, df_seg head
                       "unet_detr" — ResNet34+UNet, RGB-only, DETR sparse head
                       "detr"      — FrontierDETR (ResNet50 enc+dec, DETR-Factory)
                       "cond_detr" — FrontierConditionalDETR (ResNet50, cond dec)
        num_queries:        DETR slot queries (unet_detr / detr / cond_detr).
        detr_aux_depth:     Auxiliary dense depth head (unet_detr only).
        factory_d_model:    Transformer dim (detr / cond_detr). Default 256.
        factory_n_tokens:   Spatial tokens for training resolution (detr / cond_detr).
                            Default 374 = (544//32)×(720//32) for 544×720 input.
        factory_n_layers:   Encoder + decoder layers (detr / cond_detr). Default 6.
        factory_n_heads:    Attention heads (detr / cond_detr). Default 8.
        factory_dropout:    Dropout rate in transformer layers (detr / cond_detr). Default 0.1.
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

    elif model_type == "unet_detr":
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

    elif model_type in ("detr", "cond_detr"):
        if _FACTORY_REPO not in sys.path:
            sys.path.insert(0, _FACTORY_REPO)
        if model_type == "detr":
            from models.frontier_detr import FrontierDETR
            model = FrontierDETR(
                d_model=factory_d_model,
                n_tokens=factory_n_tokens,
                n_layers=factory_n_layers,
                n_heads=factory_n_heads,
                n_queries=num_queries,
                dropout=factory_dropout,
            )
        else:
            from models.frontier_cond_detr import FrontierConditionalDETR
            model = FrontierConditionalDETR(
                d_model=factory_d_model,
                n_tokens=factory_n_tokens,
                n_layers=factory_n_layers,
                n_heads=factory_n_heads,
                n_queries=num_queries,
                dropout=factory_dropout,
            )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Expected 'dpt', 'unet', 'unet_detr', 'detr', or 'cond_detr'."
        )

    model.eval()
    model.to(device=device)

    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Backward-compat: old checkpoints had head_uv / head_conf as single nn.Linear
    # layers; newer checkpoints use nn.Sequential with two hidden layers.
    # Detect old format by the presence of flat "head_uv.weight" and patch the
    # model before strict loading so old checkpoints run without modification.
    if (
        "head_uv.weight" in state_dict
        and hasattr(model, "head_uv")
        and isinstance(model.head_uv, torch.nn.Sequential)
    ):
        d_out_uv   = state_dict["head_uv.weight"].shape[0]    # 2
        d_out_conf = state_dict["head_conf.weight"].shape[0]   # 1
        d_in       = state_dict["head_uv.weight"].shape[1]     # d_model
        logging.warning(
            "Old checkpoint detected (flat head_uv/head_conf). "
            "Replacing Sequential heads with single Linear layers for compatibility."
        )
        model.head_uv   = torch.nn.Linear(d_in, d_out_uv).to(device)
        model.head_conf = torch.nn.Linear(d_in, d_out_conf).to(device)

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


def predict_factory_detr_from_img(
    net,
    rgb_img,
    device,
    input_img_size=(544, 720),
):
    """
    Run a DETR-Factory frontier model (FrontierDETR or FrontierConditionalDETR).

    Both models output per-decoder-layer predictions stacked on dim=1:
        (B, n_layers, N, dim)
    Only the last decoder layer is used for inference.

    Depth encoding: the factory models output log1p(z) from head_z (no activation),
    so actual metric depth = expm1(raw_z).  This is normalised here so callers
    receive the same (uv, z_metres, conf, weight, occ, None) tuple as
    predict_detr_from_img.

    Args:
        net:            Loaded FrontierDETR or FrontierConditionalDETR model.
        rgb_img:        (H, W, 3) uint8 numpy array or PIL Image.
        device:         torch.device for inference.
        input_img_size: (height, width) expected by the model. Default (544, 720).

    Returns:
        Tuple (uv_np, z_np, conf_np, weight_np, occ_np, None) where
            uv_np     : (N, 2) float32 — normalised [0,1] pixel coords (u=col, v=row)
            z_np      : (N,)   float32 — metric depth per slot [m]
            conf_np   : (N,)   float32 — slot confidence [0,1]
            weight_np : (N,)   float32 — GMM component weight [0,1]
            occ_np    : (N,)   float32 — occlusion probability [0,1]
            None               — factory models have no auxiliary dense depth head
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
    rgb_np = (rgb_np - _IMAGENET_MEAN) / _IMAGENET_STD        # ImageNet normalisation

    input_tensor = torch.from_numpy(rgb_np).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        all_uv, all_z, all_conf, all_weight, all_occ = net(input_tensor)

    # Use the last decoder layer: (B, n_layers, N, dim) → (B, N, dim)
    uv     = all_uv[:, -1]      # (B, N, 2)
    z_raw  = all_z[:, -1]       # (B, N, 1)  log1p-encoded
    conf   = all_conf[:, -1]    # (B, N, 1)
    weight = all_weight[:, -1]  # (B, N, 1)
    occ    = all_occ[:, -1]     # (B, N, 1)

    uv_np     = uv[0].cpu().numpy()                              # (N, 2)
    z_log1p   = z_raw[0].squeeze(-1).cpu().numpy()               # (N,)
    z_np      = np.expm1(z_log1p).astype(np.float32)             # (N,) metres
    conf_np   = conf[0].squeeze(-1).cpu().numpy()                # (N,)
    weight_np = weight[0].squeeze(-1).cpu().numpy()              # (N,)
    occ_np    = occ[0].squeeze(-1).cpu().numpy()                 # (N,)

    return uv_np, z_np, conf_np, weight_np, occ_np, None
