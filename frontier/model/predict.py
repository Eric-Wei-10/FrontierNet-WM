import numpy as np
import torch
import logging
from PIL import Image

_FACTORY_REPO = "/cluster/project/cvg/students/shangwu/DETR-Factory-PyTorch/src"

# Model types that use the sparse DETR detection path
DETR_MODEL_TYPES = {"detr", "cond_detr"}

# ImageNet statistics — must match FrontierLoader (frontier_loader.py)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_model(
    path,
    model_type="detr",
    num_queries=10,
    factory_d_model=256,
    factory_n_tokens=374,
    factory_n_layers=6,
    factory_n_heads=8,
    factory_dropout=0.1,
):
    """
    Load a DETR-style frontier-detection model.

    Args:
        path:               Path to checkpoint file.
        model_type:         "detr" — FrontierDETR (ResNet50 enc+dec, DETR-Factory)
                            "cond_detr" — FrontierConditionalDETR (ResNet50, cond dec)
        num_queries:        DETR slot queries.
        factory_d_model:    Transformer embedding dimension. Default 256.
        factory_n_tokens:   Spatial tokens for training resolution.
                            Default 374 = (544//32)×(720//32) for 544×720 input.
        factory_n_layers:   Encoder + decoder layers. Default 6.
        factory_n_heads:    Attention heads. Default 8.
        factory_dropout:    Dropout rate in transformer layers. Default 0.1.
    """
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading {model_type} model from: {path}")
    logging.info(f"Using device {device}")

    if model_type in ("detr", "cond_detr"):
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
            f"Unknown model_type '{model_type}'. Expected 'detr' or 'cond_detr'."
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
    if (
        "head_uv.weight" in state_dict
        and hasattr(model, "head_uv")
        and isinstance(model.head_uv, torch.nn.Sequential)
    ):
        d_out_uv   = state_dict["head_uv.weight"].shape[0]
        d_out_conf = state_dict["head_conf.weight"].shape[0]
        d_in       = state_dict["head_uv.weight"].shape[1]
        logging.warning(
            "Old checkpoint detected (flat head_uv/head_conf). "
            "Replacing Sequential heads with single Linear layers for compatibility."
        )
        model.head_uv   = torch.nn.Linear(d_in, d_out_uv).to(device)
        model.head_conf = torch.nn.Linear(d_in, d_out_conf).to(device)

    model.load_state_dict(state_dict)
    logging.info("Model loaded successfully.")
    return model


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
    so actual metric depth = expm1(raw_z).

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
