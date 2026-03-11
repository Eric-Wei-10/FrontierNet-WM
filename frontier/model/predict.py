import numpy as np
import torch
import logging
from PIL import Image

import sys
sys.path.append("/cluster/project/cvg/students/shangwu/dpt_distillation_repo")
from model import DepthEstimationModel


def load_model(path, num_classes=11, use_depth=True):
    model = DepthEstimationModel(head_mode="df_seg", num_classes=num_classes)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model from: {path}")
    logging.info(f"Using device {device}")
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
):
    """
    Preprocess an RGB image and run the model.

    Preprocessing matches the training pipeline (data.py):
      - Direct resize to (input_h, input_w) with BILINEAR interpolation
      - Normalize to [0, 1] by dividing by 255
      - No center crop, no ImageNet mean/std normalization
    """
    input_h, input_w = input_img_size

    # Convert to PIL if needed, then resize directly to model input size
    if isinstance(rgb_img, np.ndarray):
        rgb_pil = Image.fromarray(rgb_img.astype(np.uint8) if rgb_img.dtype != np.uint8 else rgb_img)
    else:
        rgb_pil = rgb_img
    rgb_pil = rgb_pil.convert("RGB").resize((input_w, input_h), Image.Resampling.BILINEAR)

    rgb_np = np.asarray(rgb_pil, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
    rgb_np = rgb_np.transpose(2, 0, 1)                       # [3, H, W]
    rgb_tensor = torch.from_numpy(rgb_np).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = net(rgb_tensor)

    return output
