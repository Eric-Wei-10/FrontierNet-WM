import numpy as np
import torch
import logging
from typing import Optional, List, Any, Tuple

from frontier.frontier import Frontier
from frontier.base import Base


class FrontierDetector(Base):
    def __init__(
        self,
        model: Any,
        camera_intrinsic: np.ndarray,
        img_size_model: Tuple[int, int] = (544, 720),
        device: str = "cuda",
        log_level: int = logging.INFO,
        model_type: str = "detr",
    ):
        super().__init__(params=None, log_level=log_level)

        self.model = model
        self.model_type = model_type
        self.device = torch.device(device if device == "cuda" else "cpu")

        self.ori_intrin: np.ndarray = camera_intrinsic
        self.pro_intrin: Optional[np.ndarray] = None
        self.img_size_model: Tuple[int, int] = img_size_model

        self.raw_rgb: Optional[np.ndarray] = None
        self.extrinsic: Optional[np.ndarray] = None
        self.ft_3D: Optional[List[Frontier]] = None

    def _cal_processed_intrinsic(self, input_img_size):
        """
        Calculate new intrinsic matrix after direct resize to model input size.

        img_size_model convention: (height, width).
        """
        W, H = input_img_size
        model_H, model_W = self.img_size_model

        K = self.ori_intrin.copy().astype(float)
        K[0, 0] *= model_W / W   # fx
        K[0, 2] *= model_W / W   # cx
        K[1, 1] *= model_H / H   # fy
        K[1, 2] *= model_H / H   # cy

        self.pro_intrin = K

    def detect_detr(
        self,
        rgb: np.ndarray,
        extrinsic: np.ndarray,
        conf_thresh: float = 0.3,
        gain_scale: float = 10.0,
        visible_gain_discount: float = 0.1,
    ) -> Optional[List[Frontier]]:
        """
        Run DETR-style frontier detection.

        The model directly predicts a sparse set of N frontier candidates.  Each
        candidate is parameterised by (u, v, z) — a 3-D point expressed as
        normalised image coordinates plus metric depth — together with a GMM
        component weight (proxy for information gain) and an occlusion flag.

        Args:
            rgb:         (H, W, 3) uint8 RGB image.
            extrinsic:   4×4 C_T_W matrix (world-to-camera, i.e. inv(W_T_C)).
            conf_thresh: Minimum slot confidence to accept (default 0.3).

        Returns:
            List of Frontier objects (one per accepted slot), or None if no
            slots exceed the confidence threshold.
        """
        from frontier.model.predict import predict_factory_detr_from_img

        H, W = rgb.shape[:2]
        self._cal_processed_intrinsic((W, H))

        rgb_f = rgb[..., :3].astype(np.float32)
        self.logger.info(
            "detect_detr input  cam_pos_world=(%.2f, %.2f, %.2f)  "
            "rgb mean=%.3f  std=%.3f",
            *np.linalg.inv(extrinsic)[:3, 3],
            rgb_f.mean(), rgb_f.std(),
        )

        uv_np, z_np, conf_np, weight_np, occ_np, depth_np = predict_factory_detr_from_img(
            net=self.model,
            rgb_img=rgb,
            device=self.device,
            input_img_size=self.img_size_model,
        )
        self.raw_rgb = rgb
        self.detr_depth_pred: Optional[np.ndarray] = depth_np

        self.logger.info(
            "detect_detr raw uv_np (all slots):\n%s",
            np.array2string(uv_np, precision=4, suppress_small=True),
        )
        self.logger.info(
            "detect_detr z_np   : %s",
            np.array2string(z_np, precision=3, suppress_small=True),
        )
        self.logger.info(
            "detect_detr conf_np:   %s",
            np.array2string(conf_np, precision=3, suppress_small=True),
        )
        self.logger.info(
            "detect_detr weight_np: %s",
            np.array2string(weight_np, precision=3, suppress_small=True),
        )
        self.logger.info(
            "detect_detr occ_np:    %s",
            np.array2string(occ_np, precision=3, suppress_small=True),
        )

        model_H, model_W = self.img_size_model
        K = self.pro_intrin
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        W_T_C = np.linalg.inv(extrinsic)
        cam_pos_world = W_T_C[:3, 3]

        prev_pos = getattr(self, "_detr_last_cam_pos", None)
        self._detr_cam_pos: np.ndarray = cam_pos_world.copy()
        self._detr_cam_displacement: float = (
            float(np.linalg.norm(cam_pos_world - prev_pos)) if prev_pos is not None else 0.0
        )
        self._detr_last_cam_pos = cam_pos_world.copy()

        frontiers: List[Frontier] = []
        self.detr_slots: List[dict] = []

        for n in range(len(conf_np)):
            if conf_np[n] < conf_thresh:
                continue

            u_px = float(uv_np[n, 0]) * model_W
            v_px = float(uv_np[n, 1]) * model_H
            z = float(z_np[n])

            if z <= 0.0:
                continue

            is_occluded = float(occ_np[n]) > 0.5

            x_cam = (u_px - cx) * z / fx
            y_cam = (v_px - cy) * z / fy
            cam_pt = np.array([x_cam, y_cam, z, 1.0])

            world_pt = (W_T_C @ cam_pt)[:3]

            vd = world_pt - cam_pos_world
            vd_norm = np.linalg.norm(vd)
            if vd_norm < 1e-6:
                continue
            vd = vd / vd_norm

            direct_angle = float(np.arctan2(v_px - cy, u_px - cx))

            base_gain = float(weight_np[n]) * gain_scale
            effective_gain = base_gain if is_occluded else base_gain * visible_gain_discount

            self.detr_slots.append({
                "u_px": u_px, "v_px": v_px,
                "z": z, "conf": float(conf_np[n]),
                "weight": effective_gain,
                "weight_raw": float(weight_np[n]),
                "occ": float(occ_np[n]),
                "is_occluded": is_occluded,
            })

            f = Frontier()
            f.pixel_pos = np.array([u_px, v_px])
            f.direct_angle = direct_angle
            f.gain = effective_gain
            f.u_gain = effective_gain
            f.pos3d = world_pt
            f.view_direction = vd
            f.set_valid()
            frontiers.append(f)

        n_occ = int(sum(s["is_occluded"] for s in self.detr_slots))
        n_vis = len(self.detr_slots) - n_occ
        self.ft_3D = frontiers if frontiers else None
        self.logger.info(
            "DETR detect: %d frontiers total  "
            "(%d occluded/priority, %d visible/discounted, conf_thresh=%.2f, "
            "visible_gain_discount=%.2f)",
            len(frontiers), n_occ, n_vis, conf_thresh, visible_gain_discount,
        )
        return frontiers if frontiers else None
