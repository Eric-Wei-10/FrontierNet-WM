import numpy as np
import torch
import cv2
import hdbscan
import logging
from typing import Optional, List, Any, Tuple

from frontier.model.predict import predict_from_img
from frontier.frontier import Frontier
from frontier.base import Base
from utils.geometry import (
    compute_gradient,
    grad_mag_and_direct_from_gradmap,
    avg_depth_from_bin_mask,
    direction_on_image_to_vec_in_world,
    find_center_point,
)


class FrontierDetector(Base):
    def __init__(
        self,
        model: Any,
        camera_intrinsic: np.ndarray,
        use_depth: bool = True,
        img_size_model: Tuple[int, int] = (320, 320),
        device: str = "cuda",
        log_level: int = logging.INFO,
        model_type: str = "dpt",
        disocclusion_only: bool = False,
        edge_spread_width: int = 20,
    ):
        """
        Args:
            model: the neural network or algorithm instance
            camera_intrinsic: original RGB camera intrinsic matrix
            use_depth: whether to use the depth image
            img_size_model: (height, width) expected by the model — index 0 is H, index 1 is W
            device: "cuda" or "cpu"
            log_level: logging level (e.g. logging.INFO)
            model_type: "dpt" or "unet" — controls preprocessing in predict_from_img
        """
        # configure logging
        super().__init__(params=None, log_level=log_level)

        # core components
        self.model = model
        self.model_type = model_type
        self.device = torch.device(device if device == "cuda" else "cpu")

        # intrinsics & preprocessing state
        self.ori_intrin: np.ndarray = camera_intrinsic
        self.scale_factor: Optional[float] = None
        self.pro_intrin: Optional[np.ndarray] = None
        self.img_size_model: Tuple[int, int] = img_size_model

        # modality flags
        self.use_depth: bool = use_depth
        self.disocclusion_only: bool = disocclusion_only
        self.edge_spread_width: int = edge_spread_width

        # inputs
        self.raw_rgb: Optional[np.ndarray] = None
        self.raw_depth: Optional[np.ndarray] = None
        self.extrinsic: Optional[np.ndarray] = None

        # outputs
        self.df_raw: Optional[np.ndarray] = None  # distance field, frame-48 coords (after disocclusion redistribution)
        self.df_raw_pre_redist: Optional[np.ndarray] = None  # df_raw before redistribution, for debug
        self.disocclusion_mask: Optional[np.ndarray] = None  # bool (H,W) in frame-48 coords, True = disocclusion
        self.df: Optional[np.ndarray] = None  # distance field, projected to frame-0
        self.ft_region: Optional[np.ndarray] = None  # frontier region mask
        self.info_gain: Optional[np.ndarray] = None
        self.ft_3D: Optional[np.ndarray] = None  # 3D frontier clusters

    def _cal_processed_intrinsic(self, input_img_size):
        """
        Calculate new intrinsic matrix after direct resize to model input size.

        The training pipeline resizes the full image to (model_H, model_W) without
        any center crop, so the correct transform is a simple per-axis scale:
            fx' = fx * model_W / W_in
            cx' = cx * model_W / W_in
            fy' = fy * model_H / H_in
            cy' = cy * model_H / H_in

        img_size_model convention: (height, width) — consistent with get_ft_feature.
        """
        W, H = input_img_size
        model_H, model_W = self.img_size_model  # (height, width)

        K = self.ori_intrin.copy().astype(float)
        K[0, 0] *= model_W / W   # fx
        K[0, 2] *= model_W / W   # cx
        K[1, 1] *= model_H / H   # fy
        K[1, 2] *= model_H / H   # cy

        self.pro_intrin = K

    def _redistribute_disocclusion_values(
        self,
        df_raw: np.ndarray,
        depth: np.ndarray,
        forward_dist: float = 1.0,
    ) -> np.ndarray:
        """
        Redistribute df_raw values that fall in the disocclusion region onto the
        nearest frame-48 pixel that IS covered by a frame-0 pixel.

        The disocclusion region is the set of frame-48 pixels that no frame-0 pixel
        forward-projects onto (i.e., coverage == 0 in the forward-warp map).
        Those pixels carry genuine model predictions but are invisible from frame 0,
        so the backward-warp in _project_frame48_to_frame0 never samples them.

        For each disocclusion pixel p at distance d from the nearest covered pixel q:
            df_out[q] += df_raw[p] * (1 / d)

        Pixels exactly on the coverage boundary (d == 0) already belong to the
        covered set and are never processed as disocclusion pixels.

        After redistribution, all disocclusion pixels are zeroed so the returned
        map only contains values at locations visible from frame 0.

        Args:
            df_raw:       (H, W) float array — raw model output in frame-48 coords.
            depth:        (H, W) float array — metric depth at frame 0 (model-resized).
            forward_dist: camera displacement along +Z in metres (default 1.0).

        Returns:
            df_out: (H, W) array — df_raw with disocclusion values scattered onto
                    their nearest visible edge pixels and zeroed at disocclusion sites.
        """
        from scipy.ndimage import distance_transform_edt

        H, W = df_raw.shape
        K = self.pro_intrin
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # --- build forward-warp coverage map ---
        # For each frame-0 pixel (u, v, z), compute the float frame-48 coordinates
        # (u48_f, v48_f) it projects to.  cv2.remap with INTER_LINEAR reads from all
        # four bilinear-neighbour integer pixels around that float location, so we mark
        # all four as covered.  A final 3×3 dilation closes any remaining sub-pixel
        # gaps (e.g. when the depth is exactly an integer causing zero fractional part).
        u_grid, v_grid = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )
        z0 = depth.astype(np.float32)
        z48 = z0 - forward_dist
        valid = (z0 > 0) & (z48 > 0)

        x_cam = (u_grid - cx) * z0 / fx
        y_cam = (v_grid - cy) * z0 / fy
        z48_safe = np.where(valid, z48, 1.0)  # avoid division by zero; invalid pixels masked by np.where
        u48_f = np.where(valid, fx * x_cam / z48_safe + cx, -2.0)
        v48_f = np.where(valid, fy * y_cam / z48_safe + cy, -2.0)

        coverage = np.zeros((H, W), dtype=np.uint8)
        # Mark all four bilinear neighbours (du, dv) ∈ {0,1}²
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            u48_i = np.floor(u48_f).astype(np.int32) + du
            v48_i = np.floor(v48_f).astype(np.int32) + dv
            m = valid & (u48_i >= 0) & (u48_i < W) & (v48_i >= 0) & (v48_i < H)
            coverage[v48_i[m], u48_i[m]] = 1

        # Small dilation to handle any residual sub-pixel gaps
        coverage = cv2.dilate(coverage, np.ones((3, 3), np.uint8), iterations=1)
        coverage_bool = coverage.astype(bool)

        # Store for debug visualisation (updated each detect() call)
        self.disocclusion_mask = ~coverage_bool

        # --- for each uncovered pixel, find distance and coords of nearest covered pixel ---
        uncovered = ~coverage_bool
        dist, nearest_idx = distance_transform_edt(uncovered, return_indices=True)

        disoccl_r, disoccl_c = np.where(uncovered)

        df_out = df_raw.copy()
        contribution = np.zeros_like(df_raw)
        if disoccl_r.size > 0:
            d = dist[disoccl_r, disoccl_c]                     # distance to nearest covered pixel
            weights = np.where(d > 0, 1.0 / d, 1.0)           # 1/d weighting; d==0 shouldn't occur here
            tgt_r = nearest_idx[0, disoccl_r, disoccl_c]
            tgt_c = nearest_idx[1, disoccl_r, disoccl_c]

            np.add.at(contribution, (tgt_r, tgt_c), df_raw[disoccl_r, disoccl_c] * weights)
            df_out[disoccl_r, disoccl_c] = 0.0                 # zero out disocclusion sites
            df_out += contribution

        if self.disocclusion_only:
            # Keep only the pixels that received redistributed contributions;
            # all unmoved (originally covered) pixels are zeroed out.
            df_out = contribution

        self.logger.info(
            "Disocclusion redistribution: %d pixels redistributed; "
            "df_out stats: min=%.3f  max=%.3f  mean=%.3f",
            disoccl_r.size,
            df_out.min(), df_out.max(), df_out.mean(),
        )
        return df_out

    def _project_frame48_to_frame0(
        self,
        value_map: np.ndarray,
        depth: np.ndarray,
        forward_dist: float = 1.0,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Project a value map from frame-48 coordinates back to frame-0 coordinates.

        The model predicts for frame 48, which corresponds to the camera having moved
        forward ``forward_dist`` metres along its +Z axis (fps=24, speed=0.5 m/s →
        1 m over 48 frames).  For each pixel (u, v) in frame 0 with depth z we:

          1. Unproject to 3-D in camera-0 frame:
                 P = [(u-cx)/fx * z,  (v-cy)/fy * z,  z]
          2. Express in camera-48 frame  (camera moved +Z by forward_dist):
                 P' = P - [0, 0, forward_dist]
          3. Project onto the frame-48 image plane:
                 u' = fx * P'x / P'z + cx,   v' = fy * P'y / P'z + cy
          4. Sample value_map at (u', v').

        Args:
            value_map:    (H, W) float array — model output at frame 48.
            depth:        (H, W) float array — metric depth at frame 0 (preprocessed).
            forward_dist: Camera displacement along +Z in metres (default 1.0).
            interpolation: cv2 interpolation flag (default cv2.INTER_LINEAR).

        Returns:
            projected: (H, W) array — value map projected to frame-0 coordinates.
        """
        H, W = value_map.shape
        K = self.pro_intrin
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # pixel-coordinate grids for frame 0
        u_grid, v_grid = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )

        z0 = depth.astype(np.float32)
        x_cam = (u_grid - cx) * z0 / fx   # X in camera-0 frame
        y_cam = (v_grid - cy) * z0 / fy   # Y in camera-0 frame
        z48 = z0 - forward_dist            # Z in camera-48 frame

        # only pixels whose depth is positive in camera-48 are projectable
        valid = z48 > 0
        map_x = np.where(valid, fx * x_cam / z48 + cx, -1.0).astype(np.float32)
        map_y = np.where(valid, fy * y_cam / z48 + cy, -1.0).astype(np.float32)

        projected = cv2.remap(
            value_map.astype(np.float32),
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return projected

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        df_thr: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run frontier detection on an RGB + depth pair.

        Returns:
            ft_region: 2D frontier-region mask
            info_gain:  2D info-gain map
        """
        # store raw inputs
        self.raw_rgb, self.raw_depth = rgb, depth
        H, W = rgb.shape[:2]
        self.logger.debug(f"Input size (WxH): {W}x{H}")
        self.scale_factor = (
            max(self.img_size_model[0] / H, self.img_size_model[1] / W) + 0.02
        )

        # compute scale & intrinsics
        self._cal_processed_intrinsic((W, H))

        # --- 1) Inference ---
        self.logger.debug("Running model inference...")
        df_tensor, cls_mask = predict_from_img(
            net=self.model,
            rgb_img=rgb,
            depth_img=depth,
            device=self.device,
            scale_factor=self.scale_factor,
            use_depth=self.use_depth,
            input_img_size=self.img_size_model,
            model_type=self.model_type,
        )
        # DepthEstimationModel returns df_tensor of shape [1, 1, H, W] (single channel).
        self.df_raw = df_tensor.cpu().detach().numpy().squeeze()  # -> [H, W], frame-48 coords
        self.df = self.df_raw  # will be overwritten with projected version below
        self.logger.info(
            "Inference complete; interest map shape: %s  raw stats: min=%.3f  max=%.3f  mean=%.3f",
            self.df_raw.shape, self.df_raw.min(), self.df_raw.max(), self.df_raw.mean(),
        )

        # --- 1.5) Project frame-48 outputs → frame-0 coordinates ---
        # The model was trained to predict for the view 48 frames ahead
        # (fps=24, speed=0.5 m/s → 1 m forward along camera +Z).
        # Resize depth to model input size using the same direct-resize strategy
        # as the training pipeline (no center crop).
        model_H, model_W = self.img_size_model
        depth_proc = cv2.resize(
            depth.astype(np.float32), (model_W, model_H), interpolation=cv2.INTER_LINEAR
        )

        # --- 1.6) Redistribution disabled at inference time ---
        # Disocclusion redistribution is now applied during training data preparation
        # (data.py) so the model learns to fit redistributed targets directly.
        self.df_raw_pre_redist = self.df_raw.copy()  # kept for debug panel consistency

        self.logger.info("Projecting frame-48 outputs to frame-0 coordinates...")
        self.df = self._project_frame48_to_frame0(self.df_raw, depth_proc)

        

        self.logger.info(
            "Projected DF stats: min=%.3f  max=%.3f  mean=%.3f  "
            ">3.51 (frontier threshold): %.1f%%  "
            "valid pixels (depth>1m): %.1f%%",
            self.df.min(),
            self.df.max(),
            self.df.mean(),
            100.0 * (self.df > 3.51).mean(),
            100.0 * (depth_proc > 1.0).mean(),
        )

        # Project each channel of the classification logits independently.
        cls_np = cls_mask.cpu().numpy()  # [1, n_classes, H, W]
        projected_channels = np.stack(
            [
                self._project_frame48_to_frame0(cls_np[0, c], depth_proc)
                for c in range(cls_np.shape[1])
            ],
            axis=0,
        )[np.newaxis]  # [1, n_classes, H, W]
        cls_mask = torch.from_numpy(projected_channels).to(device=cls_mask.device, dtype=cls_mask.dtype)

        df_tensor = torch.from_numpy(self.df)

        # --- 2) Postprocessing ---
        # The model outputs a [0, 1] interest map (IDP), NOT a normalize_df-encoded distance
        # field. The original prediction2frontiermap pipeline (denormalize_df + threshold)
        # would require outputs > 3.51, which the model never reaches. Instead:
        #
        #   ft_region  — direct threshold on the projected interest map
        #   info_gain  — segment class → interest midpoint, scaled for get_3D_ft_clusters
        #
        # Segmentation bin edges used during training: (0.05, 0.15, 0.3, 0.45, 0.6, 0.8)
        # partitioning [0, 1] interest into 7 classes (0 = low/invalid, 6 = highest).
        # get_3D_ft_clusters applies an internal *0.01 factor; scaling gain_map by 1000
        # maps interest > 0.1 → final gain > 1, which passes filter_min_gain=1 in config.

        interest_np = self.df  # projected [H, W], values in [0, 1]
        self.ft_region = (interest_np > df_thr).astype(np.float32)
        self.logger.info(
            "Postprocessing: interest threshold=%.2f, frontier pixels=%.1f%%",
            df_thr, 100.0 * self.ft_region.mean(),
        )

        _seg_edges = [0.0, 0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
        _bin_midpoints = np.array(
            [(_seg_edges[i] + _seg_edges[i + 1]) / 2.0 for i in range(len(_seg_edges) - 1)],
            dtype=np.float32,
        )
        cls_label = cls_mask.argmax(dim=1).squeeze().cpu().numpy()  # [H, W], labels 0-6
        gain_map = _bin_midpoints[cls_label]  # each pixel → midpoint interest value
        self.info_gain = np.where(self.ft_region > 0, gain_map * 1000.0, 0.0)

        return self.ft_region, self.info_gain

    def anchor_fts(
        self, depth: np.ndarray, extrinsic: np.ndarray
    ) -> Optional[List[Frontier]]:
        """
        Anchoring 2D Frontiers to 3D Frontier Clusters using depth and camera extrinsics.
        """
        # 1) Resize depth to model input size (direct resize, matching training pipeline)
        model_H, model_W = self.img_size_model
        depth = cv2.resize(
            depth.astype(np.float32), (model_W, model_H), interpolation=cv2.INTER_LINEAR
        )
        self.extrinsic = extrinsic

        # 2) Validate shapes: ft_region and depth should align
        if self.ft_region.shape != depth.shape:
            raise ValueError(f"Shape mismatch: {self.ft_region.shape} vs {depth.shape}")

        # 3) Get depth feature: direction & avg depth
        direction, depth_avg = self.get_depth_feature(self.ft_region, depth)

        # 4) Get per-pixel 2D frontier features, namely Ft^{2D} in the paper
        self.get_ft_feature(direction, depth_avg)

        # 5) get frontier clusters
        result = self.get_3D_ft_clusters(depth_avg, extrinsic)
        if result is None:
            return None
        ft_clusters, _, _ = result  # (Ft^{3D} in the paper)

        # 6) Build Frontier objects from each 3D cluster
        frontiers = []
        for cluster in ft_clusters:
            # [u, v, dx, dy, gain, …, x, y, z, vx, vy, vz]
            u, v, dx, dy, gain, *_, x, y, z, vx, vy, vz = cluster[:12]
            f = Frontier()
            f.pixel_pos = (u, v)
            f.direct_angle = np.arctan2(dy, dx)
            f.gain = f.u_gain = gain
            f.pos3d = (x, y, z)
            f.view_direction = (vx, vy, vz)
            f.set_valid()
            frontiers.append(f)

        self.ft_3D = frontiers
        return frontiers

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

        Goals that are marked occluded are still returned as valid frontiers; the
        downstream path planner handles navigation to goals that are not directly
        visible in the current view.

        Args:
            rgb:         (H, W, 3) uint8 RGB image.
            extrinsic:   4×4 C_T_W matrix (world-to-camera, i.e. inv(W_T_C)).
            conf_thresh: Minimum slot confidence to accept (default 0.3).

        Returns:
            List of Frontier objects (one per accepted slot), or None if no
            slots exceed the confidence threshold.
        """
        from frontier.model.predict import (
            predict_detr_from_img,
            predict_factory_detr_from_img,
            DETR_MODEL_TYPES,
        )

        H, W = rgb.shape[:2]
        # Compute scaled intrinsics for back-projection at model resolution
        self._cal_processed_intrinsic((W, H))

        # Log RGB statistics so we can verify the input changes between calls
        rgb_f = rgb[..., :3].astype(np.float32)
        self.logger.info(
            "detect_detr input  cam_pos_world=(%.2f, %.2f, %.2f)  "
            "rgb mean=%.3f  std=%.3f",
            *np.linalg.inv(extrinsic)[:3, 3],
            rgb_f.mean(), rgb_f.std(),
        )

        # --- Inference ---
        if self.model_type in ("detr", "cond_detr"):
            uv_np, z_np, conf_np, weight_np, occ_np, depth_np = predict_factory_detr_from_img(
                net=self.model,
                rgb_img=rgb,
                device=self.device,
                input_img_size=self.img_size_model,
            )
        else:  # unet_detr
            uv_np, z_np, conf_np, weight_np, occ_np, depth_np = predict_detr_from_img(
                net=self.model,
                rgb_img=rgb,
                device=self.device,
                input_img_size=self.img_size_model,
            )
        # Store raw inputs and aux depth for debug visualisation
        self.raw_rgb = rgb
        self.detr_depth_pred: Optional[np.ndarray] = depth_np  # (model_H, model_W) or None

        # Log raw model outputs — if UV never changes across steps the model is
        # collapsing to fixed queries; if RGB changes but UV is static it is a
        # model generalisation issue, not an exploration bug
        self.logger.info(
            "detect_detr raw uv_np (all slots):\n%s",
            np.array2string(uv_np, precision=4, suppress_small=True),
        )
        self.logger.info(
            "detect_detr z_np   : %s",
            np.array2string(z_np, precision=3, suppress_small=True),
        )
        self.logger.info(
            "detect_detr conf_np: %s",
            np.array2string(conf_np, precision=3, suppress_small=True),
        )

        model_H, model_W = self.img_size_model
        K = self.pro_intrin
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # Camera centre in world frame (for view-direction computation + debug)
        W_T_C = np.linalg.inv(extrinsic)
        cam_pos_world = W_T_C[:3, 3]

        # Track displacement between successive detect_detr calls for debug overlay
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

            # Pixel coordinates in model-input space
            u_px = float(uv_np[n, 0]) * model_W   # column
            v_px = float(uv_np[n, 1]) * model_H   # row
            z = float(z_np[n])

            if z <= 0.0:
                continue

            is_occluded = float(occ_np[n]) > 0.5

            # Unproject to camera frame
            x_cam = (u_px - cx) * z / fx
            y_cam = (v_px - cy) * z / fy
            cam_pt = np.array([x_cam, y_cam, z, 1.0])

            # Transform to world frame
            world_pt = (W_T_C @ cam_pt)[:3]

            # View direction: camera centre → frontier point (unit vector)
            vd = world_pt - cam_pos_world
            vd_norm = np.linalg.norm(vd)
            if vd_norm < 1e-6:
                continue
            vd = vd / vd_norm

            # 2-D viewing angle in image plane (from principal point to slot pixel)
            direct_angle = float(np.arctan2(v_px - cy, u_px - cx))

            # Occluded frontiers are unexplored by definition and get full gain.
            # Already-visible frontiers are partially known and receive a strong
            # discount so the planner prefers occluded (hidden) goals.
            base_gain = float(weight_np[n]) * gain_scale
            effective_gain = base_gain if is_occluded else base_gain * visible_gain_discount

            # Record for debug visualisation
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

    def get_depth_feature(
        self, bin_mask: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        get 2D viewing direction and average depth for each frontier pixel
        this corresponding to Sec. III.E.1 and Sec. III.E.3 in the paper
        """
        # quick sanity check
        if bin_mask.shape != depth.shape:
            raise ValueError(
                f"bin_mask shape {bin_mask.shape} != depth shape {depth.shape}"
            )

        # process depth image, get gradian and direction
        grad_x, grad_y = compute_gradient(depth, kernel_size=9)
        magnitude, direction = grad_mag_and_direct_from_gradmap(grad_x, grad_y)
        self.logger.debug(
            "Gradient computed: grad_x %s, grad_y %s, direction %s",
            grad_x.shape,
            grad_y.shape,
            direction.shape,
        )

        # get avg depth
        avg_depth = avg_depth_from_bin_mask(
            bin_mask=bin_mask,
            depth=depth,
            gradient_direction=direction,
            gradient_magnitude=magnitude,
            skip_close_points=0.3,
        )
        self.logger.debug("Average depth computed, shape %s", avg_depth.shape)

        return direction, avg_depth

    def get_ft_feature(self, direction: np.ndarray, depth_avg: np.ndarray) -> None:
        """construct per-pixel features from detection"""
        # find all frontier pixel coordinates
        ys, xs = np.where(self.ft_region == 1)

        # filter out invalid frontier pixels (frontier pixels with outlier depth)
        valid_mask = depth_avg[ys, xs] != 0
        ys, xs = ys[valid_mask], xs[valid_mask]

        # number of valid frontier pixels
        n = len(ys)
        # preallocate feature array: [x, y, cos(theta), sin(theta), gains, depth_avg]
        feature2D = np.zeros((n, 6), dtype=float)

        # normalized pixel coordinates
        feature2D[:, 0] = xs / self.img_size_model[1]  # x
        feature2D[:, 1] = ys / self.img_size_model[0]  # y

        # angles in radians
        radians = direction[ys, xs] * np.pi / 180
        feature2D[:, 2] = np.cos(radians)  # cos(theta)
        feature2D[:, 3] = np.sin(radians)  # sin(theta)

        # gains and depth
        feature2D[:, 4] = self.info_gain[ys, xs]
        feature2D[:, 5] = depth_avg[ys, xs]

        # assign back to instance
        self.feature2D = feature2D

    def get_3D_ft_clusters(
        self,
        depth_avg: np.ndarray,
        cam_extrinsic: np.ndarray,
        remove_close_ft_thr: float = 0.3,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        cluster and lift 2D frontier features to 3D frontiers
        this corresponds to Sec. III.E.2 in the paper
        """
        # set up clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=20,
            min_samples=25,
            cluster_selection_epsilon=0.5,
            metric="l2",
        )
        cluster_weights = np.array([10, 10, 1, 1, 0.3])
        cluster_dims = [0, 1, 2, 3, 5]

        try:
            # weight and select dims in one go
            feats = self.feature2D[:, cluster_dims] * cluster_weights
            labels = clusterer.fit_predict(feats)
            max_label = labels.max()
            num_clusters = int(max_label + 1)
        except Exception:
            self.logger.debug("No frontiers detected")
            return None

        # precompute camera-to-world
        T_cam2world = np.linalg.inv(cam_extrinsic)
        clusters_out = []

        # process each cluster
        for lbl in range(num_clusters):
            mask = labels == lbl
            if not mask.any():
                continue
            pts = self.feature2D[mask]

            _feature = np.zeros(12, dtype=float)

            # (x, y) center point (closest to centroid)
            _feature[0], _feature[1] = find_center_point(pts[:, :2])

            # mean direction unit vector
            vec_sum = pts[:, 2:4].sum(axis=0)
            norm = np.linalg.norm(vec_sum)
            if norm == 0:
                continue
            _feature[2:4] = vec_sum / norm

            # robust gain
            gains = np.sort(pts[:, 4])
            mid = len(gains) // 2
            if len(gains) < 10:
                _feature[4] = gains.mean()
            else:
                _feature[4] = np.median(gains[mid : int(0.9 * len(gains))])
            _feature[4] *= (
                10 * 0.001
            )  # recover volume NOTE: this scaling is to compensate for the volume scaling in the detection model

            # robust depth (ignore zeros)
            depths = np.sort(pts[:, 5])
            depths = depths[depths > 0]
            if depths.size == 0:
                continue
            low, high = int(0.05 * len(depths)), int(0.9 * len(depths))
            _feature[5] = np.median(depths[low:high])

            # Skip if too close
            if _feature[5] < remove_close_ft_thr:
                continue

            # backproject center pixel to 3D
            H, W = self.img_size_model
            i_y = int(_feature[0] * W)
            i_x = int(_feature[1] * H)
            z = depth_avg[i_x, i_y]
            x = (i_y - self.pro_intrin[0, 2]) * z / self.pro_intrin[0, 0]
            y = (i_x - self.pro_intrin[1, 2]) * z / self.pro_intrin[1, 1]
            cam_pt = np.array([x, y, z, 1.0]).reshape(4, 1)
            world_pt = T_cam2world @ cam_pt
            _feature[6:9] = world_pt[:3, 0]

            # Project 2D direction to 3D
            ang = np.arctan2(_feature[3], _feature[2])
            _feature[9:12] = direction_on_image_to_vec_in_world(
                ang, self.pro_intrin, cam_extrinsic
            ).ravel()

            clusters_out.append(_feature)

        if not clusters_out:
            return None

        ft_clusters = np.vstack(clusters_out)
        return ft_clusters, labels, ft_clusters.shape[0]

    def save_result_npz(self, save_path):
        """
        save the result to a npz file
        """
        np.savez(
            save_path,
            raw_rgb=self.raw_rgb,
            raw_depth=self.raw_depth,
            input_img_size=self.img_size_model,
            df=self.df,
            ft_region=self.ft_region,
            info_gain=self.info_gain,
            ft_3D=self.ft_3D,
            pro_intrinsic=self.pro_intrin,
            ori_intrinsic=self.ori_intrin,
            extrinsic=self.extrinsic,
        )

    def get_detection_result(self):
        """
        return the result as a dictionary
        """
        return {
            "raw_rgb": self.raw_rgb,
            "raw_depth": self.raw_depth,
            "input_img_size": self.img_size_model,
            "df": self.df,
            "ft_region": self.ft_region,
            "info_gain": self.info_gain,
            "ft_3D": self.ft_3D,
            "pro_intrinsic": self.pro_intrin,
            "ori_intrinsic": self.ori_intrin,
            "extrinsic": self.extrinsic,
        }
