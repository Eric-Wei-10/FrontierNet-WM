"""
Headless exploration demo using Open3D's OffscreenRenderer.
This version does not require a display and can run on clusters without monitors.
"""

import os
import shutil
import time
import logging
import argparse
from collections import deque
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image as _PILImage

from utils.vis_utils import (
    create_camera,
)

# FrontierNet
from frontier.detector import FrontierDetector
from frontier.classic_detector import ClassicFrontierDetector
from frontier.model.predict import load_model
from utils.frontier_utils import read_config_yaml

# Mapping
from mapping.wavemap import WaveMapper

# Frontier Manager
from frontier.manager import FrontierManager

# mono depth
from mono_depth.Metric3D import metric_depth_from_rgb as metric_depth_from_rgb_metric3d
from mono_depth.UniK3D import metric_depth_from_rgb as metric_depth_from_rgb_unik3d


class HeadlessRenderer:
    """
    A headless renderer using Open3D's OffscreenRenderer for rendering RGBD
    from a mesh without requiring a display.
    """

    def __init__(
        self,
        mesh_path: str,
        width: int,
        height: int,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        z_near: float = 0.02,
        z_far: float = 50.0,
    ):
        """
        Initialize the headless renderer.

        Args:
            mesh_path: Path to the mesh file.
            width: Image width in pixels.
            height: Image height in pixels.
            intrinsic: Camera intrinsic parameters.
            z_near: Near clipping plane.
            z_far: Far clipping plane.
        """
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.z_near = z_near
        self.z_far = z_far

        # Current camera extrinsic (C_T_W: camera frame expressed in world)
        self._extrinsic = np.eye(4)

        # Create the offscreen renderer
        self.renderer = rendering.OffscreenRenderer(width, height)

        # Load the mesh with materials using read_triangle_model for GLB/GLTF
        # This preserves textures and materials
        mesh_path_lower = mesh_path.lower()
        if mesh_path_lower.endswith('.glb') or mesh_path_lower.endswith('.gltf'):
            # Use read_triangle_model which preserves materials/textures
            model = o3d.io.read_triangle_model(mesh_path)
            if model is not None and len(model.meshes) > 0:
                # Add each mesh with its material
                for i, mesh_info in enumerate(model.meshes):
                    mesh_geom = mesh_info.mesh
                    material_idx = mesh_info.material_idx
                    
                    if material_idx >= 0 and material_idx < len(model.materials):
                        mat = model.materials[material_idx]
                        # Use "unlitLine" shader which shows pure colors without any lighting
                        # Or create a new unlit material with just the albedo texture
                        new_mat = rendering.MaterialRecord()
                        new_mat.shader = "defaultUnlit"
                        # Copy albedo image if exists
                        if mat.albedo_img is not None:
                            new_mat.albedo_img = mat.albedo_img
                        # Set base color to white so texture shows at full brightness
                        new_mat.base_color = [1.0, 1.0, 1.0, 1.0]
                        self.renderer.scene.add_geometry(f"mesh_{i}", mesh_geom, new_mat)
                    else:
                        # Fallback material
                        mat = rendering.MaterialRecord()
                        mat.shader = "defaultUnlit"
                        mat.base_color = [0.8, 0.8, 0.8, 1.0]
                        self.renderer.scene.add_geometry(f"mesh_{i}", mesh_geom, mat)
                print(f"Loaded GLB model with {len(model.meshes)} meshes and {len(model.materials)} materials")
            else:
                raise ValueError(f"Failed to load model from {mesh_path}")
        else:
            # For other formats, load as triangle mesh
            mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
            material = rendering.MaterialRecord()
            
            if mesh.has_vertex_colors():
                material.shader = "defaultUnlit"
            else:
                material.shader = "defaultLit"
                material.base_color = [0.8, 0.8, 0.8, 1.0]
            
            self.renderer.scene.add_geometry("mesh", mesh, material)
            print(f"Loaded mesh: {mesh}")
        
        # Disable lighting since we use unlit shader - show original colors
        self.renderer.scene.scene.enable_sun_light(False)
        self.renderer.scene.scene.enable_indirect_light(False)
        
        # Set background to white for better visibility
        self.renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        # Setup initial camera
        self._setup_camera()

    def cleanup(self):
        """Clean up renderer resources."""
        # Do NOT call clear_geometry() — it releases Filament GPU resources and
        # then the OffscreenRenderer destructor tries to free them again, causing
        # the "nonexistent resource" crash.  Setting to None destroys everything
        # in one shot through Filament's own destructor.
        self.renderer = None

    def _setup_camera(self):
        """Setup camera with current intrinsic and extrinsic."""
        # Get intrinsic matrix
        K = self.intrinsic.intrinsic_matrix

        # Setup camera using intrinsic matrix and extrinsic
        self.renderer.setup_camera(
            K,
            self._extrinsic,
            self.intrinsic.width,
            self.intrinsic.height,
        )

    def set_extrinsic(self, extrinsic: np.ndarray):
        """
        Set the camera extrinsic matrix.

        Args:
            extrinsic: 4x4 camera-to-world transformation matrix (C_T_W).
        """
        self._extrinsic = extrinsic.copy()
        self._setup_camera()

    def get_extrinsic(self) -> np.ndarray:
        """Get the current camera extrinsic matrix (C_T_W)."""
        return self._extrinsic.copy()

    def get_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        """Get the camera intrinsic parameters."""
        return self.intrinsic

    def capture_rgb(self) -> np.ndarray:
        """
        Capture RGB image from the current viewpoint.

        Returns:
            RGB image as numpy array (H, W, 3) with values in [0, 255].
        """
        img = self.renderer.render_to_image()
        rgb = np.asarray(img)
        # Stretch to model input size (540→544) using LANCZOS, matching gen3c_frontier.py
        pil_img = _PILImage.fromarray(rgb)
        pil_img = pil_img.resize((self.width, HeadlessExplorerApp.MODEL_H), _PILImage.LANCZOS)
        return np.asarray(pil_img)

    def capture_depth(self) -> np.ndarray:
        """
        Capture depth image from the current viewpoint.

        Returns:
            Depth image as numpy array (H, W) with depth in meters.
        """
        # z_in_view_space=True gives us actual depth values (distance from camera)
        depth_img = self.renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_img).astype(np.float32)

        # Replace inf values (background/sky) with 0
        depth[~np.isfinite(depth)] = 0.0

        return depth

    def capture_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture both RGB and depth images.

        Returns:
            Tuple of (rgb, depth) numpy arrays.
        """
        rgb = self.capture_rgb()
        depth = self.capture_depth()
        return rgb, depth


class HeadlessExplorerApp:
    """
    A headless version of the exploration app that uses OffscreenRenderer
    instead of interactive visualization windows.
    """

    # ---------- constants / defaults ----------
    VOX_SIZE = 0.1

    # Camera (robot) defaults
    CAM_H, CAM_W, CAM_F = 540, 720, 300.0
    # Model input height — rendered at CAM_H=540 then stretched to match training
    MODEL_H = 544

    # Depth sources
    DEPTH_GT = "GT"
    DEPTH_M3D = "Metric3D"
    DEPTH_UNIK3D = "UniK3D"

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Config
        self.config = read_config_yaml(args.config)
        self.detect_interval: int = int(self.config.get("detect_interval", 10))
        self.plan_interval: int = int(self.config.get("plan_interval", 10))

        # Depth source
        self.depth_source = args.depth_source

        # Headless renderer
        self.renderer: Optional[HeadlessRenderer] = None

        # Frontier, mapping, detector
        self.mapper: Optional[WaveMapper] = None
        self.ft_manager: Optional[FrontierManager] = None
        self.ft_detector: Optional[FrontierDetector] = None
        self.classic_detector: Optional[ClassicFrontierDetector] = None
        self.VOX_SIZE = (
            self.config["voxel_size"]
            if self.config["voxel_size"] is not None
            else self.VOX_SIZE
        )

        # Global occupancy map pre-built from voxel grid / mesh (None until setup_system)
        self.global_occ_pts: Optional[np.ndarray] = None
        self.global_free_pts: Optional[np.ndarray] = None

        # Path & motion tracking
        self.path_to_go: List[np.ndarray] = []
        self.move_enough: bool = True
        self.last_W_T_C: np.ndarray = np.eye(4)  # camera pose

        # Stuck detection: drop the current frontier if the robot's cumulative
        # path length over the last N steps is below a threshold, indicating it
        # is physically blocked (e.g. against a wall).
        self._stuck_window: int = 5              # number of recent steps to inspect
        self._stuck_disp_threshold: float = 0.3  # metres — min cumulative path length
        self._recent_positions: deque = deque(maxlen=self._stuck_window + 1)

        # Fix 2: raw loop-iteration counter independent of n_robot_poses.
        # n_robot_poses only increments when is_moving() fires; it freezes during
        # micro-loops (snap goal == robot position).  _loop_iter always advances so
        # we can force detection and bound runaway loops.
        self._loop_iter: int = 0
        self._frozen_iters: int = 0          # consecutive iterations with same n_robot_poses
        self._last_n_robot_poses: int = -1   # n_robot_poses value at previous iteration

        # Fix 3: dirty flag for gain recomputation.
        # gain_adjustment_detr is O(N*M) in frontiers × robot poses; skip it when
        # neither the pose set nor the frontier set has changed.
        self._last_gain_adj_n_poses: int = -1

        # JSON output
        save_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(save_dir, exist_ok=True)
        self.json_path: str = args.write_path or None
        # make parent dir
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

        # Debug visualisation — default to <json_parent>/debug if not explicitly set
        if args.debug_dir:
            self.debug_dir: Optional[str] = args.debug_dir
        elif self.json_path:
            self.debug_dir = os.path.join(os.path.dirname(self.json_path), "debug")
        else:
            self.debug_dir = None
        self._debug_step: int = 0

    # ---------- RGBD capture ----------

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture RGB and depth using the configured depth source.

        Returns:
            Tuple of (rgb, depth) numpy arrays.
        """
        assert self.renderer is not None

        rgb = self.renderer.capture_rgb()
        K = self.renderer.get_intrinsic().intrinsic_matrix

        if self.depth_source == self.DEPTH_GT:
            depth = self.renderer.capture_depth()
        elif self.depth_source == self.DEPTH_UNIK3D:
            depth = metric_depth_from_rgb_unik3d(rgb_input=rgb, intrinsic_mat=K)
        elif self.depth_source == self.DEPTH_M3D:
            depth = metric_depth_from_rgb_metric3d(
                rgb_input=rgb,
                intrinsic_mat=K,
                camera_W=rgb.shape[1],
                camera_H=rgb.shape[0],
                local_model_path=None,
            )
        else:
            raise ValueError(f"Unknown depth_source: {self.depth_source}")

        return rgb, depth

    def is_moving(self, current_pose: np.ndarray, trans_thre: float = 0.1, rot_thre: float = 0.26) -> bool:
        """Check if the robot has moved enough from the last recorded pose."""
        from utils.geometry import pose_difference

        trans_diff, rot_diff = pose_difference(
            self.last_W_T_C.reshape(1, 4, 4), current_pose.reshape(1, 4, 4)
        )
        return trans_diff[0, 0] > trans_thre or rot_diff[0, 0] > rot_thre

    # ---------- debug visualisation ----------

    def _save_debug_image(self) -> None:
        """
        Save a side-by-side debug strip after each inference step:
          RGB | Depth | Distance Field (projected to frame-0) | Frontier Region | Info Gain
        All panels are resized to the same height and labelled before concatenation.
        """
        if self.debug_dir is None or self.ft_detector is None:
            return

        det = self.ft_detector
        PANEL_H = 320  # target height for every panel

        def to_colormap(arr: np.ndarray, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
            """Normalise a 2-D float array to [0,255] and apply a cv2 colormap."""
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            normed = ((arr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(arr, dtype=np.uint8)
            return cv2.applyColorMap(normed, cmap)

        def resize_h(img: np.ndarray, h: int) -> np.ndarray:
            oh, ow = img.shape[:2]
            return cv2.resize(img, (max(1, int(ow * h / oh)), h))

        def add_label(panel: np.ndarray, title: str) -> np.ndarray:
            bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, title, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            return np.vstack([bar, panel])

        panels = []

        # 1. RGB (original resolution, uint8)
        if det.raw_rgb is not None:
            rgb_bgr = cv2.cvtColor(det.raw_rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            panels.append(add_label(resize_h(rgb_bgr, PANEL_H), "RGB"))

        # 2. Depth (original resolution, colourised)
        if det.raw_depth is not None:
            panels.append(add_label(resize_h(to_colormap(det.raw_depth), PANEL_H), "Depth"))

        # 3a. Raw Distance Field — original model output before disocclusion redistribution
        if det.df_raw_pre_redist is not None:
            panels.append(add_label(resize_h(to_colormap(det.df_raw_pre_redist), PANEL_H), "DF raw (pre-redist.)"))

        # 3b. Raw Distance Field — after scattering disocclusion values to visible edge pixels
        if det.df_raw is not None:
            panels.append(add_label(resize_h(to_colormap(det.df_raw), PANEL_H), "DF raw (post-redist.)"))

        # 3b. Distance Field projected to frame-0
        if det.df is not None:
            panels.append(add_label(resize_h(to_colormap(det.df), PANEL_H), "DF (frame-0)"))

        # 4. Frontier Region (binary mask)
        if det.ft_region is not None:
            ft_gray = (det.ft_region * 255).astype(np.uint8)
            panels.append(add_label(resize_h(cv2.cvtColor(ft_gray, cv2.COLOR_GRAY2BGR), PANEL_H), "FT Region"))

        # 5. Info Gain
        if det.info_gain is not None:
            panels.append(add_label(resize_h(to_colormap(det.info_gain), PANEL_H), "Info Gain"))

        # 6. Newly visible at frame-48 — use the disocclusion mask that was already
        #    computed (with the corrected bilinear-neighbour + dilation coverage) by
        #    _redistribute_disocclusion_values, so the panel and the redistribution
        #    always agree on which pixels are disocclusion.
        if det.disocclusion_mask is not None:
            newly_visible = (det.disocclusion_mask * 255).astype(np.uint8)
            panels.append(add_label(
                resize_h(cv2.cvtColor(newly_visible, cv2.COLOR_GRAY2BGR), PANEL_H),
                "New@frame-48 (disoccl.)",
            ))

        if not panels:
            return

        strip = np.hstack(panels)
        os.makedirs(self.debug_dir, exist_ok=True)
        out_path = os.path.join(self.debug_dir, f"step_{self._debug_step:04d}.png")
        cv2.imwrite(out_path, strip)
        logging.info("Saved debug image: %s", out_path)
        self._debug_step += 1

    def _save_debug_image_detr(self) -> None:
        """
        Save a debug image for the DETR mode (matplotlib style, matching plot_detr_predictions):
          lime × = visible [DISC], red × = occluded [PRIO]
          Labels show normalised uv, z, weight, confidence, and occlusion probability.
        """
        if self.debug_dir is None or self.ft_detector is None:
            return

        det = self.ft_detector
        if det.raw_rgb is None:
            return

        slots     = getattr(det, "detr_slots", [])
        n_visible  = sum(1 for s in slots if not s.get("is_occluded", False))
        n_occluded = sum(1 for s in slots if s.get("is_occluded", False))

        model_H, model_W = det.img_size_model
        img_np   = det.raw_rgb[..., :3].astype(np.float32) / 255.0
        orig_H, orig_W = img_np.shape[:2]
        scale_x  = orig_W / model_W
        scale_y  = orig_H / model_H

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.imshow(img_np)

        # Camera info (top-left corner)
        cam_pos  = getattr(det, "_detr_cam_pos", None)
        cam_disp = getattr(det, "_detr_cam_displacement", None)
        if cam_pos is not None:
            info_str = (
                f"step {self._debug_step}  "
                f"cam ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})  "
                f"moved {cam_disp:.3f} m"
            )
            ax.text(6, 20, info_str, color="cyan", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

        for s in slots:
            u_px   = s["u_px"] * scale_x
            v_px   = s["v_px"] * scale_y
            is_occ = s.get("is_occluded", False)
            color  = "red" if is_occ else "lime"

            ax.scatter(u_px, v_px, s=120, c=color, marker="o", linewidths=2)

            raw_w  = s.get("weight_raw", s["weight"])
            eff_w  = s["weight"]
            u_norm = s["u_px"] / model_W
            v_norm = s["v_px"] / model_H
            label  = (
                f"uv=({u_norm:.2f},{v_norm:.2f})  z={s['z']:.1f}m\n"
                f"w={raw_w:.2f}→{eff_w:.2f}  c={s['conf']:.2f}  occ={s['occ']:.2f}"
            )
            ax.annotate(label, xy=(u_px, v_px), xytext=(4, 6),
                        textcoords="offset points", color=color, fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4))

        ax.axis("off")
        ax.set_title(
            f"DETR: lime×=visible  red×=occluded  "
            f"occluded={n_occluded}  visible={n_visible}"
        )
        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img_rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        plt.close(fig)

        panels = [img_rgb]

        # Auxiliary depth panel (if available) — colourised, height-matched
        depth_pred = getattr(det, "detr_depth_pred", None)
        if depth_pred is not None:
            d = depth_pred.astype(np.float32)
            mn, mx = d.min(), d.max()
            normed = ((d - mn) / max(mx - mn, 1e-6) * 255).astype(np.uint8)
            depth_rgb = cv2.cvtColor(cv2.applyColorMap(normed, cv2.COLORMAP_PLASMA),
                                     cv2.COLOR_BGR2RGB)
            ph, dh, dw = img_rgb.shape[0], *depth_rgb.shape[:2]
            depth_rgb = cv2.resize(depth_rgb, (max(1, int(dw * ph / dh)), ph))
            panels.append(depth_rgb)

        strip = np.hstack(panels)
        os.makedirs(self.debug_dir, exist_ok=True)
        out_path = os.path.join(self.debug_dir, f"step_{self._debug_step:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        logging.info("Saved DETR debug image: %s", out_path)
        self._debug_step += 1

    # ---------- main exploration logic ----------

    def exploration(self) -> None:
        """
        Main exploration loop (headless version).
        """
        assert self.renderer is not None
        assert self.ft_manager is not None and self.mapper is not None

        max_steps = self.args.max_steps
        max_time_s = self.args.max_time
        start_time = time.time()

        # Initial mapping bootstrap.
        # When --voxel_grid is set the planner already has the complete global map
        # from setup_system(); we still integrate the first frame into the wavemap
        # (for any future use) but do NOT overwrite the planner's KDTrees with the
        # partial single-frame output.
        rgb, depth0 = self.get_rgbd()
        C_T_W = self.renderer.get_extrinsic()
        W_T_C = np.linalg.inv(C_T_W)
        self.mapper.insert_depth_to_buffer(depth=depth0, transform=W_T_C)
        logging.info("Initial mapping round started.")
        self.mapper.integrate_from_buffer()
        if not self.args.voxel_grid:
            self.mapper.interpolate_occupancy_grid()
            og = self.mapper.get_occupancy_grid()
            self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])
        
        

        while True:
            self._loop_iter += 1

            C_T_W = self.renderer.get_extrinsic()
            W_T_C = np.linalg.inv(C_T_W)
            n_robot_poses = len(self.ft_manager.robot_poses)

            # Fix 2: track how many consecutive iterations n_robot_poses has not grown.
            # When the snap goal equals the robot's position the step counter freezes
            # and the interval-based detect trigger never fires.
            if n_robot_poses == self._last_n_robot_poses:
                self._frozen_iters += 1
            else:
                self._frozen_iters = 0
                self._last_n_robot_poses = n_robot_poses

            rpos = W_T_C[:3, 3]
            logging.info(
                "===== step %04d  robot=(%+.2f,%+.2f,%+.2f)  "
                "frontiers=%d  graveyard=%d  path_remaining=%d =====",
                n_robot_poses,
                rpos[0], rpos[1], rpos[2],
                len(self.ft_manager.valid_frontiers),
                len(self.ft_manager._graveyard),
                len(self.path_to_go),
            )

            if n_robot_poses > max_steps:
                logging.info("Maximum steps reached, exploration finished.")
                break

            if time.time() - start_time > max_time_s:
                logging.info("Time limit reached, exploration finished.")
                break

            # Safety: bound runaway loops that don't advance n_robot_poses.
            _MAX_LOOP_ITERS = max_steps * 15
            if self._loop_iter > _MAX_LOOP_ITERS:
                logging.warning(
                    "Max loop iterations (%d) reached without enough robot progress — "
                    "terminating to avoid infinite loop.",
                    _MAX_LOOP_ITERS,
                )
                break

            no_more_frontier = (
                len(self.ft_manager.valid_frontiers) == 0 and n_robot_poses >= 1
            )
            # Fix 2: also force detection after detect_interval*2 frozen iterations so
            # a stuck snap goal doesn't permanently suppress the detection trigger.
            _FROZEN_DETECT_EVERY = self.detect_interval * 2
            frozen_force_detect = (
                self._frozen_iters > 0
                and self._frozen_iters % _FROZEN_DETECT_EVERY == 0
            )
            should_detect = no_more_frontier or (
                n_robot_poses % self.detect_interval == 0
            ) or frozen_force_detect

            # Replan only when there is no active path, or the current goal
            # frontier has disappeared (filtered / graveyard). Detection alone
            # no longer forces a mid-path goal switch: new frontiers are added
            # to the pool but the robot keeps heading toward its current goal.
            valid_ids = {ft.id for ft in self.ft_manager.valid_frontiers}
            current_goal_alive = (
                self.ft_manager.current_goal_ft_id is not None
                and self.ft_manager.current_goal_ft_id in valid_ids
            )
            should_replan = not self.path_to_go or not current_goal_alive

            detect_reason = (
                "no_frontiers" if no_more_frontier
                else "frozen" if frozen_force_detect
                else f"interval({n_robot_poses}%{self.detect_interval}==0)" if should_detect
                else "skip"
            )
            replan_reason = (
                "path_empty" if not self.path_to_go
                else "goal_gone" if not current_goal_alive
                else "skip"
            )
            logging.info("  detect=%s  replan=%s", detect_reason, replan_reason)

            ft_list = None  # Fix 3: initialise so it's visible outside the detect block
            if should_detect:
                logging.info("Running frontier detection (step %d).", n_robot_poses)
                # Record this as an observed pose so the robot_path_filter knows
                # the robot genuinely observed this location (not just passed by).
                self.ft_manager.add_detection_pose(W_T_C)
                rgb, depth = self.get_rgbd()

                # Frontier detection — branch on model type
                if self.args.model_type == "classic":
                    # Classic map-based frontier detection (Yamauchi 1997).
                    # Always interpolate the wavemap at detect steps so we have
                    # the robot's current observed free/occupied space, regardless
                    # of whether --voxel_grid is set (the voxel_grid is used only
                    # for path planning, not for frontier detection).
                    self.mapper.interpolate_occupancy_grid()
                    og = self.mapper.get_occupancy_grid()
                    ft_list = self.classic_detector.detect(
                        free_pts=og["free"],
                        occ_pts=og["occupied"],
                        W_T_C=W_T_C,
                    )
                elif self.args.model_type in {"unet_detr", "detr", "cond_detr"}:
                    # DETR: goals are (u,v,z) 3-D points; GMM weight = info-gain proxy.
                    # Occluded goals are handled transparently by the path planner.
                    ft_list = self.ft_detector.detect_detr(
                        rgb=rgb,
                        extrinsic=C_T_W,
                        conf_thresh=self.args.detr_conf_thresh,
                        gain_scale=self.args.detr_gain_scale,
                        visible_gain_discount=self.args.detr_visible_gain_discount,
                    )
                    self._save_debug_image_detr()
                else:
                    # Dense DPT/UNet pipeline
                    self.ft_detector.detect(
                        rgb=rgb,
                        depth=depth,
                        df_thr=self.config["df_thr"],
                    )
                    self._save_debug_image()
                    ft_list = self.ft_detector.anchor_fts(depth=depth, extrinsic=C_T_W)

                # Clamp frontier z to planning bounds [z_min, z_max]
                if ft_list and self.config.get("bounds") is not None:
                    z_min = float(self.config["bounds"][4])
                    z_max = float(self.config["bounds"][5])
                    for ft in ft_list:
                        ft.pos3d[2] = float(np.clip(ft.pos3d[2], z_min, z_max))

                # Add into manager
                if ft_list:
                    if self.args.model_type in {"unet_detr", "detr", "cond_detr", "classic"}:
                        ft_list = self.ft_manager.dedup_new_frontiers(
                            ft_list, radius=self.args.detr_dedup_radius
                        )
                    # Reuse the most recent robot pose if the robot hasn't moved
                    # since it was recorded, to avoid creating duplicate graph nodes
                    # at the same position every detection cycle.
                    cur_pos = W_T_C[:3, 3]
                    last_rid = self.ft_manager.current_robot_id
                    last_pose = (
                        self.ft_manager.robot_poses.get(last_rid)
                        if last_rid is not None else None
                    )
                    if (
                        last_pose is not None
                        and float(np.linalg.norm(cur_pos - last_pose[:3, 3])) < 0.1
                    ):
                        parent_ids = [last_rid]
                    else:
                        parent_ids = self.ft_manager.add_robot_poses([W_T_C])
                    self.ft_manager.add_frontiers(frontiers=ft_list, parent_ids=parent_ids)
                    self.ft_manager.filter_frontiers()

                if len(self.ft_manager.valid_frontiers) == 0:
                    logging.info("No frontiers, exploration finished.")
                    break

            # Update mapper continuously.
            # When --voxel_grid is set the planner uses the pre-loaded global map
            # throughout; wavemap still accumulates observations but its (partial)
            # output is not used to overwrite the planner's KDTrees.
            self.mapper.integrate_from_buffer()
            if not self.args.voxel_grid:
                self.mapper.interpolate_occupancy_grid()
                og = self.mapper.get_occupancy_grid()
                self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])

            # Fix 3: gain_adjustment is O(N*M) in frontiers × robot poses.  Only
            # rerun it when the inputs have actually changed: new robot poses were
            # added (step counter advanced) or new frontiers were just detected.
            # update_utility always runs because it depends on the current robot
            # position, which can change even without new poses being recorded.
            _gain_inputs_changed = (
                n_robot_poses > self._last_gain_adj_n_poses
                or bool(ft_list)
            )
            if _gain_inputs_changed:
                if self.args.model_type in {"unet_detr", "detr", "cond_detr", "classic"}:
                    self.ft_manager.gain_adjustment_detr()
                else:
                    self.ft_manager.gain_adjustment()
                self.ft_manager.filter_frontiers()
                self.ft_manager.merge_frontiers()
                self.ft_manager.filter_frontiers()
                self._last_gain_adj_n_poses = n_robot_poses
            self.ft_manager.update_utility(current_pos=W_T_C[:3, 3])

            # Replan if needed
            if should_replan and (self.move_enough or not self.path_to_go):
                logging.info("Replanning...")
                logging.debug(f"Replanning (interval={self.plan_interval}).")
                self.path_to_go = self.ft_manager.plan_path_to_goal(
                    W_T_C,
                    direct_voxel_snap=bool(self.args.voxel_grid),
                ) or []
                if self.path_to_go:
                    logging.info(
                        f"Path to goal found with {len(self.path_to_go)} steps."
                    )
                    self.move_enough = False
                else:
                    logging.warning("No path found, deleting current goal frontier.")
                    self._recent_positions.clear()
                    self.path_to_go = []
                    self.move_enough = True  # try again next cycle

            # Safety termination: if there is nothing to navigate toward and no
            # frontiers remain (e.g. all dropped by filter or stuck-detection),
            # stop rather than spinning until max_time_s.
            if not self.path_to_go and len(self.ft_manager.valid_frontiers) == 0:
                logging.info(
                    "No valid frontiers and no active path — exploration finished."
                )
                break

            # Persist state snapshot
            if self.json_path:
                logging.info(f"Writing state to {self.json_path}")
                self.ft_manager.write_to_file(file_path=self.json_path)

            # Execute one movement step if path exists.
            # Stuck detection runs inside this block so that stationary frames
            # (path exhausted, waiting for replanning or detection) never
            # contribute zero displacement and trigger a false stuck alarm.
            if self.path_to_go:
                logging.debug("Moving along the path.")
                self.move(steps=1)

                # Record position and check cumulative displacement only while
                # actively following a path. If the robot covers less than
                # _stuck_disp_threshold metres over _stuck_window consecutive
                # move-steps, the current goal is physically unreachable.
                self._recent_positions.append(W_T_C[:3, 3].copy())
                if len(self._recent_positions) == self._recent_positions.maxlen:
                    pts = list(self._recent_positions)
                    cumulative = sum(
                        float(np.linalg.norm(pts[i + 1] - pts[i]))
                        for i in range(len(pts) - 1)
                    )
                    if cumulative < self._stuck_disp_threshold:
                        goal_id = self.ft_manager.current_goal_ft_id
                        if goal_id is not None:
                            logging.warning(
                                "Stuck detected: cumulative displacement %.2f m over %d steps "
                                "< %.2f m — dropping frontier %s.",
                                cumulative, self._stuck_window,
                                self._stuck_disp_threshold, goal_id,
                            )
                            self.ft_manager.remove_frontiers([goal_id])
                            self._recent_positions.clear()
                            self.path_to_go = []
                            self.move_enough = True

        # Final state output
        logging.info("Exploration finished, total steps: %d", n_robot_poses)

    # ---------- motion & mapping ----------

    def move(self, steps: int) -> None:
        """
        Execute up to `steps` motions along the path, acquire depth, and update mapper & manager.
        """
        if self.renderer is None:
            return
        if not self.path_to_go:
            logging.info("No path to follow.")
            return

        for _ in range(steps):
            if not self.path_to_go:
                logging.info("Path exhausted.")
                break

            next_W_T_C = self.path_to_go.pop(0)
            logging.debug(f"Moving to next pose:\n{next_W_T_C}")

            # Update renderer camera extrinsic (needs C_T_W)
            self.renderer.set_extrinsic(np.linalg.inv(next_W_T_C))

            # Capture new depth
            _, depth = self.get_rgbd()

            # Insert into mapper
            C_T_W = self.renderer.get_extrinsic()
            W_T_C = np.linalg.inv(C_T_W)
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C)

            # Check if we truly moved
            if self.is_moving(W_T_C):
                self.last_W_T_C = W_T_C
                if self.ft_manager is not None:
                    self.ft_manager.add_robot_poses([W_T_C])
                self.move_enough = True

    # ---------- setup ----------

    # ---------- global map helpers ----------

    def _load_free_from_voxel_grid(self, ply_path: str) -> None:
        """
        Populate self.global_free_pts from a pre-computed navigable free-space
        voxel grid PLY file (e.g. eval_data/voxel_grid/000876-voxel_grid.ply).

        The file stores the ground-truth traversable voxel centres for the scene,
        so loading it makes the planner's free-space KDTree complete from step 0
        rather than being built up incrementally from wavemap observations.
        """
        logging.info("Loading free-space voxel grid from: %s", ply_path)
        vg = o3d.io.read_voxel_grid(ply_path)
        voxels = vg.get_voxels()
        if not voxels:
            logging.warning("Voxel grid file contained no voxels; global_free_pts unchanged.")
            return

        free_centers = np.array(
            [vg.get_voxel_center_coordinate(v.grid_index) for v in voxels],
            dtype=np.float32,
        )

        # Restrict to planning bounds so the KDTree stays tight
        bounds = self.config.get("bounds")
        if bounds is not None and len(free_centers) > 0:
            b = bounds
            mask = (
                (free_centers[:, 0] >= b[0]) & (free_centers[:, 0] <= b[1])
                & (free_centers[:, 1] >= b[2]) & (free_centers[:, 1] <= b[3])
                & (free_centers[:, 2] >= b[4]) & (free_centers[:, 2] <= b[5])
            )
            free_centers = free_centers[mask]

        self.global_free_pts = free_centers
        logging.info(
            "global_free_pts set to %d voxels from file (voxel_size=%.3f m).",
            len(free_centers), vg.voxel_size,
        )

    def _build_occ_from_mesh(self) -> None:
        """
        Populate self.global_occ_pts by voxelising the scene mesh surfaces.
        Used together with _load_free_from_voxel_grid() to give the planner
        a complete occupied-space KDTree without deriving it from wavemap.
        """
        from scipy.spatial import KDTree as _KDTree

        logging.info("Building occupied voxel map from mesh: %s", self.args.mesh)
        model = o3d.io.read_triangle_model(self.args.mesh)
        combined = o3d.geometry.TriangleMesh()
        for mesh_info in model.meshes:
            combined += mesh_info.mesh

        vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            combined, voxel_size=self.VOX_SIZE
        )
        occ_centers = np.array(
            [vg.get_voxel_center_coordinate(v.grid_index) for v in vg.get_voxels()],
            dtype=np.float32,
        )

        bounds = self.config.get("bounds")
        if bounds is not None and len(occ_centers) > 0:
            b = bounds
            mask = (
                (occ_centers[:, 0] >= b[0]) & (occ_centers[:, 0] <= b[1])
                & (occ_centers[:, 1] >= b[2]) & (occ_centers[:, 1] <= b[3])
                & (occ_centers[:, 2] >= b[4]) & (occ_centers[:, 2] <= b[5])
            )
            occ_centers = occ_centers[mask]

        self.global_occ_pts = occ_centers
        logging.info(
            "global_occ_pts set to %d occupied voxels from mesh.",
            len(occ_centers),
        )

    def setup_system(self) -> None:
        """Initialize renderer, mapper, detector, and manager."""
        # Configure logging level
        if self.args.log_level < 10:
            logging.getLogger().setLevel(logging.NOTSET)
        elif self.args.log_level < 20:
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.args.log_level < 30:
            logging.getLogger().setLevel(logging.INFO)
        elif self.args.log_level < 40:
            logging.getLogger().setLevel(logging.WARNING)
        elif self.args.log_level < 50:
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.CRITICAL)

        # Load scene mesh (just for logging, HeadlessRenderer will load it properly)
        logging.info(f"Loading mesh from: {self.args.mesh}")

        # Create camera intrinsics
        cam_intrinsic = create_camera(self.CAM_H, self.CAM_W, self.CAM_F)

        # Create headless renderer - pass mesh path for proper texture loading
        self.renderer = HeadlessRenderer(
            mesh_path=self.args.mesh,
            width=self.CAM_W,
            height=self.CAM_H,
            intrinsic=cam_intrinsic,
            z_near=0.02,
            z_far=50.0,
        )

        # Set initial camera pose: Seed takes priority over Config
        if self.args.seed is not None:
            initial_C_T_W = self.generate_random_pose(self.args.seed)
        else:
            logging.info("Using initial_cam_extrinsic from config file.")
            initial_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], dtype=float)
            
        self.renderer.set_extrinsic(initial_C_T_W)

        # Save starting pose RGB for manual inspection
        _start_rgb = self.renderer.capture_rgb()
        _start_path = os.path.join(os.path.dirname(__file__), "visualize", "starting_pose.png")
        cv2.imwrite(_start_path, cv2.cvtColor(_start_rgb, cv2.COLOR_RGB2BGR))
        # import pdb; pdb.set_trace()
        logging.info("Saved starting pose image: %s", _start_path)

        self.last_W_T_C = np.linalg.inv(initial_C_T_W)

        # Mapper
        intr = cam_intrinsic
        params = {
            "min_cell_width": self.VOX_SIZE / 2.0,
            "width": intr.width,
            "height": intr.height,
            "fx": intr.intrinsic_matrix[0, 0],
            "fy": intr.intrinsic_matrix[1, 1],
            "cx": intr.intrinsic_matrix[0, 2],
            "cy": intr.intrinsic_matrix[1, 2],
            "min_range": 0.05,
            "max_range": (
                self.config["depth_range"]
                if self.config["depth_range"] is not None
                else 3.5
            ),
            "resolution": self.VOX_SIZE,
        }
        self.mapper = WaveMapper(params=params)

        # FrontierNet detector — either neural network or classic map-based
        model_type = self.args.model_type
        if model_type == "classic":
            # Classic Yamauchi (1997) frontier detection — no neural network.
            self.ft_detector = None
            self.classic_detector = ClassicFrontierDetector(
                voxel_size=self.VOX_SIZE,
                camera_intrinsic=intr.intrinsic_matrix.copy(),
                min_frontier_size=self.args.classic_min_frontier_size,
                cluster_eps=self.args.classic_cluster_eps,
                log_level=self.args.log_level,
            )
            logging.info(
                "Classic frontier detector initialised "
                "(voxel_size=%.2f, cluster_eps=%.2f, min_size=%d).",
                self.VOX_SIZE,
                self.args.classic_cluster_eps,
                self.args.classic_min_frontier_size,
            )
        else:
            use_depth = True
            feature_layers = (
                [int(x) for x in self.args.vit_feature_layers.split(",")]
                if self.args.vit_feature_layers
                else None
            )
            net = load_model(
                path=self.args.unet_weight,
                num_classes=self.config["num_classes"],
                use_depth=use_depth,
                model_type=model_type,
                vit_depth=self.args.vit_depth,
                feature_layers=feature_layers,
                num_queries=self.args.detr_num_queries,
                detr_aux_depth=self.args.detr_aux_depth,
                factory_d_model=self.args.factory_d_model,
                factory_n_tokens=self.args.factory_n_tokens,
                factory_n_layers=self.args.factory_n_layers,
                factory_n_heads=self.args.factory_n_heads,
                factory_dropout=self.args.factory_dropout,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ft_detector = FrontierDetector(
                model=net,
                camera_intrinsic=intr.intrinsic_matrix.copy(),
                use_depth=use_depth,
                img_size_model=self.config["input_img_size"],
                device=device,
                log_level=self.args.log_level,
                model_type=model_type,
                disocclusion_only=self.args.disocclusion_only,
                edge_spread_width=self.args.edge_spread_width,
            )

        # Frontier Manager
        self.ft_manager = FrontierManager(
            params=self.config, log_level=self.args.log_level
        )

        # When --voxel_grid is provided, pre-load the complete free-space and
        # occupied maps so the planner knows the entire scene from step 0.
        # The per-step wavemap updates in exploration() are then skipped so
        # this known map is not overwritten by partial observations.
        if self.args.voxel_grid:
            self._load_free_from_voxel_grid(self.args.voxel_grid)
            self._build_occ_from_mesh()
            self.ft_manager.update_map(
                free_map=self.global_free_pts,
                occ_map=self.global_occ_pts,
            )
            logging.info(
                "Planner initialised with global map: %d free, %d occupied voxels.",
                len(self.global_free_pts) if self.global_free_pts is not None else 0,
                len(self.global_occ_pts) if self.global_occ_pts is not None else 0,
            )

        # Wipe and recreate the output directory so each run starts clean
        # (prevents multiple runs from being concatenated into the same JSON).
        if self.json_path:
            out_dir = os.path.dirname(self.json_path)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

        logging.info("Headless system setup complete.")

    
    def cleanup(self) -> None:
        """Clean up resources safely."""
        if self.renderer is not None:
            # Just call the internal cleanup if it exists, 
            # but don't manually delete sub-attributes.
            try:
                self.renderer.cleanup() 
            except:
                pass
            self.renderer = None # Set to None and let GC handle the rest
        
        import gc
        gc.collect()

    def run(self) -> None:
        """Run the headless exploration."""
        try:
            self.setup_system()
            logging.info("Starting headless exploration...")
            self.exploration()
            logging.info("Headless exploration complete.")
        finally:
            self.cleanup()

    def generate_random_pose(self, seed: int) -> np.ndarray:
        """
        Generates a random initial camera extrinsic matrix (C_T_W)
        based on the orientation style of the provided config.
        """
        rng = np.random.default_rng(seed)
        
        # 1. Random Position (W_P_C)
        # Based on your config translation [0.29, 1.0, 0.27]
        # We keep Z around 1.0 (height) and vary X/Y
        x = rng.uniform(-3, 3)
        y = rng.uniform(-3, 3)
        z = 1.0  # Stable height based on your config's y-translation
        pos = np.array([x, y, z])

        # 2. Define Look-At Direction
        # Point toward the origin (0,0,0) or a random point near it
        target = rng.uniform(-0.5, 0.5, size=3)
        target[2] = z  # Keep the "gaze" horizontal
        
        forward = target - pos
        forward /= np.linalg.norm(forward)
        
        # 3. Align axes with the provided extrinsic style
        # In your matrix, world Z maps to camera -Y (the camera is 'upright')
        up_world = np.array([0, 0, 1])
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward) # Camera Y
        
        # 4. Construct W_R_C (Camera orientation in World)
        # The extrinsic matrix is C_T_W (World to Camera)
        # C_R_W = W_R_C.T
        W_R_C = np.zeros((3, 3))
        W_R_C[:, 0] = right    # Camera X
        W_R_C[:, 1] = -up      # Camera Y (aligned with your config's -1.0 Z mapping)
        W_R_C[:, 2] = forward  # Camera Z
        
        C_R_W = W_R_C.T
        
        # 5. Build final 4x4 matrix
        initial_C_T_W = np.eye(4)
        initial_C_T_W[:3, :3] = C_R_W
        initial_C_T_W[:3, 3] = -C_R_W @ pos
        
        logging.info(f"Generated seeded initial pose at {pos} looking toward {target}")
        return initial_C_T_W
    
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Headless FrontierNet exploration demo (no display required)"
    )
    p.add_argument("--mesh", type=str, required=True, help="Path to the mesh file")
    p.add_argument(
        "--voxel_grid",
        type=str,
        default=None,
        help=(
            "Path to a pre-computed free-space voxel grid PLY file "
            "(e.g. eval_data/voxel_grid/000876-voxel_grid.ply). "
            "When provided the planner's free-space KDTree is built from this "
            "file and the occupied KDTree from the mesh, so the entire navigable "
            "area is known from step 0. Per-step wavemap-based map updates are "
            "skipped, keeping the known map fixed throughout exploration."
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_exploration.yaml",
        help="FrontierNet configuration file",
    )
    p.add_argument(
        "--write_path", type=str, help="JSON file to write the ftmanager state"
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of exploration steps",
    )
    p.add_argument(
        "--max_time", type=int, default=1800, help="Maximum exploration time in seconds"
    )
    p.add_argument(
        "--unet_weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
        help="Path to model weights (UNet or DPT checkpoint)",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="dpt",
        choices=["dpt", "unet", "unet_detr", "detr", "cond_detr", "classic"],
        help=(
            "Model architecture: "
            "'dpt' (ViT+DPT, RGB-only, dense df_seg head), "
            "'unet' (ResNet34+UNet, RGB-D, dense df_seg head), "
            "'unet_detr' (ResNet34+UNet, RGB-only, sparse DETR head), "
            "'detr' (FrontierDETR: ResNet50+enc+dec, DETR-Factory), "
            "'cond_detr' (FrontierConditionalDETR: ResNet50+enc+cond dec, DETR-Factory), "
            "'classic' (Yamauchi 1997 map-based, no neural network)"
        ),
    )
    p.add_argument(
        "--vit_depth",
        type=int,
        default=6,
        help="(DPT only) Number of ViT transformer layers used during training",
    )
    p.add_argument(
        "--vit_feature_layers",
        type=str,
        default="3,4,5",
        help="(DPT only) Comma-separated ViT layer indices used for DPT fusion, e.g. '4,5,6,7'. "
             "Defaults to the last min(4, vit_depth) layers.",
    )
    p.add_argument(
        "--depth_source",
        type=str,
        default=HeadlessExplorerApp.DEPTH_GT,
        choices=[
            HeadlessExplorerApp.DEPTH_GT,
            HeadlessExplorerApp.DEPTH_M3D,
            HeadlessExplorerApp.DEPTH_UNIK3D,
        ],
        help="Depth source",
    )
    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="logging level (0=notset, 10=debug, 20=info...)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for generating random initial camera extrinsics. If None, uses config."
    )
    p.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="Directory to save per-step debug visualisation strips (RGB|Depth|DF|FT|Gain). "
             "Defaults to <json_parent>/debug. Set to empty string to disable.",
    )
    p.add_argument(
        "--disocclusion_only",
        action="store_true",
        default=False,
        help="If set, df_raw post-redistribution contains only the pixels that received "
             "contributions from disocclusion regions; all unmoved (originally covered) "
             "pixels are zeroed out.",
    )
    p.add_argument(
        "--edge_spread_width",
        type=int,
        default=15,
        help="Width in pixels of the horizontal dilation applied to df_raw after "
             "disocclusion redistribution (must be odd for symmetric spread; default 20).",
    )
    # --- DETR-specific args (unet_detr mode only) ---
    p.add_argument(
        "--detr_conf_thresh",
        type=float,
        default=0.3,
        help="(unet_detr) Minimum slot confidence to accept as a frontier candidate.",
    )
    p.add_argument(
        "--detr_num_queries",
        type=int,
        default=10,
        help="(unet_detr / detr / cond_detr) Number of DETR slot queries (must match the trained checkpoint).",
    )
    p.add_argument(
        "--factory_d_model",
        type=int,
        default=256,
        help="(detr / cond_detr) Transformer embedding dimension. Default 256.",
    )
    p.add_argument(
        "--factory_n_tokens",
        type=int,
        default=374,
        help=(
            "(detr / cond_detr) Number of spatial tokens for the training resolution. "
            "Default 374 = (544//32)×(720//32) for 544×720 input."
        ),
    )
    p.add_argument(
        "--factory_n_layers",
        type=int,
        default=6,
        help="(detr / cond_detr) Number of encoder and decoder layers. Default 6.",
    )
    p.add_argument(
        "--factory_n_heads",
        type=int,
        default=8,
        help="(detr / cond_detr) Number of attention heads. Default 8.",
    )
    p.add_argument(
        "--factory_dropout",
        type=float,
        default=0.1,
        help="(detr / cond_detr) Dropout rate in transformer layers. Must match training. Default 0.1.",
    )
    p.add_argument(
        "--detr_aux_depth",
        action="store_true",
        default=False,
        help=(
            "(unet_detr) Set this flag when the checkpoint was trained with an "
            "auxiliary dense depth supervision head (detr_aux_depth=True in "
            "TwoHeadUnet). Must match the training configuration so the state-dict "
            "keys align; also enables the aux-depth panel in debug images."
        ),
    )
    p.add_argument(
        "--detr_gain_scale",
        type=float,
        default=100.0,
        help=(
            "(unet_detr) Multiplicative scale applied to the raw GMM weight before "
            "it is stored as frontier gain.  The dense pipeline produces gains in "
            "[~1, 9] (midpoint × 10); set this so DETR weights pass filter_min_gain. "
            "Default 10.0."
        ),
    )
    p.add_argument(
        "--detr_dedup_radius",
        type=float,
        default=0.5,
        help=(
            "(unet_detr) Exclusion radius in metres for cross-frame deduplication. "
            "A newly detected frontier is dropped if any existing valid frontier "
            "lies within this distance. Default 0.5 m."
        ),
    )
    p.add_argument(
        "--detr_visible_gain_discount",
        type=float,
        default=0.1,
        help=(
            "(unet_detr) Multiplicative discount applied to the gain of non-occluded "
            "(already-visible) frontiers. Occluded frontiers keep their full gain so "
            "the planner prefers exploring hidden/unseen areas. Default 0.1."
        ),
    )
    # --- Classic (Yamauchi 1997) args ---
    p.add_argument(
        "--classic_min_frontier_size",
        type=int,
        default=5,
        help=(
            "(classic) Minimum number of frontier voxels in a DBSCAN cluster to be "
            "accepted as a valid frontier. Smaller clusters are treated as noise. "
            "Default 5."
        ),
    )
    p.add_argument(
        "--classic_cluster_eps",
        type=float,
        default=0.4,
        help=(
            "(classic) DBSCAN neighbourhood radius (metres) for clustering frontier "
            "voxels into frontier regions. Default 0.4 m."
        ),
    )
    return p


def main():
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )
    print(f"Open3D version: {o3d.__version__}")

    parser = build_arg_parser()
    args = parser.parse_args()

    app = HeadlessExplorerApp(args)
    app.run()


if __name__ == "__main__":
    main()