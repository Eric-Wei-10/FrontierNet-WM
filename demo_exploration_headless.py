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
from frontier.mapex_detector import MapExFrontierDetector
from frontier.model.predict import load_model, DETR_MODEL_TYPES
from utils.frontier_utils import read_config_yaml

# Mapping
from mapping.wavemap import WaveMapper

# Frontier Manager
from frontier.manager import FrontierManager


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

        # Depth source
        self.depth_source = args.depth_source

        # Headless renderer
        self.renderer: Optional[HeadlessRenderer] = None

        # Frontier, mapping, detector
        self.mapper: Optional[WaveMapper] = None
        self.ft_manager: Optional[FrontierManager] = None
        self.ft_detector: Optional[FrontierDetector] = None
        self.mapex_detector: Optional[MapExFrontierDetector] = None
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

        # Stuck detection: fires only when the robot occupies the EXACT same pose
        # (position + orientation) for all steps in the window AND no waypoints
        # have been consumed.  This catches hard blocks without false-positives from
        # rotation-heavy or slow path segments.
        self._stuck_window: int = 5
        self._recent_poses: deque = deque(maxlen=self._stuck_window + 1)
        self._recent_path_remaining: deque = deque(maxlen=self._stuck_window + 1)

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
            from mono_depth.UniK3D import metric_depth_from_rgb as metric_depth_from_rgb_unik3d
            depth = metric_depth_from_rgb_unik3d(rgb_input=rgb, intrinsic_mat=K)
        elif self.depth_source == self.DEPTH_M3D:
            from mono_depth.Metric3D import metric_depth_from_rgb as metric_depth_from_rgb_metric3d
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

            self.renderer.set_extrinsic(np.linalg.inv(next_W_T_C))

            _, depth = self.get_rgbd()

            C_T_W = self.renderer.get_extrinsic()
            W_T_C = np.linalg.inv(C_T_W)
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C)

            if self.is_moving(W_T_C):
                self.last_W_T_C = W_T_C
                if self.ft_manager is not None:
                    self.ft_manager.add_robot_poses([W_T_C])
                self.move_enough = True

    def _load_free_from_voxel_grid(self, ply_path: str) -> None:
        """
        Populate self.global_free_pts from a pre-computed navigable free-space
        voxel grid PLY file (e.g. eval_data/voxel_grid/000876-voxel_grid.ply).
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

        logging.info(f"Loading mesh from: {self.args.mesh}")

        cam_intrinsic = create_camera(self.CAM_H, self.CAM_W, self.CAM_F)

        self.renderer = HeadlessRenderer(
            mesh_path=self.args.mesh,
            width=self.CAM_W,
            height=self.CAM_H,
            intrinsic=cam_intrinsic,
            z_near=0.02,
            z_far=50.0,
        )

        if self.args.seed is not None:
            initial_C_T_W = self.generate_random_pose(self.args.seed)
        else:
            logging.info("Using initial_cam_extrinsic from config file.")
            initial_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], dtype=float)

        self.renderer.set_extrinsic(initial_C_T_W)

        _start_rgb = self.renderer.capture_rgb()
        _start_path = os.path.join(os.path.dirname(__file__), "visualize", "starting_pose.png")
        cv2.imwrite(_start_path, cv2.cvtColor(_start_rgb, cv2.COLOR_RGB2BGR))
        logging.info("Saved starting pose image: %s", _start_path)

        self.last_W_T_C = np.linalg.inv(initial_C_T_W)

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

        model_type = self.args.model_type
        if model_type == "mapex":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mapex_detector = MapExFrontierDetector(
                mapex_dir=self.args.mapex_dir,
                device=device,
                map_size=self.args.mapex_map_size,
                map_margin_m=self.args.mapex_map_margin,
                min_frontier_size=self.args.mapex_min_frontier_size,
                gain_scale=self.args.mapex_gain_scale,
                voxel_size=self.VOX_SIZE,
                log_level=self.args.log_level,
            )
            logging.info(
                "MapEx frontier detector initialised "
                "(map_size=%d, margin=%.1f m, min_ft=%d, gain_scale=%.0f).",
                self.args.mapex_map_size,
                self.args.mapex_map_margin,
                self.args.mapex_min_frontier_size,
                self.args.mapex_gain_scale,
            )
        elif model_type in DETR_MODEL_TYPES:
            net = load_model(
                path=self.args.unet_weight,
                model_type=model_type,
                num_queries=self.args.detr_num_queries,
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
                img_size_model=self.config["input_img_size"],
                device=device,
                log_level=self.args.log_level,
                model_type=model_type,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.ft_manager = FrontierManager(
            params=self.config, log_level=self.args.log_level
        )

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

    def exploration_baseline(self) -> None:
        """
        Main exploration loop.

        Detection and replanning are coupled on the same predict_interval cadence.
        gain_adjustment() (volumetric) is used for all model types.
        Terminates when all frontiers are exhausted.
        """
        assert self.renderer is not None
        assert self.ft_manager is not None and self.mapper is not None

        max_steps = self.args.max_steps
        max_time_s = self.args.max_time
        start_time = time.time()

        predict_interval: int = int(
            self.config.get("predict_interval", self.config.get("detect_interval", 6))
        )
        logging.info(
            "exploration_baseline START  model_type=%s  predict_interval=%d  "
            "max_steps=%d  max_time=%.0fs",
            self.args.model_type, predict_interval, max_steps, max_time_s,
        )

        # Bootstrap: integrate first frame into wavemap
        rgb, depth0 = self.get_rgbd()
        C_T_W = self.renderer.get_extrinsic()
        W_T_C = np.linalg.inv(C_T_W)
        self.mapper.insert_depth_to_buffer(depth=depth0, transform=W_T_C)
        logging.info("Initial mapping round started.")
        self.mapper.integrate_from_buffer()
        if self.args.model_type == "mapex":
            for _ in range(9):
                self.mapper.insert_depth_to_buffer(depth=depth0, transform=W_T_C)
                self.mapper.integrate_from_buffer()
        self.mapper.interpolate_occupancy_grid()
        og = self.mapper.get_occupancy_grid()
        if self.args.voxel_grid:
            # Keep planner's global KDTree intact; only refresh gain maps.
            self.ft_manager.free_map = og["free"]
            self.ft_manager.occ_map = og["occupied"]
        else:
            self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])

        while True:
            C_T_W = self.renderer.get_extrinsic()
            W_T_C = np.linalg.inv(C_T_W)
            n_robot_poses = len(self.ft_manager.robot_poses)

            logging.info(
                "===== step %04d  frontiers=%d  path_remaining=%d =====",
                n_robot_poses,
                len(self.ft_manager.valid_frontiers),
                len(self.path_to_go),
            )

            if n_robot_poses > max_steps:
                logging.info("Maximum steps reached, exploration finished.")
                break
            if time.time() - start_time > max_time_s:
                logging.info("Time limit reached, exploration finished.")
                break

            no_more_frontier = (
                len(self.ft_manager.valid_frontiers) == 0 and n_robot_poses > 10
            )
            reach_next_update = len(self.path_to_go) == 0 or (
                (n_robot_poses - 1) % predict_interval == 0
            )

            if no_more_frontier or reach_next_update:
                logging.info("Updating frontiers (step %d).", n_robot_poses)
                rgb, depth = self.get_rgbd()

                if self.args.model_type == "mapex":
                    self.mapper.interpolate_occupancy_grid()
                    og_det = self.mapper.get_occupancy_grid()
                    cam_z = float(W_T_C[2, 3])
                    z_filter = (cam_z - 1.5, cam_z + 0.8)
                    ft_list = self.mapex_detector.detect(
                        free_pts=og_det["free"],
                        occ_pts=og_det["occupied"],
                        W_T_C=W_T_C,
                        z_filter=z_filter,
                    )
                else:  # detr / cond_detr
                    ft_list = self.ft_detector.detect_detr(
                        rgb=rgb,
                        extrinsic=C_T_W,
                        conf_thresh=self.args.detr_conf_thresh,
                        gain_scale=self.args.detr_gain_scale,
                    )

                if ft_list:
                    _g = [f.gain for f in ft_list]
                    logging.info(
                        "  raw detections: %d  gain min/mean/max = %.2f / %.2f / %.2f",
                        len(ft_list), min(_g), float(np.mean(_g)), max(_g),
                    )
                else:
                    logging.info("  raw detections: 0")


                # Debug mode: immediately drop frontiers that are not reachable according
                # to the pre-loaded free-space voxel grid.  Only meaningful when --voxel_grid
                # is also set (so _free_kdt exists).  Uses freespace_filter_dist from config
                # (default 0 → strict isfree() check).
                if ft_list and self.args.free_vox_filter:
                    kdt = self.ft_manager.planner._free_kdt
                    if kdt is None:
                        logging.warning(
                            "free_vox_filter is set but _free_kdt is None "
                            "(did you pass --voxel_grid?); filter skipped."
                        )
                    else:
                        thresh = self.ft_manager.freespace_filter_dist
                        before = len(ft_list)
                        if thresh > 0.0:
                            ft_list = [
                                ft for ft in ft_list
                                if float(kdt.query(np.asarray(ft.pos3d, dtype=float))[0]) <= thresh
                            ]
                        else:
                            ft_list = [
                                ft for ft in ft_list
                                if self.ft_manager.planner.isfree(np.asarray(ft.pos3d, dtype=float))
                            ]
                        dropped = before - len(ft_list)
                        if dropped:
                            logging.info(
                                "free_vox_filter: dropped %d / %d frontiers not in free space "
                                "(thresh=%.2f m).",
                                dropped, before,
                                thresh if thresh > 0.0 else 0.0,
                            )

                if ft_list:
                    new_ids = self.ft_manager.add_robot_poses([W_T_C])
                    self.ft_manager.add_frontiers(frontiers=ft_list, parent_ids=new_ids)
                    self.ft_manager.filter_frontiers()
                    self.ft_manager.gain_adjustment()
                    self.ft_manager.filter_frontiers()
                    _vf = self.ft_manager.valid_frontiers
                    if _vf:
                        _ug = [f.u_gain for f in _vf]
                        logging.info(
                            "  after detect-adjust: %d valid  u_gain min/mean/max = %.2f / %.2f / %.2f",
                            len(_vf), min(_ug), float(np.mean(_ug)), max(_ug),
                        )
                    else:
                        logging.info("  after detect-adjust: 0 valid frontiers")

                if len(self.ft_manager.valid_frontiers) == 0:
                    logging.info("No frontiers, exploration finished.")
                    break

            # Update mapper every step
            self.mapper.integrate_from_buffer()
            self.mapper.interpolate_occupancy_grid()
            og = self.mapper.get_occupancy_grid()
            if self.args.voxel_grid:
                self.ft_manager.free_map = og["free"]
                self.ft_manager.occ_map = og["occupied"]
            else:
                self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])

            self.ft_manager.gain_adjustment()
            self.ft_manager.filter_frontiers()
            self.ft_manager.merge_frontiers()
            self.ft_manager.filter_frontiers()
            self.ft_manager.update_utility(current_pos=W_T_C[:3, 3])

            # Periodic frontier state summary (at detection cadence)
            if reach_next_update:
                _vf = self.ft_manager.valid_frontiers
                if _vf:
                    _ug_sorted = sorted([f.u_gain for f in _vf], reverse=True)
                    logging.info(
                        "  frontier state: %d valid  top-5 u_gain = %s",
                        len(_vf),
                        " ".join(f"{g:.2f}" for g in _ug_sorted[:5]),
                    )

            # Replan when the update interval fired and the robot has moved
            if reach_next_update and self.move_enough:
                logging.info("Replanning...")
                self.path_to_go = self.ft_manager.plan_path_to_goal(
                    W_T_C,
                    use_graph=not bool(self.args.voxel_grid),
                ) or []
                if self.path_to_go:
                    logging.info("Path found with %d steps.", len(self.path_to_go))
                    self.move_enough = False
                else:
                    logging.warning("No path found, dropping current goal frontier.")
                    self.path_to_go = []
                    self.move_enough = True

            # Persist state snapshot
            if self.json_path:
                self.ft_manager.write_to_file(file_path=self.json_path)

            # If the path just emptied but move_enough is still False (is_moving() never
            # fired because all waypoints were within v_tras_thre of the last recorded
            # pose — a degenerate near-zero-length path), reset move_enough so replanning
            # can fire next iteration.  Without this the replan condition
            # (reach_next_update AND move_enough) is permanently locked out.
            if not self.path_to_go and not self.move_enough:
                logging.warning(
                    "Path exhausted without is_moving() firing — resetting move_enough."
                )
                self.move_enough = True

            # Execute one movement step; check for stuck while actively following path
            if self.path_to_go:
                self.move(steps=1)

                self._recent_poses.append(W_T_C.copy())
                self._recent_path_remaining.append(len(self.path_to_go))
                if len(self._recent_poses) == self._recent_poses.maxlen:
                    poses = list(self._recent_poses)
                    path_rem_list = list(self._recent_path_remaining)
                    # Stuck fires only when BOTH:
                    #   1. all poses are numerically identical (exact same location + rotation)
                    #   2. path_remaining has not decreased (no waypoints consumed)
                    all_same_pose = all(
                        np.allclose(poses[i], poses[0], atol=1e-6, rtol=0)
                        for i in range(1, len(poses))
                    )
                    path_made_progress = path_rem_list[-1] < path_rem_list[0]
                    if all_same_pose and not path_made_progress:
                        goal_id = self.ft_manager.current_goal_ft_id
                        if goal_id is not None:
                            logging.warning(
                                "Stuck detected: pose unchanged for %d steps "
                                "and path_remaining did not decrease (%d→%d) — "
                                "dropping frontier %s.",
                                self._stuck_window,
                                path_rem_list[0], path_rem_list[-1], goal_id,
                            )
                            self.ft_manager.remove_frontiers([goal_id])
                            self._recent_poses.clear()
                            self._recent_path_remaining.clear()
                            self.path_to_go = []
                            self.move_enough = True

        # Final write
        if self.json_path:
            self.ft_manager.write_to_file(file_path=self.json_path)
        logging.info(
            "Baseline-loop exploration finished, total steps: %d", n_robot_poses
        )

    def run(self) -> None:
        """Run the headless exploration."""
        try:
            self.setup_system()
            logging.info("Starting headless exploration...")
            self.exploration_baseline()
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
        default=1000,  # shell script passes this explicitly; kept in sync
        help="Maximum number of exploration steps",
    )
    p.add_argument(
        "--max_time", type=int, default=1800, help="Maximum exploration time in seconds"
    )
    p.add_argument(
        "--unet_weight",
        type=Path,
        default=None,
        help="Path to DETR model checkpoint",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="detr",
        choices=["detr", "cond_detr", "mapex"],
        help=(
            "Model architecture: "
            "'detr' (FrontierDETR: ResNet50+enc+dec, DETR-Factory), "
            "'cond_detr' (FrontierConditionalDETR: ResNet50+enc+cond dec, DETR-Factory), "
            "'mapex' (MapEx ICRA-2025: LaMa ensemble on 2-D top-down occupancy map)"
        ),
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
    # --- DETR-specific args ---
    p.add_argument(
        "--detr_conf_thresh",
        type=float,
        default=0.3,
        help="Minimum slot confidence to accept as a frontier candidate.",
    )
    p.add_argument(
        "--detr_num_queries",
        type=int,
        default=10,
        help="Number of DETR slot queries (must match the trained checkpoint).",
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
        "--detr_gain_scale",
        type=float,
        default=100.0,
        help=(
            "Multiplicative scale applied to the raw GMM weight before "
            "it is stored as frontier gain. Default 100.0."
        ),
    )
    p.add_argument(
        "--detr_dedup_radius",
        type=float,
        default=0.5,
        help=(
            "Exclusion radius in metres for cross-frame deduplication. "
            "A newly detected frontier is dropped if any existing valid frontier "
            "lies within this distance. Default 0.5 m."
        ),
    )
    # --- MapEx args ---
    p.add_argument(
        "--mapex_dir",
        type=str,
        default="/cluster/project/cvg/students/shangwu/MapEx",
        help="Path to the MapEx repository root (used when --model_type mapex).",
    )
    p.add_argument(
        "--mapex_map_size",
        type=int,
        default=512,
        help="(mapex) Top-down occupancy map resolution in pixels (must be multiple of 16). "
             "Default 512.",
    )
    p.add_argument(
        "--mapex_map_margin",
        type=float,
        default=3.0,
        help="(mapex) Metres of unknown padding around the observed bounding box. "
             "Default 3.0 m.",
    )
    p.add_argument(
        "--mapex_min_frontier_size",
        type=int,
        default=10,
        help="(mapex) Minimum frontier pixel-cluster size to accept. Default 10.",
    )
    p.add_argument(
        "--mapex_gain_scale",
        type=float,
        default=360.0,
        help=(
            "(mapex) Multiplier applied to LaMa per-frontier variance to produce "
            "Frontier.gain. Calibrated so that LaMa variance [0, 0.083] maps to "
            "gain [2, 30], matching the baseline UNet frontier gain range and making "
            "gain_adjustment() volumetric decay (reduction_2 ~ 0-5 m3) meaningful. "
            "Default 360."
        ),
    )
    p.add_argument(
        "--free_vox_filter",
        action="store_true",
        default=True,
        help=(
            "Filter detected frontiers against the pre-loaded free-space voxel grid "
            "(requires --voxel_grid): immediately discard any frontier whose pos3d is "
            "not within freespace_filter_dist metres of a free voxel."
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
    os._exit(0)


if __name__ == "__main__":
    main()