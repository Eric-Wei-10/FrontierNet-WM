#!/usr/bin/env python3
"""
Headless Replay script for FrontierNet exploration.

This script replays exploration trajectories from JSON files in headless mode,
performs TSDF integration using WaveMapper, and computes the mapped volume.
"""
import sys
import json
import time
import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import torch
import open3d as o3d
import open3d.visualization.rendering as rendering

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.vis_utils import create_camera
from utils.frontier_utils import read_config_yaml
from mapping.wavemap import WaveMapper
from frontier.manager import FrontierManager


class HeadlessRenderer:
    """
    A headless renderer using Open3D's OffscreenRenderer for rendering depth
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
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.z_near = z_near
        self.z_far = z_far
        self._extrinsic = np.eye(4)

        self.renderer = rendering.OffscreenRenderer(width, height)

        # Logic to load GLB/GLTF or standard meshes with unlit shaders
        mesh_path_lower = mesh_path.lower()
        if mesh_path_lower.endswith('.glb') or mesh_path_lower.endswith('.gltf'):
            model = o3d.io.read_triangle_model(mesh_path)
            if model is not None and len(model.meshes) > 0:
                for i, mesh_info in enumerate(model.meshes):
                    mesh_geom = mesh_info.mesh
                    material_idx = mesh_info.material_idx
                    
                    new_mat = rendering.MaterialRecord()
                    new_mat.shader = "defaultUnlit"
                    if material_idx >= 0 and material_idx < len(model.materials):
                        mat = model.materials[material_idx]
                        if mat.albedo_img is not None:
                            new_mat.albedo_img = mat.albedo_img
                    new_mat.base_color = [1.0, 1.0, 1.0, 1.0]
                    self.renderer.scene.add_geometry(f"mesh_{i}", mesh_geom, new_mat)
            else:
                raise ValueError(f"Failed to load model from {mesh_path}")
        else:
            mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit" if mesh.has_vertex_colors() else "defaultLit"
            material.base_color = [0.8, 0.8, 0.8, 1.0]
            self.renderer.scene.add_geometry("mesh", mesh, material)
        
        self.renderer.scene.scene.enable_sun_light(False)
        self.renderer.scene.scene.enable_indirect_light(False)
        self.renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self._setup_camera()

    def _setup_camera(self):
        K = self.intrinsic.intrinsic_matrix
        self.renderer.setup_camera(K, self._extrinsic, self.width, self.height)

    def set_extrinsic(self, extrinsic: np.ndarray):
        """Set the camera extrinsic matrix (C_T_W)."""
        self._extrinsic = extrinsic.copy()
        self._setup_camera()

    def capture_depth(self) -> np.ndarray:
        depth_img = self.renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_img).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def cleanup(self):
        try:
            self.renderer.scene.clear_geometry()
        except Exception:
            pass


class ReplayApp:
    """
    Replay exploration trajectories and compute mapped volume in Headless mode.
    """
    # Camera defaults
    CAM_H, CAM_W, CAM_F = 480, 480, 300.0

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = read_config_yaml(args.config)
        
        self.CAM_H = int(self.config.get("cam_height", self.CAM_H))
        self.CAM_W = int(self.config.get("cam_width", self.CAM_W))
        self.CAM_F = float(self.config.get("focal_length", self.CAM_F))
        
        self.renderer: Optional[HeadlessRenderer] = None
        self.mapper: Optional[WaveMapper] = None
        self.voxel_size = float(self.config.get("voxel_size", 0.1))
        self.depth_range = float(self.config.get("depth_range", 3.5))
        
        self.results: List[Dict[str, Any]] = []
        self.output_path: Optional[str] = None

    def setup_renderer(self) -> None:
        """Initialize HeadlessRenderer and load mesh."""
        cam_intrinsic = create_camera(self.CAM_H, self.CAM_W, self.CAM_F)
        
        self.renderer = HeadlessRenderer(
            mesh_path=self.args.mesh,
            width=self.CAM_W,
            height=self.CAM_H,
            intrinsic=cam_intrinsic,
            z_near=0.02,
            z_far=50.0,
        )
        logging.info(f"Headless renderer setup with mesh: {self.args.mesh}")

    def setup_mapper(self) -> None:
        """Initialize WaveMapper with camera parameters."""
        intr = self.renderer.intrinsic
        
        params = {
            "min_cell_width": self.voxel_size / 2.0,
            "width": intr.width,
            "height": intr.height,
            "fx": intr.intrinsic_matrix[0, 0],
            "fy": intr.intrinsic_matrix[1, 1],
            "cx": intr.intrinsic_matrix[0, 2],
            "cy": intr.intrinsic_matrix[1, 2],
            "min_range": 0.05,
            "max_range": self.depth_range,
            "resolution": self.voxel_size,
        }
        
        self.mapper = WaveMapper(params=params)
        logging.info("WaveMapper initialized")

    def load_trajectory(self) -> List[Dict[str, Any]]:
        entries = FrontierManager.read_from_file(self.args.json_file)
        logging.info(f"Loaded {len(entries)} entries from {self.args.json_file}")
        return entries

    def capture_depth_at_pose(self, W_T_C: np.ndarray) -> np.ndarray:
        """Teleport camera to pose and capture depth in headless mode."""
        # Convert W_T_C (world to camera) to C_T_W (camera to world/extrinsic)
        C_T_W = np.linalg.inv(W_T_C)
        self.renderer.set_extrinsic(C_T_W)
        
        # Capture depth
        depth = self.renderer.capture_depth()
        depth[depth > self.depth_range] = 0.0
        return depth

    def compute_mapped_volume(self, only_free=True) -> float:
        self.mapper.interpolate_occupancy_grid()
        og = self.mapper.get_occupancy_grid()
        
        occ_pts = og.get("occupied", [])
        free_pts = og.get("free", [])
        
        num_occ_voxels = len(occ_pts)
        num_free_voxels = len(free_pts)
        num_voxels = num_free_voxels if only_free else (num_free_voxels + num_occ_voxels)
        return num_voxels * (self.voxel_size ** 3)

    def replay_trajectory(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entries:
            return []
        
        max_steps = min(len(entries), self.args.max_steps)
        updated_entries = []
        
        for i, entry in enumerate(entries):
            if i >= max_steps:
                entry_copy = entry.copy()
                entry_copy["mapped_vol"] = None
                updated_entries.append(entry_copy)
                continue
            
            robot_poses = entry.get("robot_poses", {})
            
            # Determine which poses are new compared to the previous entry
            if i == 0:
                new_poses = robot_poses
            else:
                prev_poses = entries[i - 1].get("robot_poses", {})
                new_poses = {rid: p for rid, p in robot_poses.items() if rid not in prev_poses}
            
            # Process new poses
            for rid, pose_list in new_poses.items():
                pose = np.array(pose_list, dtype=np.float64)
                depth = self.capture_depth_at_pose(pose)
                self.mapper.insert_depth_to_buffer(depth=depth, transform=pose)
            
            self.mapper.integrate_from_buffer()
            volume = self.compute_mapped_volume()
            
            logging.info(f"Step {i + 1}/{len(entries)}: mapped_vol = {volume:.4f} m^3")
            
            entry_copy = entry.copy()
            entry_copy["mapped_vol"] = volume
            updated_entries.append(entry_copy)
            
            if self.args.save_interval > 0 and (i + 1) % self.args.save_interval == 0:
                self.save_results(updated_entries, self.output_path)
        
        return updated_entries

    def save_results(self, entries: List[Dict[str, Any]], output_path: str) -> None:
        with open(output_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def cleanup(self) -> None:
        if self.renderer is not None:
            self.renderer.cleanup()
            del self.renderer.renderer
            del self.renderer
        import gc
        gc.collect()

    def run(self) -> None:
        try:
            self.setup_renderer()
            self.setup_mapper()
            
            entries = self.load_trajectory()
            if not entries:
                logging.error("No entries found")
                return
            
            self.output_path = self.args.output or self.args.json_file.replace(".json", "_with_volume.json")
            updated_entries = self.replay_trajectory(entries)
            self.save_results(updated_entries, self.output_path)
            
            logging.info("Replay complete.")
        finally:
            self.cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless Replay and Volume Computation")
    p.add_argument("--mesh", type=str, required=True, help="Path to the scene mesh file")
    p.add_argument("--json_file", "-j", type=str, required=True, help="Path to exploration JSON")
    p.add_argument("--config", type=str, default="config/hm3d_exploration.yaml")
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--save_interval", "-s", type=int, default=10)
    p.add_argument("--log_level", "-ll", type=int, default=20)
    return p


if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)
    args = build_arg_parser().parse_args()
    
    # Adjust log level based on args
    logging.getLogger().setLevel(args.log_level)
    
    app = ReplayApp(args)
    app.run()