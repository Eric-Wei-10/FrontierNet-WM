#!/usr/bin/env python3
"""
Headless Replay Visualization for FrontierNet exploration.

Same as replay_visualization.py but runs without a display, rendering
observer-view and robot-view frames to disk at each step using
Open3D's OffscreenRenderer.

Usage:
    python eval/replay_visualization_headless.py \
        --mesh examples/mv2HUxq3B53.glb \
        --json_file output/exploration_state.json \
        --output_dir output/frames/
"""
import sys
import json
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.vis_utils import create_camera
from utils.frontier_utils import read_config_yaml
from mapping.wavemap import WaveMapper


def _save_jsonl(entries: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def clip_mesh_by_z(mesh: o3d.geometry.TriangleMesh,
                   max_z: float) -> o3d.geometry.TriangleMesh:
    """Return a copy of *mesh* with all triangles that have any vertex above
    *max_z* removed.  UV coordinates, vertex colours, and normals are
    preserved for the surviving triangles/vertices.
    """
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    if len(tris) == 0 or len(verts) == 0:
        return mesh

    # keep triangles whose highest vertex is at or below max_z
    keep = verts[tris, 2].max(axis=1) <= max_z
    if keep.all():
        return mesh
    if not keep.any():
        return o3d.geometry.TriangleMesh()

    new_tris = tris[keep]

    # remap vertex indices to a compact range
    used = np.unique(new_tris)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    out = o3d.geometry.TriangleMesh()
    out.vertices  = o3d.utility.Vector3dVector(verts[used])
    out.triangles = o3d.utility.Vector3iVector(remap[new_tris])

    # triangle UVs: 3 consecutive entries per triangle → slice with keep mask
    if mesh.has_triangle_uvs():
        uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)  # (N, 3, 2)
        out.triangle_uvs = o3d.utility.Vector2dVector(uvs[keep].reshape(-1, 2))
    if mesh.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors)[used])
    if mesh.has_vertex_normals():
        out.vertex_normals = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_normals)[used])
    if mesh.has_triangle_material_ids():
        out.triangle_material_ids = o3d.utility.IntVector(
            np.asarray(mesh.triangle_material_ids)[keep])

    logging.debug("clip_mesh_by_z: kept %d / %d triangles (max_z=%.3f)",
                  keep.sum(), len(tris), max_z)
    return out


def read_exploration_entries(file_path: str) -> List[Dict[str, Any]]:
    entries = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        logging.info(f"Loaded {len(entries)} entries from {file_path}")
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
    return entries


class HeadlessSceneRenderer:
    """
    Wraps OffscreenRenderer with named-geometry tracking and cached camera state.
    Camera extrinsic must be C_T_W (camera-from-world / Open3D convention).
    """

    def __init__(self, width: int, height: int, intrinsic: o3d.camera.PinholeCameraIntrinsic):
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self._K = intrinsic.intrinsic_matrix
        self._extrinsic = np.eye(4)
        self._counter = 0

        self.renderer = rendering.OffscreenRenderer(width, height)
        self.renderer.scene.scene.enable_sun_light(False)
        self.renderer.scene.scene.enable_indirect_light(False)
        self.renderer.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self._apply_camera()

    def _apply_camera(self) -> None:
        self.renderer.setup_camera(self._K, self._extrinsic, self.width, self.height)

    def set_extrinsic(self, extrinsic: np.ndarray) -> None:
        self._extrinsic = extrinsic.copy()
        self._apply_camera()

    def get_extrinsic(self) -> np.ndarray:
        return self._extrinsic.copy()

    def _next_name(self, prefix: str) -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def add_mesh(self, geom: o3d.geometry.TriangleMesh, name: str = None,
                 lit: bool = True) -> str:
        if name is None:
            name = self._next_name("mesh")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit" if lit else "defaultUnlit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        self.renderer.scene.add_geometry(name, geom, mat)
        return name

    def add_lineset(self, geom: o3d.geometry.LineSet, name: str = None) -> str:
        if name is None:
            name = self._next_name("line")
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 2.0
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        self.renderer.scene.add_geometry(name, geom, mat)
        return name

    def remove(self, name: str) -> None:
        try:
            self.renderer.scene.remove_geometry(name)
        except Exception:
            pass

    def capture_rgb(self) -> np.ndarray:
        return np.asarray(self.renderer.render_to_image())

    def capture_depth(self) -> np.ndarray:
        depth = np.asarray(
            self.renderer.render_to_depth_image(z_in_view_space=True)
        ).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def load_scene_mesh(self, mesh_path: str,
                        clip_z: Optional[float] = None) -> List[str]:
        """Load GLB/GLTF or standard mesh. Returns list of added geometry names.

        If *clip_z* is given, triangles with any vertex above that world-frame Z
        are removed before adding to the scene (removes ceiling / upper floors).
        """
        names: List[str] = []
        lower = mesh_path.lower()
        if lower.endswith(".glb") or lower.endswith(".gltf"):
            model = o3d.io.read_triangle_model(mesh_path)
            if model is None or len(model.meshes) == 0:
                raise ValueError(f"Failed to load GLB model from {mesh_path}")
            for i, mesh_info in enumerate(model.meshes):
                geom = mesh_info.mesh
                if clip_z is not None:
                    geom = clip_mesh_by_z(geom, clip_z)
                    if len(np.asarray(geom.triangles)) == 0:
                        continue
                mat = rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                mid = mesh_info.material_idx
                if 0 <= mid < len(model.materials):
                    orig = model.materials[mid]
                    if orig.albedo_img is not None:
                        mat.albedo_img = orig.albedo_img
                mat.base_color = [1.0, 1.0, 1.0, 1.0]
                name = f"scene_mesh_{i}"
                self.renderer.scene.add_geometry(name, geom, mat)
                names.append(name)
        else:
            mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
            if clip_z is not None:
                mesh = clip_mesh_by_z(mesh, clip_z)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit" if mesh.has_vertex_colors() else "defaultLit"
            mat.base_color = [0.8, 0.8, 0.8, 1.0]
            self.renderer.scene.add_geometry("scene_mesh_0", mesh, mat)
            names.append("scene_mesh_0")
        return names

    def cleanup(self) -> None:
        # Do NOT call clear_geometry() before destroying the renderer — it releases
        # Filament GPU resources, then the OffscreenRenderer destructor tries to free
        # the same resources again, causing the "nonexistent resource" crash.
        # Setting the renderer to None lets Filament destroy everything in one shot.
        self.renderer = None


def _write_frame_pair(
    obs_rgb: np.ndarray,
    rob_rgb: np.ndarray,
    step: int,
    output_dir: Path,
    ext: str,
    save_kw: dict,
    sem: threading.BoundedSemaphore,
) -> None:
    """Write one observer + robot frame pair to disk (runs in a background thread)."""
    try:
        Image.fromarray(obs_rgb).save(output_dir / f"obs_{step:05d}.{ext}", **save_kw)
        Image.fromarray(rob_rgb).save(output_dir / f"robot_{step:05d}.{ext}", **save_kw)
    finally:
        sem.release()


class ReplayApp:
    # Observer view
    CAM1_H, CAM1_W, CAM1_F = 960, 1280, 700.0
    # Robot ego view
    CAM2_H, CAM2_W, CAM2_F = 480, 480, 300.0

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = read_config_yaml(args.config)

        self.renderer_1: Optional[HeadlessSceneRenderer] = None  # observer
        self.renderer_2: Optional[HeadlessSceneRenderer] = None  # robot ego

        self.entries: List[Dict[str, Any]] = []
        self.current_step: int = 0
        self.current_robot_poses: Dict[str, np.ndarray] = {}
        self.current_frontiers: List[Dict[str, Any]] = []
        self.current_goal_pose: Optional[np.ndarray] = None
        self.current_robot_id: Optional[int] = None
        self.last_W_T_C2: np.ndarray = np.eye(4)

        # Each overlay is a single fixed-name geometry swapped every step.
        # Per-object name lists caused O(N_frontiers) Filament entity allocations
        # per step; after ~65k total allocations the entity pool exhausts → segfault.
        self._traj_lineset_present: bool = False
        self._traj_lineset_counter: int = 0

        # Fixed observer extrinsic (C_T_W)
        self.obs_C_T_W: Optional[np.ndarray] = None

        # Overlay change-tracking: only rebuild an overlay when its inputs changed.
        self._prev_n_robot_poses: int = 0
        self._prev_robot_id: Optional[int] = None
        self._prev_frontier_ids: frozenset = frozenset()
        self._prev_goal_id: Optional[int] = None

        # Volume computation (optional — enabled by --compute_volume)
        self._volume_renderer: Optional[HeadlessSceneRenderer] = None
        self._mapper: Optional[WaveMapper] = None
        self._processed_pose_ids: set = set()
        self._entries_with_volume: List[Dict[str, Any]] = []
        self._last_mapped_vol: float = 0.0

        # Frame format: "jpg" is 3-5× faster than "png" for I/O; ffmpeg handles both.
        self._frame_ext: str = getattr(args, "frame_format", "jpg")

    # ------------------------------------------------------------------ setup

    def setup_renderers(self) -> None:
        intr_1 = create_camera(self.CAM1_H, self.CAM1_W, self.CAM1_F)
        intr_2 = create_camera(self.CAM2_H, self.CAM2_W, self.CAM2_F)

        self.renderer_1 = HeadlessSceneRenderer(self.CAM1_W, self.CAM1_H, intr_1)
        self.renderer_2 = HeadlessSceneRenderer(self.CAM2_W, self.CAM2_H, intr_2)

        # Derive ceiling clip height from initial robot camera position + offset
        clip_z: Optional[float] = None
        if self.args.clip_ceiling_offset is not None:
            rob_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], dtype=float)
            cam_world_z = (-rob_C_T_W[:3, :3].T @ rob_C_T_W[:3, 3])[2]
            clip_z = cam_world_z + self.args.clip_ceiling_offset
            logging.info(
                f"Ceiling clip Z = {clip_z:.3f} m "
                f"(cam_z={cam_world_z:.3f} + offset={self.args.clip_ceiling_offset})"
            )

        self.renderer_1.load_scene_mesh(self.args.mesh, clip_z=clip_z)
        self.renderer_2.load_scene_mesh(self.args.mesh, clip_z=clip_z)
        logging.info(f"Loaded mesh: {self.args.mesh}")

        # World axis in observer view
        world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        world_axis.compute_vertex_normals()
        self.renderer_1.add_mesh(world_axis, name="world_axis", lit=True)

        # Initial camera poses from config
        self.obs_C_T_W = np.asarray(self.config["observer_cam_extrinsic"], dtype=float)
        rob_C_T_W = np.asarray(self.config["initial_cam_extrinsic"], dtype=float)
        self.renderer_1.set_extrinsic(self.obs_C_T_W)
        self.renderer_2.set_extrinsic(rob_C_T_W)

    def setup_system(self) -> bool:
        self.entries = read_exploration_entries(self.args.json_file)
        if not self.entries:
            logging.error("No entries found.")
            return False
        logging.info(f"Replay system ready: {len(self.entries)} steps.")
        return True

    def setup_volume_computation(self) -> None:
        """Create a dedicated depth renderer + WaveMapper for volume estimation."""
        intr = create_camera(self.CAM2_H, self.CAM2_W, self.CAM2_F)
        self._volume_renderer = HeadlessSceneRenderer(self.CAM2_W, self.CAM2_H, intr)
        self._volume_renderer.load_scene_mesh(self.args.mesh)  # bare mesh, no overlays

        voxel_size  = float(self.config.get("voxel_size",   0.1))
        depth_range = float(self.config.get("depth_range",  3.5))
        params = {
            "min_cell_width": voxel_size / 2.0,
            "width":  intr.width,
            "height": intr.height,
            "fx": intr.intrinsic_matrix[0, 0],
            "fy": intr.intrinsic_matrix[1, 1],
            "cx": intr.intrinsic_matrix[0, 2],
            "cy": intr.intrinsic_matrix[1, 2],
            "min_range": 0.05,
            "max_range": depth_range,
            "resolution": voxel_size,
        }
        self._mapper = WaveMapper(params=params)
        logging.info("Volume computation enabled (voxel_size=%.3f m, depth_range=%.2f m)",
                     voxel_size, depth_range)

    def _integrate_new_poses(self, entry: Dict[str, Any]) -> bool:
        """Integrate depth for any robot poses not yet processed.

        Returns True if any new depth data was inserted so the caller knows
        whether to recompute the mapped volume.
        """
        depth_range = float(self.config.get("depth_range", 3.5))
        any_new = False
        for rid, pose_list in entry.get("robot_poses", {}).items():
            if rid in self._processed_pose_ids:
                continue
            self._processed_pose_ids.add(rid)
            any_new = True
            W_T_C = np.array(pose_list, dtype=np.float64)
            self._volume_renderer.set_extrinsic(np.linalg.inv(W_T_C))
            depth = self._volume_renderer.capture_depth()
            depth[depth > depth_range] = 0.0
            self._mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C)
        if any_new:
            self._mapper.integrate_from_buffer()
        return any_new

    def _compute_mapped_volume(self) -> float:
        voxel_size = float(self.config.get("voxel_size", 0.1))
        self._mapper.interpolate_occupancy_grid()
        og = self._mapper.get_occupancy_grid()
        return len(og.get("free", [])) * (voxel_size ** 3)

    # ---------------------------------------------------------------- helpers

    def _pose_from_pos_dir(self, position: np.ndarray, direction: np.ndarray) -> np.ndarray:
        z = direction / (np.linalg.norm(direction) + 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x) + 1e-8
        y = np.cross(z, x)
        y /= np.linalg.norm(y) + 1e-8
        pose = np.eye(4)
        pose[:3, 0] = x
        pose[:3, 1] = y
        pose[:3, 2] = z
        pose[:3, 3] = position
        return pose

    # ---------------------------------------------------- per-step overlays

    def _update_agent_marker(self) -> None:
        """Single sphere at the agent's current position — fixed name, O(1) GPU objects."""
        self.renderer_1.remove("agent_marker")
        pos = self.last_W_T_C2[:3, 3].copy()
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4, resolution=12)
        sphere.translate(pos)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [0.0, 0.8, 1.0, 1.0]
        self.renderer_1.renderer.scene.add_geometry("agent_marker", sphere, mat)

    def _update_frontier_markers(self) -> None:
        """All frontier spheres merged into one batched mesh per renderer — O(1) GPU objects."""
        self.renderer_1.remove("frontier_markers_obs")
        self.renderer_2.remove("frontier_markers_ego")

        if not self.current_frontiers:
            return

        current_goal_id = None
        if self.entries and self.current_step < len(self.entries):
            current_goal_id = self.entries[self.current_step].get("current_ft_goal_id")

        batch_obs = o3d.geometry.TriangleMesh()
        batch_ego = o3d.geometry.TriangleMesh()
        for ft_data in self.current_frontiers:
            pos3d = ft_data.get("3d_pos")
            if pos3d is None:
                continue
            pos3d = np.array(pos3d, dtype=np.float64)
            is_goal = ft_data.get("id") == current_goal_id
            radius = 0.9 if is_goal else 0.6
            color = [1.0, 0.85, 0.0] if is_goal else [1.0, 0.35, 0.1]

            for batch in (batch_obs, batch_ego):
                s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
                s.translate(pos3d)
                s.paint_uniform_color(color)
                batch += s

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        if len(np.asarray(batch_obs.vertices)) > 0:
            self.renderer_1.renderer.scene.add_geometry("frontier_markers_obs", batch_obs, mat)
            self.renderer_2.renderer.scene.add_geometry("frontier_markers_ego", batch_ego, mat)

    def _update_robot_frustum(self) -> None:
        """Camera frustum as a single LineSet — replaces per-cylinder meshes."""
        self.renderer_1.remove("robot_frustum")

        intr = self.renderer_2.intrinsic
        W, H = intr.width, intr.height
        fx = intr.intrinsic_matrix[0, 0]
        fy = intr.intrinsic_matrix[1, 1]
        cx = intr.intrinsic_matrix[0, 2]
        cy = intr.intrinsic_matrix[1, 2]
        d = 0.8  # frustum depth scale

        corners_cam = np.array([
            [(0 - cx) / fx * d, (0 - cy) / fy * d, d],
            [(W - cx) / fx * d, (0 - cy) / fy * d, d],
            [(W - cx) / fx * d, (H - cy) / fy * d, d],
            [(0 - cx) / fx * d, (H - cy) / fy * d, d],
        ])
        origin = self.last_W_T_C2[:3, 3]
        corners_world = (self.last_W_T_C2[:3, :3] @ corners_cam.T).T + origin

        pts = np.vstack([origin, corners_world])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[0.0, 0.4, 1.0]] * len(lines))
        self.renderer_1.add_lineset(ls, name="robot_frustum")

    def _update_frontier_overlays(self) -> None:
        """All frontier view-direction arrows as one LineSet — O(1) GPU objects."""
        self.renderer_1.remove("frontier_overlays")

        if not self.current_frontiers:
            return

        current_goal_id = None
        if self.entries and self.current_step < len(self.entries):
            current_goal_id = self.entries[self.current_step].get("current_ft_goal_id")

        pts: List[np.ndarray] = []
        lines: List[List[int]] = []
        colors: List[List[float]] = []

        for ft_data in self.current_frontiers:
            pos3d = ft_data.get("3d_pos")
            if pos3d is None:
                continue
            pos3d = np.array(pos3d, dtype=np.float64)
            vd = ft_data.get("vd")
            is_goal = ft_data.get("id") == current_goal_id
            scale = 1.2 if is_goal else 0.7
            color = [1.0, 0.85, 0.0] if is_goal else [0.8, 0.5, 0.0]

            if vd is not None:
                v = np.array(vd, dtype=np.float64)
                n = np.linalg.norm(v)
                tip = pos3d + (v / n * scale) if n > 1e-8 else pos3d + np.array([0, 0, scale])
            else:
                tip = pos3d + np.array([0, 0, scale])

            idx = len(pts)
            pts.extend([pos3d, tip])
            lines.append([idx, idx + 1])
            colors.append(color)

        if pts:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.array(pts))
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(colors)
            self.renderer_1.add_lineset(ls, name="frontier_overlays")

    def _update_trajectory_cylinders(self) -> None:
        # Use a single LineSet instead of per-segment cylinder meshes.
        # Cylinders are TriangleMesh objects that accumulate in Filament's GPU
        # resource pool (they were never removed), exhausting its handle table
        # after many steps and causing a segmentation fault.
        if len(self.current_robot_poses) < 2:
            return

        if self._traj_lineset_present:
            old_name = f"traj_lineset_{self._traj_lineset_counter - 1}"
            self.renderer_1.remove(old_name)
            self._traj_lineset_present = False

        sorted_ids = sorted(self.current_robot_poses.keys(), key=lambda x: int(x))
        pts = [self.current_robot_poses[rid][:3, 3] for rid in sorted_ids]

        pts_arr = np.array(pts, dtype=np.float64)
        if np.max(pts_arr.max(axis=0) - pts_arr.min(axis=0)) < 1e-6:
            return  # all poses identical — Filament rejects zero-volume AABB

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
        ls.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(pts) - 1)]
        )
        ls.colors = o3d.utility.Vector3dVector(
            [[0.0, 0.4, 1.0]] * (len(pts) - 1)
        )
        new_name = f"traj_lineset_{self._traj_lineset_counter}"
        self._traj_lineset_counter += 1
        self.renderer_1.add_lineset(ls, name=new_name)
        self._traj_lineset_present = True

    def _update_graph_overlay(self) -> None:
        """All graph edges in one LineSet — O(1) GPU objects regardless of edge count."""
        self.renderer_1.remove("graph_edges")

        if not self.entries or self.current_step >= len(self.entries):
            return
        entry = self.entries[self.current_step]
        edges = entry.get("graph", {}).get("edges", [])
        nodes = entry.get("graph", {}).get("nodes", [])
        if not edges or not nodes:
            return

        node_pos: Dict[Any, np.ndarray] = {}
        for node in nodes:
            if len(node) < 2:
                continue
            nid, attrs = node[0], node[1]
            if attrs.get("type") == "R" and str(nid) in self.current_robot_poses:
                node_pos[nid] = self.current_robot_poses[str(nid)][:3, 3]
            elif attrs.get("type") == "F":
                ft_by_id = {ft.get("id"): ft for ft in self.current_frontiers}
                ft = ft_by_id.get(nid)
                if ft and ft.get("3d_pos"):
                    node_pos[nid] = np.array(ft["3d_pos"], dtype=np.float64)

        pts: List[np.ndarray] = []
        lines: List[List[int]] = []
        for edge in edges:
            if len(edge) < 2:
                continue
            src, dst = edge[0], edge[1]
            if src in node_pos and dst in node_pos:
                idx = len(pts)
                pts.extend([node_pos[src], node_pos[dst]])
                lines.append([idx, idx + 1])

        if pts:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.array(pts))
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]] * len(lines))
            self.renderer_1.add_lineset(ls, name="graph_edges")

    # ---------------------------------------------------------------- replay

    def goto_step(self, step: int) -> None:
        step = max(0, min(step, len(self.entries) - 1))
        self.current_step = step
        entry = self.entries[step]

        self.current_robot_poses = {
            k: np.array(v, dtype=np.float64)
            for k, v in entry.get("robot_poses", {}).items()
        }
        self.current_frontiers = entry.get("valid_frontiers", [])
        self.current_robot_id = entry.get("current_robot_id")
        gp = entry.get("current_goal_pose")
        self.current_goal_pose = np.array(gp, dtype=np.float64) if gp is not None else None

        # Pick the latest known robot pose for the ego view
        W_T_C2 = None
        if self.current_robot_id is not None:
            key = (
                str(self.current_robot_id - 1)
                if str(self.current_robot_id) not in self.current_robot_poses
                else str(self.current_robot_id)
            )
            W_T_C2 = self.current_robot_poses.get(key)
        if W_T_C2 is None and self.current_robot_poses:
            last_key = sorted(self.current_robot_poses, key=lambda x: int(x))[-1]
            W_T_C2 = self.current_robot_poses[last_key]

        if W_T_C2 is not None:
            self.renderer_2.set_extrinsic(np.linalg.inv(W_T_C2))
            self.last_W_T_C2 = W_T_C2
        else:
            logging.warning(f"No valid robot pose for step {step + 1}")

        # --- Conditional overlay updates -----------------------------------
        # Each overlay is only rebuilt when its underlying data actually changed.
        n_poses = len(self.current_robot_poses)
        current_frontier_ids = frozenset(
            ft.get("id") for ft in self.current_frontiers if ft.get("id") is not None
        )
        current_goal_id = (
            self.entries[step].get("current_ft_goal_id") if self.entries else None
        )

        pose_grew      = n_poses > self._prev_n_robot_poses
        robot_moved    = self.current_robot_id != self._prev_robot_id or pose_grew
        frontiers_changed = (
            current_frontier_ids != self._prev_frontier_ids
            or current_goal_id   != self._prev_goal_id
        )

        if pose_grew:
            self._update_trajectory_cylinders()
        if robot_moved:
            self._update_robot_frustum()
            self._update_agent_marker()
        if frontiers_changed:
            self._update_frontier_overlays()
            self._update_frontier_markers()
        if self.args.vis_graph and (pose_grew or frontiers_changed):
            self._update_graph_overlay()

        self._prev_n_robot_poses  = n_poses
        self._prev_robot_id       = self.current_robot_id
        self._prev_frontier_ids   = current_frontier_ids
        self._prev_goal_id        = current_goal_id
        # -------------------------------------------------------------------

        # Keep observer camera fixed
        self.renderer_1.set_extrinsic(self.obs_C_T_W)

        logging.info(
            f"Step {step + 1}/{len(self.entries)} — "
            f"robots: {n_poses}, frontiers: {len(self.current_frontiers)}"
        )

    def _save_frames(self, step: int, output_dir: Path) -> None:
        ext = self._frame_ext
        save_kw = {"quality": 92} if ext == "jpg" else {}
        obs_rgb = self.renderer_1.capture_rgb()
        rob_rgb = self.renderer_2.capture_rgb()
        Image.fromarray(obs_rgb).save(output_dir / f"obs_{step:05d}.{ext}", **save_kw)
        Image.fromarray(rob_rgb).save(output_dir / f"robot_{step:05d}.{ext}", **save_kw)

    def _make_video(self, output_dir: Path, num_steps: int) -> None:
        import subprocess
        ext = self._frame_ext
        obs_frames = sorted(output_dir.glob(f"obs_*.{ext}"))
        if not obs_frames:
            logging.warning("No observer frames found; skipping video creation.")
            return

        exp_name = output_dir.parent.name
        video_path = output_dir / f"replay_{exp_name}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "5",
            "-i", str(output_dir / f"obs_%05d.{ext}"),
            "-framerate", "5",
            "-i", str(output_dir / f"robot_%05d.{ext}"),
            "-filter_complex", "[1:v]scale=-2:960[rob];[0:v][rob]hstack=inputs=2",
            "-c:v", "mpeg4",
            "-q:v", "2",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        logging.info("Re-encoding side-by-side video with ffmpeg (mpeg4)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("ffmpeg failed:\n%s", result.stderr)
        else:
            logging.info("Video saved to %s", video_path)

    def run(self) -> None:
        self.setup_renderers()
        if not self.setup_system():
            return

        if self.args.compute_volume:
            self.setup_volume_computation()

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving frames to: {output_dir}")

        max_steps = min(len(self.entries), self.args.max_steps)
        ext = self._frame_ext
        save_kw = {"quality": 92} if ext == "jpg" else {}

        # BoundedSemaphore caps the number of frames queued for I/O at once,
        # preventing unbounded memory accumulation when rendering is faster than
        # disk writes (each frame pair is ~5-10 MB).
        _io_sem = threading.BoundedSemaphore(6)

        with ThreadPoolExecutor(max_workers=1) as io_pool:
            for step in range(max_steps):
                self.goto_step(step)

                # --- Capture on the main (GPU) thread ---
                obs_rgb = self.renderer_1.capture_rgb()
                rob_rgb = self.renderer_2.capture_rgb()

                # --- Submit disk write to background thread ---
                # Acquires the semaphore slot; the background thread releases it on
                # completion. If 6 writes are already pending the main thread blocks
                # here until one finishes, keeping memory bounded.
                _io_sem.acquire()
                io_pool.submit(
                    _write_frame_pair,
                    obs_rgb, rob_rgb, step, output_dir, ext, save_kw, _io_sem,
                )

                if self.args.compute_volume:
                    had_new = self._integrate_new_poses(self.entries[step])

                    # _compute_mapped_volume() runs wavemap's interpolate_occupancy_grid(),
                    # which is expensive. Depth is integrated every step (cheap), but the
                    # volume figure is only recomputed at save-interval checkpoints and at
                    # the final step. Intermediate steps carry the last known value.
                    at_checkpoint = (
                        self.args.volume_save_interval > 0
                        and (step + 1) % self.args.volume_save_interval == 0
                    )
                    at_end = step == max_steps - 1
                    if had_new and (at_checkpoint or at_end):
                        self._last_mapped_vol = self._compute_mapped_volume()
                        logging.info("Step %d/%d: mapped_vol = %.4f m^3",
                                     step + 1, max_steps, self._last_mapped_vol)

                    entry_copy = dict(self.entries[step])
                    entry_copy["mapped_vol"] = self._last_mapped_vol
                    self._entries_with_volume.append(entry_copy)

                    if at_checkpoint and self.args.volume_output:
                        _save_jsonl(self._entries_with_volume, self.args.volume_output)

        # ThreadPoolExecutor.__exit__ waits for all pending I/O to finish.
        logging.info(f"Saved {max_steps} frame pairs to {output_dir}")

        if self.args.compute_volume and self.args.volume_output and self._entries_with_volume:
            _save_jsonl(self._entries_with_volume, self.args.volume_output)
            logging.info("Volume JSON saved to %s", self.args.volume_output)

        if self.args.make_video:
            self._make_video(output_dir, max_steps)

        # Both OffscreenRenderer instances share a singleton Filament
        # FilamentResourceManager.  Destroying renderer_1 first frees its vertex
        # buffers; renderer_2's destructor then tries to free the same handles
        # → "nonexistent resource" → std::terminate → SIGABRT.
        # All output has already been written, so skip Python/C++ destructors
        # entirely and let the OS reclaim GPU memory on process exit.
        import os as _os
        _os._exit(0)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Headless replay visualization for FrontierNet exploration trajectories"
    )
    p.add_argument("--mesh", type=str, required=True, help="Path to the scene mesh file")
    p.add_argument("--json_file", "-j", type=str, required=True,
                   help="Path to the exploration state JSON file")
    p.add_argument("--config", type=str, default="config/hm3d_exploration.yaml",
                   help="Configuration YAML file")
    p.add_argument("--output_dir", "-o", type=str, default=None,
                   help="Directory to save rendered frames (default: <json_file>_frames/)")
    p.add_argument("--max_steps", type=int, default=10000,
                   help="Maximum number of steps to replay")
    p.add_argument("--play_speed", type=float, default=0.5,
                   help="Seconds per step (used only for video FPS)")
    p.add_argument("--vis_graph", action="store_true", default=False,
                   help="Overlay topological graph edges in observer view")
    p.add_argument("--make_video", action="store_true", default=False,
                   help="Stitch frames into replay.mp4 after rendering")
    p.add_argument("--log_level", "-ll", type=int, default=20,
                   help="Logging level (10=debug, 20=info, 30=warning)")
    p.add_argument("--clip_ceiling_offset", type=float, default=1.0,
                   help="Remove mesh triangles above (initial_cam_z + this offset). "
                        "Set to None/very large to disable. Default 1.0 m.")
    p.add_argument("--frame_format", choices=["jpg", "png"], default="jpg",
                   help="Image format for saved frames. jpg is 3-5× faster than png "
                        "and ffmpeg re-encodes either. Default: jpg.")
    # Volume computation (replaces a separate replay_headless.py invocation)
    p.add_argument("--compute_volume", action="store_true", default=False,
                   help="Also compute mapped volume via TSDF integration at each step.")
    p.add_argument("--volume_output", type=str, default=None,
                   help="Path to write exploration JSON enriched with mapped_vol fields.")
    p.add_argument("--volume_save_interval", type=int, default=10,
                   help="Write volume JSON every N steps (0 = only at end). Default 10.")
    return p


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    args = build_arg_parser().parse_args()
    logging.getLogger().setLevel(args.log_level)

    if args.output_dir is None:
        args.output_dir = args.json_file.replace(".json", "_frames")

    app = ReplayApp(args)
    app.run()
