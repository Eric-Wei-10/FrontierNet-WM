import json
from typing import Optional, Iterable, List, Dict
import logging
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from frontier.base import Base
from frontier.frontier import Frontier
from frontier.graph import FrontierGraph

from utils.frontier_utils import ft_pos_direct_distance
from utils.geometry import compute_alignment_transforms, pose_difference
from utils.vis_utils import create_cylinder_between_points
from utils.mapping_utils import (
    render_voxel_depth,
    select_visible_points,
    compute_visible_voxels,
)
from planner.occ_rrt_point3d import OccupancyGrid3DPathPlanner as PathPlanner


class FrontierManager(Base):

    def __init__(
        self, params: Optional[dict] = None, log_level: int = logging.INFO
    ) -> None:
        """
        params keys (optional):
        - filter_bbox: list[6] or None
        - filter_max_vd_z: float or None
        - filter_min_gain: float
        - visited_trans_threshold: float
        - visited_angle_threshold: float
        - visited_gain_reduction_factor: float
        - render_K: (3,3) intrinsics
        - render_H: int
        - render_W: int
        - render_depth_range: float
        - voxel_size: float
        - render_decrease_factor: float
        - force_all_frontier_to_xy: bool
        - utility_gain_factor: float
        - max_planning_time: float
        - planning_algo: str
        - (some planner-related keys are forwarded to PathPlanner)
        """
        super().__init__(params, log_level=log_level)
        self.logger.info("FrontierManager initialized")

        p = params or {}

        # frontiers and robot poses
        self.frontiers: Dict[int, Frontier] = {}
        self.robot_poses: Dict[int, np.ndarray] = {}

        # Graveyard: positions of removed frontiers — blocks re-detection nearby.
        self._graveyard: List[np.ndarray] = []
        self.graveyard_radius: float = float(p.get("graveyard_radius", 1.0))

        # Graph/Maps
        self.graph = FrontierGraph(p, log_level=log_level)
        self.occ_map: Optional[np.ndarray] = None
        self.free_map: Optional[np.ndarray] = None

        # Planner
        self.planner = PathPlanner(p)
        self.current_goal_pose: Optional[np.ndarray] = None
        self.current_goal_ft_id: Optional[int] = None

        # IDs
        self._current_frontier_id: int = self.graph.type_range["F"][0]
        self._current_robot_id: int = self.graph.type_range["R"][0]

        # Parameters
        self.filter_bbox = p.get("filter_bbox", None)
        self.filter_max_vd_z = p.get("filter_max_vd_z", None)
        self.filter_min_gain = float(p.get("filter_min_gain", 0.1))
        self.max_planning_time = float(p.get("max_planning_time", 1.5))
        self.planning_algo = p.get("planning_algo", "rrtstar")

        self.v_tras_thre = float(p.get("visited_trans_threshold", 0.4))
        self.v_angl_thre = float(p.get("visited_angle_threshold", 0.4))
        self.v_gain_reduction_factor = float(
            p.get("visited_gain_reduction_factor", 1000)
        )
        # Multiplicative decay per close robot pose, used by gain_adjustment_detr().
        # 0.5 means gain is halved for each previously visited nearby pose.
        self.detr_v_gain_reduction_factor = float(
            p.get("detr_visited_gain_reduction_factor", 0.5)
        )

        # Rendering
        if p.get("render_K") is not None:
            self.render_K = np.array(p["render_K"], dtype=np.float32).reshape(3, 3)
        else:
            self.render_K = np.array(
                [[60, 0, 64], [0, 60, 64], [0, 0, 1]], dtype=np.float32
            )

        self.render_H = int(p.get("render_H", 128))
        self.render_W = int(p.get("render_W", 128))
        self.render_d_range = float(p.get("render_depth_range", 3.5))
        self.voxel_size = float(p.get("voxel_size", 0.1))
        self.render_decrease_factor = float(p.get("render_decrease_factor", 1.0))

        self.force_all_frontier_to_xy = bool(p.get("force_all_frontier_to_xy", False))
        self.utility_g_factor = float(p.get("utility_gain_factor", 1.0))
        # Exponent applied to distance in the utility denominator.  With the
        # default value of 1.0 the formula is gain/distance (information per
        # unit travel).  Values > 1 penalise distant frontiers more steeply,
        # encouraging the robot to clear its local neighbourhood before chasing
        # high-gain targets across the room.  1.5 means a 3x-farther frontier
        # needs 5x more gain; 2.0 means it needs 9x more gain.
        self.utility_dist_exponent = float(p.get("utility_dist_exponent", 1.0))
        # Distance floor in the utility denominator. Prevents frontiers that
        # happen to land very close to the robot from getting astronomically
        # high utility and permanently out-competing distant high-gain goals.
        self.min_utility_dist = float(p.get("min_utility_dist", 0.5))

        # Distance threshold (metres) for the DETR visited-area check in
        # gain_adjustment_detr().  A robot pose within this distance of a frontier's
        # pos3d counts as "visited", decaying u_gain by detr_v_gain_reduction_factor.
        # Should be large enough to catch the robot passing near a frontier's target
        # area (including its snapped free-voxel goal), but not so large that
        # unrelated nearby frontiers get incorrectly penalised.  Default 2.0 m.
        self.detr_visited_dist_threshold: float = float(
            p.get("detr_visited_dist_threshold", 2.0)
        )

        # Hard cap on the number of active frontiers kept after filter_frontiers().
        # When the count exceeds this value, the lowest-utility frontiers are retired
        # first.  0 means unlimited.
        self.max_active_frontiers: int = int(p.get("max_active_frontiers", 0))

        # Filter: remove frontiers not in the free-voxel map (outside walls /
        # unreachable). Safe to enable only when the free map is pre-loaded and
        # complete (voxel_grid mode). In incremental wavemap mode, unmapped rooms
        # are also isfree=False, so enabling this would remove valid frontiers there.
        self.filter_not_in_freespace: bool = bool(p.get("filter_not_in_freespace", False))
        # Maximum distance (m) from the nearest free voxel for a frontier to pass
        # the freespace filter.  0.0 means use planner.isfree() (strict, within
        # max_dist2free=0.15 m).  For DETR mode, use ~1.5 m: DETR frontiers land
        # at the depth of occluded surfaces which may be slightly beyond the free-
        # space boundary, but wall/exterior frontiers are typically 2+ m from any
        # navigable free voxel.
        self.freespace_filter_dist: float = float(p.get("freespace_filter_dist", 0.0))

        # Maximum number of "close visit" counts (n_close) before a DETR frontier is
        # retired to the graveyard.  n_close is incremented by gain_adjustment_detr()
        # each time a stored robot pose is within detr_visited_dist_threshold of the
        # frontier's pos3d.  A threshold of ~5 means the robot has physically passed
        # within detr_visited_dist_threshold metres of this location 5+ times — it has
        # been explored.  0 disables this retirement.
        self.detr_max_n_close: int = int(p.get("detr_max_n_close", 0))

        # Immediate retirement radius: any non-goal valid frontier whose pos3d is
        # within this distance of the robot's current position is retired on the spot,
        # skipping the decay pipeline.  0.0 disables.  A value of ~1.0 m catches
        # frontiers the robot passes next to en route to another goal — these are
        # most likely in already-explored or immediately observable space and should
        # not persist in the active set.
        self.proximity_retirement_radius: float = float(
            p.get("proximity_retirement_radius", 0.0)
        )

        # Recent-trajectory penalty in update_utility(): frontiers whose pos3d falls
        # within trajectory_penalty_radius of any of the last trajectory_penalty_window
        # robot poses have their utility scaled by trajectory_penalty_factor.
        # This discourages the robot from immediately turning back to areas it just
        # traversed, reducing zigzag backtracking without disrupting long-range gain
        # comparisons.  0.0 radius disables the penalty entirely.
        self.trajectory_penalty_radius: float = float(
            p.get("trajectory_penalty_radius", 0.0)
        )
        self.trajectory_penalty_factor: float = float(
            p.get("trajectory_penalty_factor", 0.3)
        )
        self.trajectory_penalty_window: int = int(
            p.get("trajectory_penalty_window", 20)
        )


        # Validation
        if self.filter_bbox is not None:
            if not (
                isinstance(self.filter_bbox, (list, tuple))
                and len(self.filter_bbox) == 6
            ):
                raise ValueError(
                    "filter_bbox must be a list/tuple of 6 numbers [xmin,xmax,ymin,ymax,zmin,zmax]."
                )
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        if self.render_H <= 0 or self.render_W <= 0:
            raise ValueError("render_H and render_W must be positive.")
        if self.render_d_range <= 0:
            raise ValueError("render_depth_range must be positive.")

    def _alloc_frontier_id(self) -> int:
        """
        Return a fresh frontier ID. Skips over any IDs that might
        already be in use (e.g., after deletions).
        """
        nid = self._current_frontier_id
        # Advance counter for next time
        self._current_frontier_id += 1
        # If this nid is somehow in use, keep advancing
        while nid in self.frontiers:
            nid = self._current_frontier_id
            self._current_frontier_id += 1
        return nid

    def _alloc_robot_id(self) -> int:
        """
        Return a fresh robot ID. Skips over any IDs that might
        already be in use (e.g., after deletions).
        """
        rid = self._current_robot_id
        # Advance counter for next time
        self._current_robot_id += 1
        # If this rid is somehow in use, keep advancing
        while rid in self.robot_poses:
            rid = self._current_robot_id
            self._current_robot_id += 1
        return rid

    @property
    def all_frontiers(self) -> List[Frontier]:
        return list(self.frontiers.values())

    @property
    def all_frontiers_ids(self) -> List[int]:
        return list(self.frontiers.keys())

    @property
    def valid_frontiers(self) -> List[Frontier]:
        return [ft for ft in self.frontiers.values() if ft.is_valid]

    @property
    def utility(self) -> List[float]:
        return [ft.utility for ft in self.valid_frontiers]

    @property
    def current_robot_id(self) -> Optional[int]:
        """
        Latest existing robot ID (None if no robot poses yet).
        Previously this returned the *next* ID; this is safer.
        """
        if self.robot_poses:
            return max(self.robot_poses.keys())
        return None

    @property
    def goal_pose(self) -> np.ndarray | None:
        return self.current_goal_pose

    def add_robot_poses(self, robot_poses: list[np.ndarray]):
        """
        Add a list of robot poses to the manager.
        Args:
            robot_poses: List of robot poses, each pose is a 4x4 matrix.

        Returns:
            List of added robot IDs.
        """
        if len(robot_poses) == 0:
            self.logger.debug("No robot poses to add.")
            return []

        added_robot_ids = []
        for pose in robot_poses:
            assert pose.shape == (4, 4), "Each robot pose must be a 4x4 matrix."
            robot_id = self._alloc_robot_id()  # Assign a unique ID
            self.robot_poses[robot_id] = pose
            # add the robot node to the graph
            self.graph.add_node_R(robot_id)
            added_robot_ids.append(robot_id)
            # link the node with the latest robot pose (as the robot ID increments, the latest pose is always the last one added)
            if robot_id - 1 in self.robot_poses:  # avoid the first robot pose
                prev_pose = self.robot_poses[robot_id - 1]
                distance = np.linalg.norm(
                    pose[:3, 3] - prev_pose[:3, 3]
                )  # Euclidean distance in 3D space
                self.graph.add_edge_RR(robot_id, robot_id - 1, weight=distance)

        return added_robot_ids  # Return the list of added robot IDs

    def get_frontier(
        self, frontier_id: int, *, required: bool = False
    ) -> Optional[Frontier]:
        """
        Fetch a frontier by ID.

        Args:
            frontier_id: Frontier ID to fetch.
            required: If True, raise KeyError when missing; otherwise return None.

        Returns:
            Frontier or None.
        """
        ft = self.frontiers.get(frontier_id)
        if ft is None:
            msg = f"Frontier ID {frontier_id} not found."
            if required:
                raise KeyError(msg)
            self.logger.debug(msg)
        return ft

    def get_robot_pose(
        self, robot_id: int, *, required: bool = False
    ) -> Optional[np.ndarray]:
        """
        Fetch a robot pose (4x4) by ID.

        Args:
            robot_id: Robot ID to fetch.
            required: If True, raise KeyError when missing; otherwise return None.

        Returns:
            np.ndarray shape (4,4) or None.
        """
        pose = self.robot_poses.get(robot_id)
        if pose is None:
            msg = f"Robot pose ID {robot_id} not found."
            if required:
                raise KeyError(msg)
            self.logger.debug(msg)
        return pose

    @staticmethod
    def get_frontier_pose(frontier) -> np.ndarray:
        """
        Build a 4x4 pose for a frontier by aligning camera +Z to the frontier's
        view_direction and placing it at pos3d. Uses CV camera convention:
        +Z forward, +Y down (approx), +X right.

        Returns:
            (4,4) float ndarray
        """
        pos = np.asarray(frontier.pos3d, dtype=float).reshape(3)
        vd = np.asarray(frontier.view_direction, dtype=float).reshape(3)

        T = compute_alignment_transforms(
            origins=[pos],
            align_vec=vd,
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],  # CV convention: Y down approx
            appr_axis=[0, 1, 0],
        )[0]

        # Ensure exact translation (in case the util modifies it)
        T = np.asarray(T, dtype=float)
        T[:3, 3] = pos
        return T

    def add_frontiers(
        self, frontiers: List[Frontier], parent_ids: Optional[Iterable[int]] = None
    ) -> List[int]:
        """
        Add a list of frontiers. Optionally connect each to given robot parent IDs.

        Args:
            frontiers: Frontier objects to add.
            parent_ids: Robot pose IDs to connect (edges F-R). Duplicates ignored.
        """
        if len(frontiers) == 0:
            self.logger.debug("No frontiers to add.")
            return
        else:
            self.logger.debug(f"Adding {len(frontiers)} frontiers.")

        parent_ids = list(parent_ids) if parent_ids is not None else []

        for frontier in frontiers:
            # if force_all_frontier_to_xy is True, set the 3d vector to be in the XY plane
            if self.force_all_frontier_to_xy:
                vd = np.array(
                    [frontier.view_direction[0], frontier.view_direction[1], 0.0],
                    dtype=float,
                )
                n = np.linalg.norm(vd)
                frontier.view_direction = vd / n if n > 1e-8 else vd
            frontier.id = self._alloc_frontier_id()  # Assign a unique ID
            self.frontiers[frontier.id] = frontier  # Use ID as key for fast access
            # add the frontier node to the graph
            self.graph.add_node_F(frontier.id)
            # set the frontier's parent IDs
            frontier.parent_ids = []
            # assign 6D pose
            frontier.pose6d = self.get_frontier_pose(frontier)

            for parent_id in parent_ids:
                assert (
                    parent_id in self.robot_poses
                ), f"Parent robot ID {parent_id} does not exist in robot poses."
                # calculate the distance between frontier position and the robot position
                distance = np.linalg.norm(
                    frontier.pos3d - self.robot_poses[parent_id][:3, 3]
                )
                self.graph.add_edge_FR(frontier.id, parent_id, weight=distance)
                frontier.parent_ids.append(parent_id)  # Add parent ID to the frontier

    def remove_frontiers(self, frontier_ids, _record_graveyard: bool = True):
        """
        Remove frontiers by their IDs.
        Args:
            frontier_ids:      List of frontier IDs to remove.
            _record_graveyard: If True (default), add each removed frontier's pos3d to
                               the graveyard so it cannot be re-detected nearby.
                               Pass False from merge_frontiers: the merged frontier sits
                               at the weighted average of its members, so adding members
                               to the graveyard would immediately kill the merged result.
        """
        if not isinstance(frontier_ids, list):
            raise ValueError("frontier_ids must be a list.")

        # Deduplicate while preserving order
        seen = set()
        ids = [fid for fid in frontier_ids if not (fid in seen or seen.add(fid))]

        if not ids:
            self.logger.debug("No frontier IDs provided to remove.")
            return

        removed_any = False
        for fid in ids:
            if fid in self.frontiers:
                if _record_graveyard:
                    ft_pos = self.frontiers[fid].pos3d
                    if ft_pos is not None:
                        self._graveyard.append(np.asarray(ft_pos, dtype=float))
                del self.frontiers[fid]
                try:
                    self.graph.remove_node_F(fid)
                except Exception as e:
                    self.logger.warning(f"Graph removal failed for frontier {fid}: {e}")
                removed_any = True

        # Clear goal if it was removed
        if (
            removed_any
            and self.current_goal_ft_id is not None
            and self.current_goal_ft_id not in self.frontiers
        ):
            self.current_goal_ft_id = None
            self.current_goal_pose = None

        if removed_any:
            self.logger.debug(f"Removed frontiers: {ids}")
        else:
            self.logger.debug("No matching frontiers to remove.")

    def remove_invalid_frontiers(self) -> None:
        """
        Remove all frontiers currently marked invalid.
        Frontiers removed this way were flagged by geometric / gain filters —
        they were never truly visited, so their positions must NOT be added to
        the graveyard.  Only explicit remove_frontiers() calls from planning
        failure and stuck detection should feed the graveyard.
        """
        # Build a stable list before mutating self.frontiers
        invalid_ids = [
            fid for fid, ft in list(self.frontiers.items()) if not ft.is_valid
        ]

        if not invalid_ids:
            self.logger.debug("No invalid frontiers to remove.")
            return

        self.remove_frontiers(invalid_ids, _record_graveyard=False)
        self.logger.debug(f"Removed {len(invalid_ids)} invalid frontiers.")

    def merge_frontiers(self):
        dbscan_params = {
            "eps": 1.8,
            "min_samples": 1,
            "weights": [1, 2],
        }

        valid_frontiers = list(self.valid_frontiers)
        _n_valid_before = len(valid_frontiers)
        if _n_valid_before < 2:
            self.logger.debug("Not enough valid frontiers to merge.")
            return

        positions = np.array([ft.pos3d for ft in valid_frontiers])
        directions = np.array([ft.view_direction for ft in valid_frontiers])
        ft_features = np.concatenate((positions, directions), axis=1)

        metric_func = lambda x, y: ft_pos_direct_distance(
            x, y, weights=dbscan_params["weights"]
        )
        dbscan = DBSCAN(
            eps=dbscan_params["eps"],
            min_samples=dbscan_params["min_samples"],
            metric=metric_func,
            n_jobs=-1,
        )
        labels = dbscan.fit_predict(ft_features)
        _n_valid_before = len(valid_frontiers)

        # Prepare merges
        to_remove = []
        to_add = []

        for cls_id in np.unique(labels):
            if cls_id == -1:
                continue
            _fts = [ft for i, ft in enumerate(valid_frontiers) if labels[i] == cls_id]
            if len(_fts) <= 1:
                continue
            else:
                _ft = Frontier()
                _gains = np.array([max(ft.gain if ft.gain is not None else 1e-4, 1e-4) for ft in _fts])
                _ft.pos3d = np.average(
                    [ft.pos3d for ft in _fts], axis=0, weights=_gains
                )
                cls_vd = np.sum(np.array([ft.view_direction for ft in _fts]), axis=0)
                _ft.view_direction = cls_vd / np.linalg.norm(cls_vd) + 1e-6
                _ft.direct_angle = np.average([ft.direct_angle for ft in _fts], axis=0)
                _ft.pixel_pos = np.average([ft.pixel_pos for ft in _fts], axis=0)
                _ft.set_valid()
                _ft.gain = float(np.max(_gains))
                _ft.u_gain = _ft.gain
                _ft.id = self._alloc_frontier_id()
                _ft.parent_ids = list(
                    set([pid for ft in _fts for pid in ft.parent_ids])
                )
                to_remove.extend([ft.id for ft in _fts])
                to_add.append((_ft, _ft.parent_ids))

        # Remove and add after loop.
        # _record_graveyard=False: members must not enter the graveyard because the
        # merged frontier is placed at their weighted average — within graveyard_radius —
        # and would be immediately killed by filter_frontiers() if they did.
        self.remove_frontiers(to_remove, _record_graveyard=False)
        for _ft, _parent_ids in to_add:
            self.add_frontiers([_ft], parent_ids=_parent_ids)

        _n_valid_after = len(self.valid_frontiers)
        self.logger.debug(
            f"Merged frontiers from {_n_valid_before} to {_n_valid_after} valid frontiers."
        )

    def update_map(self, free_map=None, occ_map=None) -> None:
        """
        Update the planner's space with the occupancy (occ_map) and free space (free_map).
        At least one must be provided. No shape/dtype checks here by design.
        """
        if free_map is None and occ_map is None:
            raise ValueError("At least one of free_map or occ_map must be provided.")

        if free_map is not None:
            self.free_map = free_map
        if occ_map is not None:
            self.occ_map = occ_map

        # Only build KDTree if provided and non-empty
        free_for_kdt = (
            free_map if (free_map is not None and len(free_map) > 0) else None
        )
        occ_for_kdt = occ_map if (occ_map is not None and len(occ_map) > 0) else None
        self.planner.update_space(free_vx=free_for_kdt, occ_vx=occ_for_kdt)

        self.logger.debug(
            "Maps updated: free=%s, occ=%s",
            None if free_map is None else len(free_map),
            None if occ_map is None else len(occ_map),
        )

    def select_goal_frontier_id(self) -> int | None:
        """
        Select the frontier with the highest utility among valid frontiers.
        Sets and returns self.current_goal_ft_id. Returns None if none available.
        """
        fts = self.valid_frontiers
        if not fts:
            self.logger.debug("No valid frontiers available to select as goal.")
            self.current_goal_ft_id = None
            return None

        # Keep only finite utilities
        candidates = [
            (ft.id, float(ft.utility))
            for ft in fts
            if np.isfinite(getattr(ft, "utility", np.nan))
        ]
        if not candidates:
            self.logger.debug("No finite utilities available for goal selection.")
            self.current_goal_ft_id = None
            return None

        # Tie-break: utility desc, then u_gain desc (if present), then smaller ID
        def key_fn(fid_util):
            fid, util = fid_util
            ug = getattr(self.frontiers.get(fid), "u_gain", -np.inf)
            return (util, ug, -fid)

        fid, util = max(candidates, key=key_fn)
        self.current_goal_ft_id = fid

        # Log top-3 candidates so it's easy to see why the winner was chosen
        top3 = sorted(candidates, key=key_fn, reverse=True)[:3]
        lines = []
        for rank, (cid, cu) in enumerate(top3):
            ft = self.frontiers.get(cid)
            marker = " <-- GOAL" if cid == fid else ""
            pos_str = (
                f"({ft.pos3d[0]:+.2f},{ft.pos3d[1]:+.2f},{ft.pos3d[2]:+.2f})"
                if ft and ft.pos3d is not None else "N/A"
            )
            ug = ft.u_gain if ft else float("nan")
            nc = getattr(ft, "n_close", "?") if ft else "?"
            lines.append(
                f"  #{rank+1} F{cid:03d} pos={pos_str} u_gain={ug:.2f} "
                f"n_cl={nc} utility={cu:.4f}{marker}"
            )
        self.logger.info(
            "select_goal: winner F%03d (utility=%.4f)  total_candidates=%d\n%s",
            fid, util, len(candidates), "\n".join(lines),
        )
        return fid

    def gain_adjustment(self, w_history_path: bool = True, w_map: bool = True) -> None:
        """
        Adjust each valid frontier's u_gain based on:
        1) proximity to past robot poses (history), and
        2) optional: visible free voxels from the frontier's camera pose (map).
        Keeps ft.gain as the base and writes ft.u_gain.
        """
        fts = self.valid_frontiers
        n = len(fts)
        if n == 0:
            self.logger.debug("No valid frontiers to adjust gains.")
            return

        # -------------------- ADJUST 1: history proximity --------------------
        if w_history_path:
            robot_ids = self.graph.get_node_R()
            if robot_ids:
                ft_W_T_C = np.stack(
                    [ft.pose6d for ft in fts], axis=0
                )  # (N,4,4)  W_T_C (C→W)
                robot_W_T_R = np.stack(
                    [self.robot_poses[rid] for rid in robot_ids], axis=0
                )  # (M,4,4)

                trans_diff, rot_diff = pose_difference(
                    ft_W_T_C, robot_W_T_R
                )  # (N,M), (N,M)
                close_mask = (trans_diff < self.v_tras_thre) & (
                    rot_diff < self.v_angl_thre
                )
                n_close = close_mask.sum(axis=1).astype(np.float32)  # (N,)
                reduction_1 = self.v_gain_reduction_factor * n_close
            else:
                reduction_1 = np.zeros(n, dtype=np.float32)
        else:
            reduction_1 = np.zeros(n, dtype=np.float32)

        # -------------------- ADJUST 2: map-based visibility --------------------
        voxel_vol = float(self.voxel_size**3)
        reduction_2 = np.zeros(n, dtype=np.float32)

        if w_map and (self.occ_map is not None) and (self.free_map is not None):
            # Frontier pose is W_T_C; for rendering we need C_T_W (extrinsics)
            C_T_W_batch = np.linalg.inv(
                np.stack([ft.pose6d for ft in fts], axis=0)
            )  # (N,4,4)  C_T_W (W→C)

            depths = render_voxel_depth(
                occ_pts=self.occ_map,
                camera_extrinsics=C_T_W_batch,
                render_params={
                    "K": self.render_K,
                    "H": self.render_H,
                    "W": self.render_W,
                    "radius": self.voxel_size / 2,
                },
            )

            for i, (ft, C_T_W, depth) in enumerate(zip(fts, C_T_W_batch, depths)):
                # Count free voxels visible from this frontier view (respecting depth)
                vox_in_fov, _ = select_visible_points(
                    self.free_map,
                    K=self.render_K,
                    T=C_T_W,  # world→camera
                    img_size=(self.render_H, self.render_W),
                    max_depth=self.render_d_range,
                    min_depth_map=depth,
                )

                reduction_2[i] = (
                    float(vox_in_fov.shape[0])
                    * voxel_vol
                    * float(self.render_decrease_factor)
                )

                # Optional refinement: clamp base gain by max possible visible volume from this view
                if (ft.gain is not None) and not np.all(depth < 1e-6):
                    _, max_n_vis = compute_visible_voxels(
                        K=self.render_K,
                        T=C_T_W,  # world→camera
                        img_size=(self.render_H, self.render_W),
                        voxel_size=self.voxel_size,
                        max_depth=self.render_d_range,
                        min_depth_map=depth,
                    )
                    ft.gain = min(float(max_n_vis) * voxel_vol, float(ft.gain))

        self.logger.debug(
            "Before gain adjustment: " + ", ".join(f"{ft.id}: {ft.gain}" for ft in fts)
        )

        for ft, r1, r2 in zip(fts, reduction_1, reduction_2):
            base = float(ft.gain) if ft.gain is not None else 1e-4
            ft.u_gain = max(base - r1 - r2, 1e-4)

        self.logger.debug(
            "After gain adjustment: " + ", ".join(f"{ft.id}: {ft.u_gain}" for ft in fts)
        )

    def gain_adjustment_detr(self) -> None:
        """
        Scale-agnostic history penalty for the DETR pipeline.

        For each valid frontier, counts how many past robot poses lie "close" to
        the frontier's target area and applies an exponential multiplicative decay
        controlled by detr_visited_gain_reduction_factor (default 0.2):

            u_gain = max(gain * decay_factor ^ n_close, 1e-4)

        "Close" is determined by two complementary signals:
          1. pos3d proximity (threshold: detr_visited_dist_threshold, default 2.0 m):
             fires when the robot enters the frontier's target region in world space.
             DETR frontiers sit in occluded space beyond the free-space boundary, so a
             generous threshold is required.
          2. snapped_pos proximity (threshold: v_tras_thre, default 0.25 m):
             fires when the robot physically traverses the free-voxel goal the planner
             navigated to. Set only after path planning; more precise than pos3d.

        Unlike the dense gain_adjustment(), this never subtracts a fixed volume and
        therefore works regardless of the gain's unit or magnitude.
        """
        fts = self.valid_frontiers
        if not fts:
            return

        robot_ids = self.graph.get_node_R()
        if not robot_ids:
            for ft in fts:
                ft.u_gain = max(float(ft.gain) if ft.gain is not None else 1e-4, 1e-4)
            return

        robot_W_T_R = np.stack(
            [self.robot_poses[rid] for rid in robot_ids], axis=0
        )                                                              # (M, 4, 4)
        robot_pos = robot_W_T_R[:, :3, 3]                             # (M, 3)

        # Signal 1: pos3d proximity — generous threshold covers the frontier's
        # target area even when the robot never physically reaches it.
        ft_pos = np.stack([ft.pos3d for ft in fts], axis=0)           # (N, 3)
        dist_pos3d = np.linalg.norm(
            ft_pos[:, np.newaxis, :] - robot_pos[np.newaxis, :, :], axis=2
        )                                                              # (N, M)
        close_mask = dist_pos3d < self.detr_visited_dist_threshold    # (N, M)

        # Signal 2: snapped_pos proximity — tight threshold, fires only when the
        # robot physically passed through the planned navigation goal.
        for i, ft in enumerate(fts):
            snap = ft.snapped_pos
            if snap is not None:
                d_snap = np.linalg.norm(
                    robot_pos - np.asarray(snap, dtype=float), axis=1
                )                                                      # (M,)
                close_mask[i] |= (d_snap < self.v_tras_thre)

        n_close = close_mask.sum(axis=1).astype(np.float32)           # (N,)

        decay = float(self.detr_v_gain_reduction_factor)
        for ft, nc in zip(fts, n_close):
            base = float(ft.gain) if ft.gain is not None else 1e-4
            ft.u_gain = max(base * (decay ** float(nc)), 1e-4)
            ft.n_close = int(nc)

        lines = [
            f"  F{ft.id:03d} gain={ft.gain:6.2f} * {decay:.2f}^{ft.n_close} "
            f"= u_gain={ft.u_gain:6.2f}  pos=({ft.pos3d[0]:+.2f},{ft.pos3d[1]:+.2f},{ft.pos3d[2]:+.2f})"
            for ft in fts
        ]
        self.logger.debug(
            "gain_adjustment_detr: decay=%.2f  n_poses=%d  frontiers=%d\n%s",
            decay, len(robot_ids), len(fts), "\n".join(lines),
        )

    def dedup_new_frontiers(
        self, new_frontiers: List[Frontier], radius: float = 0.5
    ) -> List[Frontier]:
        """
        Filter out frontiers whose 3-D position is within `radius` metres of
        any existing valid frontier, to avoid adding near-duplicates across
        detection intervals.

        Args:
            new_frontiers: Candidate Frontier objects (not yet added to the manager).
            radius:        Exclusion radius in metres (default 0.5 m).

        Returns:
            Subset of new_frontiers that are sufficiently far from all existing
            valid frontiers.
        """
        n_in = len(new_frontiers)

        # Step 1: intra-batch dedup — remove frontiers within radius of an
        # earlier frontier in the same detection batch.
        # Sort by gain descending first so the highest-gain prediction in each
        # spatial cluster is kept instead of the first slot in index order.
        sorted_frontiers = sorted(
            new_frontiers,
            key=lambda f: float(f.gain if f.gain is not None else 0.0),
            reverse=True,
        )
        batch_kept: List[Frontier] = []
        batch_pos: List[np.ndarray] = []
        for ft in sorted_frontiers:
            pos = np.asarray(ft.pos3d, dtype=float)
            if batch_pos:
                nearest_in_batch = min(float(np.linalg.norm(pos - p)) for p in batch_pos)
                if nearest_in_batch <= radius:
                    self.logger.debug(
                        "Intra-batch dedup at (%.2f, %.2f, %.2f): nearest %.3f m",
                        *pos, nearest_in_batch,
                    )
                    continue
            batch_kept.append(ft)
            batch_pos.append(pos)

        # Step 2: cross-frame dedup — remove frontiers within radius of any
        # currently valid frontier from previous detection steps.
        existing = self.valid_frontiers
        if not existing:
            kept = batch_kept
        else:
            existing_pos = np.array([ft.pos3d for ft in existing], dtype=float)
            kept: List[Frontier] = []
            for ft in batch_kept:
                pos = np.asarray(ft.pos3d, dtype=float)
                nearest = float(np.linalg.norm(existing_pos - pos, axis=1).min())
                if nearest > radius:
                    kept.append(ft)
                else:
                    self.logger.debug(
                        "Cross-frame dedup at (%.2f, %.2f, %.2f): nearest existing %.3f m",
                        *pos, nearest,
                    )

        # Step 3: graveyard filter — reject new frontiers near planning-failure positions.
        if self._graveyard:
            graveyard_arr = np.array(self._graveyard, dtype=float)
            survived: List[Frontier] = []
            for ft in kept:
                pos = np.asarray(ft.pos3d, dtype=float)
                nearest = float(np.linalg.norm(graveyard_arr - pos, axis=1).min())
                if nearest > self.graveyard_radius:
                    survived.append(ft)
                else:
                    self.logger.debug(
                        "Graveyard filter (new): dropped at (%.2f,%.2f,%.2f), nearest=%.2f m",
                        *pos, nearest,
                    )
            kept = survived

        n_dropped = n_in - len(kept)
        if n_dropped:
            self.logger.info(
                "dedup_new_frontiers: dropped %d / %d (radius=%.2f m)",
                n_dropped, n_in, radius,
            )
        return kept

    def update_utility(self, current_pos) -> None:
        """
        Update each valid frontier's utility (Eq.(8) in the paper):
            utility = (u_gain ** utility_g_factor) / max(distance, 1e-6)
        """
        p = np.asarray(current_pos, dtype=float).reshape(3)

        valid = self.valid_frontiers
        for ft in valid:
            # Base gain: prefer u_gain, fall back to gain; clamp tiny positive
            base_gain = getattr(ft, "u_gain", None)
            if base_gain is None:
                base_gain = getattr(ft, "gain", 0.0)
            base_gain = float(max(base_gain, 1e-4))

            # Distance to current position.
            # Use min_utility_dist as a floor so that frontiers almost on top
            # of the robot don't get astronomically high utility and permanently
            # out-compete distant but more informative goals.
            d = float(np.linalg.norm(np.asarray(ft.pos3d, dtype=float) - p))
            denom = max(d, self.min_utility_dist) ** self.utility_dist_exponent

            ft.utility = (base_gain ** float(self.utility_g_factor)) / denom

        # Recent-trajectory penalty: scale down utility for frontiers whose pos3d
        # lies within trajectory_penalty_radius of any of the last
        # trajectory_penalty_window robot poses.  This discourages the planner from
        # immediately reversing into a corridor the robot just traversed, reducing
        # the zigzag backtracking pattern without changing the gain accounting.
        if self.trajectory_penalty_radius > 0 and self.robot_poses:
            sorted_rids = sorted(self.robot_poses.keys())
            recent_rids = sorted_rids[-self.trajectory_penalty_window:]
            recent_pts = np.array(
                [self.robot_poses[rid][:3, 3] for rid in recent_rids], dtype=float
            )
            for ft in valid:
                if ft.utility is None:
                    continue
                ft_pos = np.asarray(ft.pos3d, dtype=float)
                d_path = float(np.linalg.norm(recent_pts - ft_pos, axis=1).min())
                if d_path < self.trajectory_penalty_radius:
                    ft.utility = float(ft.utility) * self.trajectory_penalty_factor

        if valid:
            sorted_fts = sorted(valid, key=lambda f: f.utility or 0.0, reverse=True)
            header = (
                f"  {'ID':>5}  {'pos3d (x,y,z)':^26}  "
                f"{'gain':>7}  {'u_gain':>7}  {'n_cl':>4}  {'dist':>5}  {'utility':>8}"
            )
            rows = []
            for ft in sorted_fts:
                d = float(np.linalg.norm(np.asarray(ft.pos3d, dtype=float) - p))
                rows.append(
                    f"  F{ft.id:03d}  "
                    f"({ft.pos3d[0]:+6.2f},{ft.pos3d[1]:+6.2f},{ft.pos3d[2]:+5.2f})  "
                    f"{ft.gain or 0:7.2f}  {ft.u_gain or 0:7.2f}  "
                    f"{getattr(ft, 'n_close', 0):4d}  {d:5.2f}  {ft.utility or 0:8.4f}"
                )
            self.logger.debug(
                "update_utility: %d frontiers  graveyard=%d\n%s\n%s",
                len(valid), len(self._graveyard),
                header, "\n".join(rows),
            )

    def filter_frontiers(self) -> None:
        """
        Mark frontiers invalid based on:
        1) bounding box (if set)
        2) min gain threshold
        3) view-direction z limit (if set)
        4) proximity to occupied space (if occ_map present)
        Then remove all invalid frontiers.
        """
        if not self.frontiers:
            self.logger.debug("No frontiers to filter.")
            return

        bbox = self.filter_bbox
        min_gain = float(self.filter_min_gain)
        max_vd_z = self.filter_max_vd_z
        check_occ = self.occ_map is not None

        # graveyard_arr is kept mutable during the loop (fix defect 6): when a
        # frontier is gain-filtered with n_close>0 its position is appended to
        # self._graveyard and graveyard_arr is extended in-place so that
        # subsequent frontiers in the same call are checked against it.
        graveyard_arr = np.array(self._graveyard, dtype=float) if self._graveyard else None

        # Current robot position for the proximity-retirement check (computed once).
        cur_robot_pos = None
        if self.proximity_retirement_radius > 0 and self.robot_poses:
            latest_rid = self.current_robot_id
            if latest_rid is not None:
                cur_robot_pos = np.asarray(
                    self.robot_poses[latest_rid][:3, 3], dtype=float
                )

        n_before = sum(1 for ft in self.frontiers.values() if ft.is_valid)
        self.logger.info(f"[filter] Start: {n_before} valid frontiers")

        n_bbox_removed = 0
        n_prox_removed = 0
        n_gain_removed = 0
        n_ugain_removed = 0
        n_nclose_removed = 0
        n_vdz_removed = 0
        n_occ_removed = 0
        n_grave_removed = 0
        n_freespace_removed = 0

        for fid, ft in list(self.frontiers.items()):
            if not ft.is_valid:
                continue  # already invalid elsewhere

            # 0) Proximity retirement: immediately retire any non-goal frontier whose
            # pos3d is within proximity_retirement_radius of the current robot position.
            # The active goal is excluded so in-progress navigation is never disrupted.
            # Retired frontiers are added to the graveyard so they are not re-detected.
            if cur_robot_pos is not None and fid != self.current_goal_ft_id:
                d_robot = float(
                    np.linalg.norm(np.asarray(ft.pos3d, dtype=float) - cur_robot_pos)
                )
                if d_robot < self.proximity_retirement_radius:
                    ft.set_invalid()
                    n_prox_removed += 1
                    pos_arr = np.asarray(ft.pos3d, dtype=float).copy()
                    self._graveyard.append(pos_arr)
                    if graveyard_arr is None:
                        graveyard_arr = pos_arr.reshape(1, 3)
                    else:
                        graveyard_arr = np.vstack([graveyard_arr, pos_arr.reshape(1, 3)])
                    self.logger.debug(
                        "Frontier %d retired (proximity, d=%.2f m < %.2f m).",
                        fid, d_robot, self.proximity_retirement_radius,
                    )
                    continue

            # 1) Bounding box filter
            if bbox is not None:
                x, y, z = map(float, ft.pos3d)
                if not (
                    bbox[0] <= x <= bbox[1]
                    and bbox[2] <= y <= bbox[3]
                    and bbox[4] <= z <= bbox[5]
                ):
                    ft.set_invalid()
                    n_bbox_removed += 1
                    self.logger.debug(f"Frontier {fid} invalid (bbox).")
                    continue

            # 2) Gain filter — use raw detected gain, NOT the visit-decayed u_gain.
            # u_gain decay is for utility *ranking* only; filtering on it would
            # eliminate valid room frontiers the moment the robot passes within
            # detr_visited_dist_threshold of them (peek-and-move-on bug).
            g = float(ft.gain if ft.gain is not None else 0.0)
            if g < min_gain:
                ft.set_invalid()
                n_gain_removed += 1
                self.logger.debug(f"Frontier {fid} invalid (gain<{min_gain}).")
                continue

            # 2b) u_gain retirement: retire frontiers whose visit-decayed effective
            # gain has fallen to the 1e-4 clamp floor in gain_adjustment_detr().
            # The raw-gain filter (above) only removes never-detected frontiers;
            # this check removes frontiers that have been physically visited so many
            # times that they carry no usable information signal, even though their
            # raw gain is still above filter_min_gain.  We use 1e-3 as the threshold
            # (10× the clamp floor) to retire slightly before the absolute floor so
            # near-zero-utility frontiers never win goal selection.
            ug = float(getattr(ft, "u_gain", None) or 0.0)
            if ug <= 1e-3 and fid != self.current_goal_ft_id:
                ft.set_invalid()
                n_ugain_removed += 1
                pos_arr = np.asarray(ft.pos3d, dtype=float).copy()
                self._graveyard.append(pos_arr)
                if graveyard_arr is None:
                    graveyard_arr = pos_arr.reshape(1, 3)
                else:
                    graveyard_arr = np.vstack([graveyard_arr, pos_arr.reshape(1, 3)])
                self.logger.debug(
                    "Frontier %d retired (u_gain=%.2e <= 1e-3, n_close=%d).",
                    fid, ug, getattr(ft, "n_close", 0),
                )
                continue

            # 2c) n_close retirement: retire frontiers the robot has passed near enough
            # times to be considered definitively explored.  Separate from the gain
            # filter so a single doorway peek (n_close=1) never eliminates a frontier,
            # but thorough traversal (n_close >= detr_max_n_close) does.
            if self.detr_max_n_close > 0:
                nc = getattr(ft, "n_close", 0)
                if nc >= self.detr_max_n_close and fid != self.current_goal_ft_id:
                    ft.set_invalid()
                    n_nclose_removed += 1
                    pos_arr = np.asarray(ft.pos3d, dtype=float).copy()
                    self._graveyard.append(pos_arr)
                    if graveyard_arr is None:
                        graveyard_arr = pos_arr.reshape(1, 3)
                    else:
                        graveyard_arr = np.vstack([graveyard_arr, pos_arr.reshape(1, 3)])
                    self.logger.debug(
                        "Frontier %d retired (n_close=%d >= %d).", fid, nc, self.detr_max_n_close
                    )
                    continue

            # 3) View-direction z filter
            if max_vd_z is not None:
                vd_z = float(ft.view_direction[2])
                if abs(vd_z) > max_vd_z:
                    ft.set_invalid()
                    n_vdz_removed += 1
                    self.logger.debug(f"Frontier {fid} invalid (|vd_z|>{max_vd_z}).")
                    continue

            # 4) Too close to occupied space
            if check_occ and self.planner.isoccupied(ft.pos3d):
                ft.set_invalid()
                n_occ_removed += 1
                self.logger.debug(f"Frontier {fid} invalid (near occupied).")
                continue

            # 5) Too close to a graveyard entry (previously removed frontier location)
            if graveyard_arr is not None:
                nearest_grave = float(
                    np.linalg.norm(graveyard_arr - np.asarray(ft.pos3d, dtype=float), axis=1).min()
                )
                if nearest_grave <= self.graveyard_radius:
                    ft.set_invalid()
                    n_grave_removed += 1
                    self.logger.debug(
                        "Frontier %d invalid (graveyard, nearest=%.2f m).", fid, nearest_grave
                    )
                    continue

            # 6) Not in free navigable space (outside walls / unreachable).
            # Only meaningful when the free map is pre-loaded and complete (voxel_grid mode).
            # Enable via filter_not_in_freespace=True in config.  When freespace_filter_dist > 0,
            # the check is relaxed: reject only frontiers whose nearest free voxel is farther
            # than that distance.  Use ~1.5 m for DETR mode where frontier positions land at
            # the depth of occluded surfaces and may be slightly beyond the free-space boundary.
            if self.filter_not_in_freespace and self.planner._free_kdt is not None:
                if self.freespace_filter_dist > 0.0:
                    dist_to_free, _ = self.planner._free_kdt.query(ft.pos3d)
                    is_too_far = float(dist_to_free) > self.freespace_filter_dist
                else:
                    is_too_far = not self.planner.isfree(ft.pos3d)
                if is_too_far:
                    ft.set_invalid()
                    n_freespace_removed += 1
                    self.logger.debug("Frontier %d invalid (not in free space).", fid)
                    continue

        n_after_prox = n_before - n_prox_removed
        n_after_bbox = n_after_prox - n_bbox_removed
        n_after_gain = n_after_bbox - n_gain_removed
        n_after_ugain = n_after_gain - n_ugain_removed
        n_after_nclose = n_after_ugain - n_nclose_removed
        n_after_vdz = n_after_nclose - n_vdz_removed
        n_after_occ = n_after_vdz - n_occ_removed
        n_after_grave = n_after_occ - n_grave_removed
        n_after_freespace = n_after_grave - n_freespace_removed

        if n_prox_removed:
            self.logger.info(
                f"[filter] After proximity:  {n_after_prox} valid  "
                f"(-{n_prox_removed}, r={self.proximity_retirement_radius:.2f}m)"
            )
        if bbox is not None:
            self.logger.info(f"[filter] After bbox:       {n_after_bbox} valid  (-{n_bbox_removed})")
        self.logger.info(f"[filter] After gain:       {n_after_gain} valid  (-{n_gain_removed}, threshold={min_gain})")
        if n_ugain_removed:
            self.logger.info(f"[filter] After u_gain:     {n_after_ugain} valid  (-{n_ugain_removed}, threshold=1e-3)")
        if self.detr_max_n_close > 0:
            self.logger.info(
                f"[filter] After n_close:    {n_after_nclose} valid  "
                f"(-{n_nclose_removed}, max_n_close={self.detr_max_n_close})"
            )
        if max_vd_z is not None:
            self.logger.info(f"[filter] After vd_z:       {n_after_vdz} valid  (-{n_vdz_removed}, max_vd_z={max_vd_z})")
        if check_occ:
            self.logger.info(f"[filter] After occ:        {n_after_occ} valid  (-{n_occ_removed})")
        if self._graveyard:
            self.logger.info(f"[filter] After graveyard:  {n_after_grave} valid  (-{n_grave_removed}, r={self.graveyard_radius:.2f}m)")
        if self.filter_not_in_freespace:
            self.logger.info(f"[filter] After freespace:  {n_after_freespace} valid  (-{n_freespace_removed})")

        # 7) Hard cap: retire lowest-utility frontiers when count exceeds max_active_frontiers.
        # Record retired positions to the graveyard so they are not immediately re-detected.
        n_cap_removed = 0
        if self.max_active_frontiers > 0:
            valid_now = [
                (fid, ft) for fid, ft in self.frontiers.items() if ft.is_valid
            ]
            if len(valid_now) > self.max_active_frontiers:
                valid_now.sort(key=lambda x: float(x[1].utility or 0.0))
                n_retire = len(valid_now) - self.max_active_frontiers
                for fid, ft in valid_now[:n_retire]:
                    ft.set_invalid()
                    n_cap_removed += 1
                    self._graveyard.append(np.asarray(ft.pos3d, dtype=float).copy())
            if n_cap_removed:
                self.logger.info(
                    "[filter] After cap:     %d valid  (-%d, max=%d)",
                    self.max_active_frontiers, n_cap_removed, self.max_active_frontiers,
                )

        n_final = n_after_freespace - n_cap_removed
        self.logger.info(f"[filter] End:   {n_final} valid frontiers remain")

        # Purge all marked frontiers (graveyard entries were already written
        # inline during the loop; no separate flush needed).
        self.remove_invalid_frontiers()

    def _segment_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Return True if no point along the segment p1→p2 is within min_dist2occ
        of an occupied voxel. Samples at half the min_dist2occ spacing.
        """
        seg = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            return not self.planner.isoccupied(p1)
        step = max(self.planner._min_dist2occ / 2.0, 1e-3)
        n = max(2, int(np.ceil(length / step)))
        for t in np.linspace(0.0, 1.0, n):
            if self.planner.isoccupied(p1 + t * seg):
                return False
        return True

    def get_waypoints_from_graph(self, ft_id: int, robot_pose_id: int) -> np.ndarray:
        """
        Compute a final approach pose W_T_C for the target frontier, following the graph path
        from the given robot pose. Densifies with ~0.1 m spacing, finds the last reachable
        free-space point, and orients toward the next point (or the frontier if none).
        """
        # --- Validate inputs and fetch frontier ---
        assert ft_id in self.frontiers, f"Frontier ID {ft_id} not found."
        frontier = self.frontiers[ft_id]  # has .pos3d, .view_direction, .pose6d (W_T_C)

        # --- Shortest path (node ids R…→F) ---
        path, _ = self.graph.get_shortest_R_to_F(
            robot_id=robot_pose_id, frontier_id=ft_id
        )
        if not path:
            # No route; fall back to frontier’s own pose
            return frontier.pose6d

        # --- Coarse waypoints: robot node positions, then frontier position ---
        coarse_pts = []
        for node_id in path[:-1]:
            W_T_R = self.get_robot_pose(node_id)
            if W_T_R is None:
                raise KeyError(f"Robot pose for node {node_id} not found.")
            coarse_pts.append(W_T_R[:3, 3])
        coarse_pts.append(frontier.pos3d)

        # --- Densify path at ~0.1 m (no duplicate endpoints between segments) ---
        STEP = 0.1
        MAX_PER_SEG = 500
        fine_pts = []
        prev = None
        for cur in coarse_pts:
            cur = np.asarray(cur, dtype=float)
            if prev is None:
                prev = cur
                continue
            seg = cur - prev
            dist = float(np.linalg.norm(seg))
            if dist <= 1e-6:
                fine_pts.append(cur)
                prev = cur
                continue
            n = int(np.ceil(dist / STEP))
            n = min(max(n, 1), MAX_PER_SEG)
            # sample t in (0,1], i.e., exclude prev to avoid duplicates, include cur
            ts = np.linspace(1.0 / n, 1.0, n)
            pts = prev + ts[:, None] * seg
            fine_pts.extend(pts)
            prev = cur

        if not fine_pts:
            # No densification possible; use the frontier pose
            return frontier.pose6d

        # --- Reverse search: last reachable point (free and not occupied) ---
        break_idx = None
        for idx in range(len(fine_pts) - 1, -1, -1):
            pt = fine_pts[idx]
            if self.planner.isfree(pt) and not self.planner.isoccupied(pt):
                break_idx = idx
                break

        if break_idx is None:
            # Nothing along the path is free → go with the frontier pose
            return frontier.pose6d

        break_pt = np.asarray(fine_pts[break_idx], dtype=float)

        # If the direct approach is blocked far from the frontier (wall in the way),
        # query the free-voxel KD-tree for the nearest free voxel to the frontier.
        # That voxel is on the mapper-observed side of any opening (doorway, gap) and
        # gives RRT* a goal it can actually route through, rather than the wall face.
        #
        # Crucially, reachability is checked against *every robot-pose waypoint* on
        # the graph path (closest-to-frontier first), not only the current position.
        # A frontier may be blocked from the current standpoint but perfectly reachable
        # from an earlier waypoint; in that case a valid snap goal exists and RRT*
        # (with the full global map) can plan the backtracking route automatically.
        dist_to_ft = float(np.linalg.norm(break_pt - np.asarray(frontier.pos3d, dtype=float)))
        if dist_to_ft > self.v_tras_thre and self.planner._free_kdt is not None:
            k = min(50, self.planner._free_kdt.n)
            _, nn_idxs = self.planner._free_kdt.query(frontier.pos3d, k=k)
            # Probe positions: all robot waypoints on the path, closest to the
            # frontier first (most likely to have line-of-sight to the snap goal).
            probe_positions = [
                np.asarray(p, dtype=float) for p in reversed(coarse_pts[:-1])
            ]
            found = False
            for nn_idx in np.atleast_1d(nn_idxs):
                candidate = np.asarray(self.planner._free_kdt.data[nn_idx], dtype=float)
                if self.planner.isoccupied(candidate):
                    continue
                for probe in probe_positions:
                    if self._segment_collision_free(probe, candidate):
                        self.logger.debug(
                            "Behind-wall frontier: snap goal (%.2f,%.2f,%.2f) reachable "
                            "from path waypoint (%.2f,%.2f,%.2f), dist_to_ft=%.2f m.",
                            *candidate, *probe, dist_to_ft,
                        )
                        break_pt = candidate
                        found = True
                        break
                if found:
                    break
            if not found:
                # The frontier is not reachable from any known waypoint along the
                # graph path.  Penalise gain so it is deprioritised; repeated
                # failures will drive it below filter_min_gain and remove it.
                ft = self.frontiers[ft_id]
                ft.gain   = max(float(ft.gain   or 1e-4) * 0.5, 1e-4)
                ft.u_gain = max(float(ft.u_gain or 1e-4) * 0.5, 1e-4)
                self.logger.debug(
                    "Behind-wall frontier %d: no reachable snap found from any of %d "
                    "path waypoints among %d candidates — penalising gain (×0.5).",
                    ft_id, len(probe_positions), k,
                )
                return None

        # Close enough to the frontier? keep its orientation, place at break_pt
        if np.linalg.norm(break_pt - frontier.pos3d) < self.v_tras_thre:
            W_T_C = frontier.pose6d.copy()
            W_T_C[:3, 3] = break_pt
            return W_T_C

        # --- Orientation at the break point ---
        # Prefer direction toward the next waypoint if there is one; otherwise toward the frontier.
        if break_idx < len(fine_pts) - 1:
            target = np.asarray(fine_pts[break_idx + 1], dtype=float)
        else:
            target = np.asarray(frontier.pos3d, dtype=float)

        direction = target - break_pt
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            # Degenerate → fall back to the frontier's view direction, or +Z
            vd = np.asarray(frontier.view_direction, dtype=float)
            vd_norm = float(np.linalg.norm(vd))
            direction = (
                (vd / vd_norm)
                if vd_norm > 1e-8
                else np.array([0.0, 0.0, 1.0], dtype=float)
            )
        else:
            direction /= norm

        W_T_C = compute_alignment_transforms(
            origins=[break_pt],
            align_vec=direction,  # +Z aligns with travel direction
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],  # CV convention
            appr_axis=[0, 1, 0],
        )[0]
        W_T_C[:3, 3] = break_pt
        return W_T_C

    def _snap_goal_to_free_voxel(self, frontier, robot_pos=None) -> np.ndarray:
        """
        Find the nearest valid (free and not occupied) voxel to the frontier position
        and return a goal pose there, oriented toward the frontier.
        Falls back to the frontier's own pose if no valid voxel is found.

        Args:
            frontier:  Frontier whose position is used as the KDTree query point.
            robot_pos: (3,) current robot position in world frame.  When provided,
                       candidates within v_tras_thre of the robot are skipped in the
                       first pass so the snap goal is never placed at the robot's own
                       position (which would produce a zero-length path and a
                       micro-loop where the step counter never advances).
        """
        if self.planner._free_kdt is None:
            return self.get_frontier_pose(frontier)

        pos = np.asarray(frontier.pos3d, dtype=float)
        robot = np.asarray(robot_pos, dtype=float).reshape(3) if robot_pos is not None else None
        k = min(200, self.planner._free_kdt.n)
        _, nn_idxs = self.planner._free_kdt.query(pos, k=k)

        snap_pt = None
        for idx in np.atleast_1d(nn_idxs):
            candidate = np.asarray(self.planner._free_kdt.data[idx], dtype=float)
            if self.planner.isoccupied(candidate):
                continue
            # Skip candidates that are at (or very near) the robot's current position.
            # DETR frontiers land in occluded space beyond walls; their nearest free
            # voxel is often the robot's own voxel, producing a start==goal path.
            if robot is not None and float(np.linalg.norm(candidate - robot)) < self.v_tras_thre:
                continue
            snap_pt = candidate
            break

        # Fallback: if every candidate was within v_tras_thre of the robot (e.g. the
        # frontier is genuinely very close), relax the exclusion and take the best one.
        if snap_pt is None:
            for idx in np.atleast_1d(nn_idxs):
                candidate = np.asarray(self.planner._free_kdt.data[idx], dtype=float)
                if not self.planner.isoccupied(candidate):
                    snap_pt = candidate
                    break

        if snap_pt is None:
            self.logger.warning(
                "snap_goal_to_free_voxel: no valid free voxel found near frontier %s; "
                "using raw frontier pose.",
                frontier.id,
            )
            return self.get_frontier_pose(frontier)

        # Orient toward the frontier from the snap point
        direction = pos - snap_pt
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = np.asarray(frontier.view_direction, dtype=float)
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        else:
            direction /= norm

        W_T_C = compute_alignment_transforms(
            origins=[snap_pt],
            align_vec=direction,
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],
            appr_axis=[0, 1, 0],
        )[0]
        W_T_C[:3, 3] = snap_pt
        # Record the snap point on the frontier so gain_adjustment_detr can detect
        # when the robot physically traverses this planned navigation goal.
        frontier.snapped_pos = snap_pt.copy()
        self.logger.debug(
            "snap_goal_to_free_voxel: snapped frontier %s from (%.2f,%.2f,%.2f) "
            "to (%.2f,%.2f,%.2f), dist=%.2f m.",
            frontier.id, *pos, *snap_pt, float(np.linalg.norm(snap_pt - pos)),
        )
        return W_T_C

    def plan_path_to_goal(
        self,
        current_pose: np.ndarray,
        interpolate_solution: bool = True,
        use_graph: bool = True,
        direct_voxel_snap: bool = False,
    ) -> list[np.ndarray] | None:
        """
        Plan a path from the current robot pose W_T_R to the selected goal frontier.

        Args:
            current_pose: (4,4) W_T_R (R→W) robot pose in world frame.
            interpolate_solution: If True, densify the planner's solution.
            use_graph: If True, compute a goal pose via graph waypoints; else use the frontier pose.
            direct_voxel_snap: If True, skip graph logic entirely and snap the goal to the
                nearest valid free voxel, then let RRT* plan the full path. Useful when a
                complete pre-built voxel map is available from the start.

        Returns:
            List of (4,4) poses along the path (world-frame), or None if planning fails.
        """

        # Select goal frontier
        goal_ft_id = self.select_goal_frontier_id()
        self.logger.debug(f"Planning path to goal frontier ID: {goal_ft_id}")
        if goal_ft_id is None:
            self.logger.debug("No goal frontier available for path planning.")
            return None
        goal_ft = self.frontiers[goal_ft_id]

        # Decide goal pose
        if direct_voxel_snap:
            # Bypass all graph / intermediate-step logic.  The RRT* has the full voxel
            # map, so just snap the goal to the nearest free voxel and let it route.
            # Pass the robot position so snap avoids landing on the robot's own voxel.
            goal_pose = self._snap_goal_to_free_voxel(goal_ft, robot_pos=current_pose[:3, 3])
        elif use_graph:
            current_pose_id = self.add_robot_poses([current_pose])[0]
            goal_pose = self.get_waypoints_from_graph(goal_ft_id, current_pose_id)
            if goal_pose is None:
                # Frontier is temporarily unreachable from the current position;
                # gain has already been penalised — skip this cycle without dropping.
                self.current_goal_ft_id = None
                self.current_goal_pose = None
                return None
        else:
            goal_pose = self.get_frontier_pose(goal_ft)

        self.planner.update_start_goal(start=current_pose, goal=goal_pose)
        self.current_goal_pose = (
            goal_pose  # Update the current goal pose to the last waypoint
        )

        if self.robot_poses:
            robot_pts = np.array(
                [pose[:3, 3] for pose in self.robot_poses.values()], dtype=float
            )
            self.planner.set_visit_data(robot_pts)

        try:
            path_found = self.planner.solve(
                time_limit=self.max_planning_time, method=self.planning_algo
            )

            if not path_found:
                # Drop this frontier as a goal candidate but do NOT add it to the
                # graveyard: path failure means the robot cannot reach it from its
                # current position, not that the frontier itself is invalid.  Adding
                # to the graveyard would block future detections of valid areas once
                # the robot navigates to a different vantage point.
                self.remove_frontiers([goal_ft_id], _record_graveyard=False)
                self.current_goal_ft_id = None
                self.current_goal_pose = None
                return None

            if interpolate_solution:
                self.planner.interpolate_path()

            return self.planner.get_solution_path(return_type="mat")

        except Exception as e:
            self.logger.error(f"Path planning failed: {e}")
            self.remove_frontiers([goal_ft_id], _record_graveyard=False)
            self.current_goal_ft_id = None
            self.current_goal_pose = None
            self.logger.debug("Dropped goal frontier due to planning failure.")
            return None

    def get_graph_vis(self):
        """
        Get the visualization of the frontier graph.
        Returns:
            o3d geometry to visualize the graph.
            Frontiers as red spheres, robots as blue spheres, edges as thin cylinders.
        """
        ft_ids = self.graph.get_node_F()
        robot_ids = self.graph.get_node_R()
        FR_edges = self.graph.get_edge_FR()
        RR_edges = self.graph.get_edge_RR()
        geometries = []
        # Visualize frontiers
        for ft_id in ft_ids:
            ft = self.frontiers[ft_id]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.075)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(ft.pos3d)
            geometries.append(sphere)
        # Visualize robots/cameras
        for robot_id in robot_ids:
            pose = self.robot_poses[robot_id]
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            coordinate_frame.transform(pose)
            geometries.append(coordinate_frame)
        # Visualize edges
        for edge in FR_edges:
            start_id, end_id, _ = edge
            start_pos = (
                self.frontiers[start_id].pos3d
                if start_id in self.frontiers
                else self.robot_poses[start_id][:3, 3]
            )
            end_pos = (
                self.frontiers[end_id].pos3d
                if end_id in self.frontiers
                else self.robot_poses[end_id][:3, 3]
            )
            cylinder = create_cylinder_between_points(
                start_pos, end_pos, radius=0.012, color=[0, 0, 0]
            )
            if cylinder is not None:
                geometries.append(cylinder)
        for edge in RR_edges:
            start_id, end_id, _ = edge
            start_pos = self.robot_poses[start_id][:3, 3]
            end_pos = self.robot_poses[end_id][:3, 3]
            cylinder = create_cylinder_between_points(
                start_pos, end_pos, radius=0.02, color=[0.2, 0.8, 0.2]
            )
            if cylinder is not None:
                geometries.append(cylinder)
        return geometries

    ## -- File I/O --
    @staticmethod
    def _json_default(o):
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    def write_to_file(self, file_path: str) -> None:
        """
        Append a JSON line snapshot of the manager state.
        """
        entry = {
            "all_frontiers": [ft.to_dict() for ft in self.all_frontiers],
            "valid_frontiers": [ft.to_dict() for ft in self.valid_frontiers],
            "robot_poses": {
                rid: pose.tolist() for rid, pose in self.robot_poses.items()
            },
            "current_robot_id": self._current_robot_id,
            "current_ft_goal_id": (
                self.current_goal_ft_id if self.current_goal_ft_id is not None else None
            ),
            "current_goal_pose": (
                None
                if self.current_goal_pose is None
                else self.current_goal_pose.tolist()
            ),
            "graph": self.graph.return_current_graph(),
        }

        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(entry, default=self._json_default) + "\n")
            self.logger.debug(
                f"FrontierManager state appended to {file_path} (JSON lines)."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to write FrontierManager state to {file_path}: {e}"
            )

    @staticmethod
    def read_from_file(file_path):
        """
        Read the state of the FrontierManager from a file.
        Args:
            file_path: Path to the file to read from.

        Returns:
            A List of entrys, each entry is a dictionary containing the state of the FrontierManager.
        """
        entries = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    entries.append(entry)
            return entries
        except Exception as e:
            print(f"Failed to read FrontierManager state from {file_path}: {e}")
            return []
