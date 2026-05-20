"""
Classic frontier detector — Yamauchi (1997).

A frontier is the boundary between observed-free space and unobserved
(unknown) space in the occupancy map.  This implementation detects 3-D
frontier voxels directly from the wavemap free/occupied point clouds,
clusters them with DBSCAN, and wraps each cluster as a Frontier object
compatible with FrontierManager.

No neural network is required; frontiers are derived purely from the
robot's accumulated occupancy observations.
"""

import numpy as np
import logging
from typing import List, Optional

from sklearn.cluster import DBSCAN

from frontier.frontier import Frontier


class ClassicFrontierDetector:
    """
    Map-based frontier detector (Yamauchi 1997).

    Algorithm:
      1. Quantise observed free voxels to a regular grid.
      2. A free voxel is a *frontier voxel* if any of its 6-connected
         grid neighbours is neither free nor occupied (i.e. unknown).
      3. Frontier voxels are clustered with DBSCAN.
      4. Each cluster becomes one Frontier goal with gain proportional
         to cluster size (unexplored-volume proxy).
    """

    # 6-connected neighbourhood offsets
    _OFFSETS_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def __init__(
        self,
        voxel_size: float = 0.1,
        camera_intrinsic: Optional[np.ndarray] = None,
        min_frontier_size: int = 5,
        cluster_eps: float = 0.4,
        log_level: int = logging.INFO,
    ):
        """
        Args:
            voxel_size:         Must match the wavemap resolution (metres).
            camera_intrinsic:   3×3 K matrix for projecting frontier centroids
                                to pixel coordinates (may be None).
            min_frontier_size:  DBSCAN min_samples — clusters smaller than this
                                are treated as noise and dropped.
            cluster_eps:        DBSCAN neighbourhood radius (metres).
            log_level:          Python logging level.
        """
        self.voxel_size = voxel_size
        self.camera_intrinsic = camera_intrinsic
        self.min_frontier_size = min_frontier_size
        self.cluster_eps = cluster_eps

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        free_pts: np.ndarray,
        occ_pts: Optional[np.ndarray],
        W_T_C: np.ndarray,
    ) -> List[Frontier]:
        """
        Detect frontiers from the current occupancy map.

        Args:
            free_pts:  (N, 3) float32 — observed free voxel centres.
            occ_pts:   (M, 3) float32 — observed occupied voxel centres.
            W_T_C:     (4, 4) float64 — camera pose in world frame
                       (W_T_C[:3, 3] == camera position in world).

        Returns:
            List of Frontier objects (may be empty).
        """
        if free_pts is None or len(free_pts) == 0:
            self.logger.warning("Classic: no free points — skipping detection.")
            return []

        # Step 1 — find frontier voxels
        frontier_pts = self._find_frontier_voxels(free_pts, occ_pts)
        self.logger.info("Classic: %d frontier voxels found.", len(frontier_pts))
        if len(frontier_pts) == 0:
            return []

        # Step 2 — cluster
        labels = self._cluster(frontier_pts)
        unique_labels = [lbl for lbl in set(labels) if lbl >= 0]
        self.logger.info(
            "Classic: %d frontier clusters (eps=%.2f m, min_samples=%d).",
            len(unique_labels), self.cluster_eps, self.min_frontier_size,
        )
        if not unique_labels:
            return []

        # Step 3 — build Frontier objects
        cam_pos = W_T_C[:3, 3]
        frontiers: List[Frontier] = []
        for lbl in unique_labels:
            ft = self._make_frontier(frontier_pts[labels == lbl], cam_pos, W_T_C)
            if ft is not None:
                frontiers.append(ft)

        self.logger.info("Classic: returning %d frontiers.", len(frontiers))
        return frontiers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_frontier_voxels(
        self,
        free_pts: np.ndarray,
        occ_pts: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return world-frame 3-D positions of all frontier voxels."""
        vs = self.voxel_size

        # Quantise points to integer voxel indices
        free_idx = np.round(free_pts / vs).astype(np.int32)

        # Build sets for O(1) neighbour lookup
        free_set: set = set(map(tuple, free_idx))
        known_set: set = set(free_set)
        if occ_pts is not None and len(occ_pts) > 0:
            known_set.update(
                map(tuple, np.round(occ_pts / vs).astype(np.int32))
            )

        # A free voxel is a frontier voxel if any 6-neighbour is unknown
        frontier_centers: List[np.ndarray] = []
        for ijk in free_set:
            i, j, k = ijk
            for di, dj, dk in self._OFFSETS_6:
                if (i + di, j + dj, k + dk) not in known_set:
                    frontier_centers.append(
                        np.array([i, j, k], dtype=np.float32) * vs
                    )
                    break  # only add each free voxel once

        if not frontier_centers:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(frontier_centers, dtype=np.float32)

    def _cluster(self, pts: np.ndarray) -> np.ndarray:
        """Run DBSCAN on frontier voxel positions; return label array."""
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_frontier_size,
            algorithm="ball_tree",
            n_jobs=1,
        ).fit(pts)
        return clustering.labels_

    def _make_frontier(
        self,
        cluster_pts: np.ndarray,
        cam_pos: np.ndarray,
        W_T_C: np.ndarray,
        retreat_dist: float = 0.3,
    ) -> Optional[Frontier]:
        """Build one Frontier from a cluster of frontier voxels.

        Two adjustments are made so the frontier goal survives the
        FrontierManager's downstream filters (which were designed for
        neural-network outputs, not classic map-based frontiers):

        1. Goal position retreated toward robot:
           Classic frontier voxels sit right on the free/unknown boundary,
           ~0.1 m from the nearest occupied mesh voxel.  The manager's
           isoccupied() filter rejects points within min_dist2occ=0.2 m of
           occupied voxels, so we pull the centroid `retreat_dist` metres
           toward the robot in XY to move it safely inside free space.

        2. View direction projected to XY plane:
           The filter_max_vd_z=0.65 filter was designed for neural-network
           frontiers that sometimes predict straight up/down.  Classic
           frontiers are always horizontal goals, so we zero out vd_z to
           ensure the filter is never triggered.
        """
        centroid = cluster_pts.mean(axis=0).astype(np.float64)

        # Horizontal direction from frontier toward robot (retreat direction)
        to_robot_xy = np.array([cam_pos[0] - centroid[0],
                                cam_pos[1] - centroid[1], 0.0])
        to_robot_norm = float(np.linalg.norm(to_robot_xy))

        if to_robot_norm < 1e-6:
            return None

        to_robot_xy /= to_robot_norm

        # Fix 1: retreat centroid toward robot to clear occupied-proximity filter
        goal = centroid.copy()
        goal[:2] += to_robot_xy[:2] * retreat_dist

        # Fix 2: horizontal-only view direction (camera → frontier, Z zeroed)
        vd_xy = -to_robot_xy.copy()   # frontier direction = away from robot
        vd_xy[2] = 0.0
        vd_xy /= float(np.linalg.norm(vd_xy) + 1e-9)

        # Gain = cluster size (proxy for unexplored volume behind frontier)
        gain = float(len(cluster_pts))

        # 2-D viewing angle in XY plane
        direct_angle = float(np.arctan2(vd_xy[1], vd_xy[0]))

        # Approximate pixel coordinates (use original centroid for projection)
        pixel_pos = self._project_to_pixel(centroid, W_T_C)

        f = Frontier()
        f.pos3d = goal
        f.gain = gain
        f.u_gain = gain
        f.view_direction = vd_xy
        f.direct_angle = direct_angle
        f.pixel_pos = pixel_pos
        f.set_valid()
        return f

    def _project_to_pixel(
        self, world_pt: np.ndarray, W_T_C: np.ndarray
    ) -> np.ndarray:
        """Project a world point into image pixel coordinates."""
        if self.camera_intrinsic is None:
            return np.zeros(2)
        C_T_W = np.linalg.inv(W_T_C)
        cam_pt = (C_T_W @ np.append(world_pt, 1.0))[:3]
        if cam_pt[2] <= 0.0:
            return np.zeros(2)
        K = self.camera_intrinsic
        u = K[0, 0] * cam_pt[0] / cam_pt[2] + K[0, 2]
        v = K[1, 1] * cam_pt[1] / cam_pt[2] + K[1, 2]
        return np.array([u, v])
