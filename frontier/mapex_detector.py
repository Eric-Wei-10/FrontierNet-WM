"""
MapEx-based frontier detector (ICRA 2025 baseline).

Uses the LaMa inpainting ensemble (MapEx) to predict unknown regions of the
top-down 2-D occupancy map, detects frontier candidates at the boundary
between free and unknown space, and scores each by the ensemble's predictive
variance — a proxy for the information gain used by MapEx's 'visvarprob' mode.

Interface is identical to ClassicFrontierDetector:
    ft_list = detector.detect(free_pts, occ_pts, W_T_C)

No dependency on range_libc or sim_utils — only lama_pred_utils is used.
"""

import sys
import logging
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from frontier.frontier import Frontier


class MapExFrontierDetector:
    """
    Frontier detector using MapEx's LaMa inpainting ensemble.

    Pipeline on each call to detect():
      1.  Project 3-D free / occupied voxels to a square 2-D top-down map
          (0 = free, 0.5 = unknown, 1 = occupied).
      2.  Run the 3-model LaMa ensemble → per-pixel predictive variance.
      3.  Detect frontier cells (free cells adjacent to unknown cells) and
          cluster them into regions.
      4.  Convert each region's centre to a 3-D Frontier object; gain is
          proportional to local LaMa variance.
    """

    FREE: float = 0.0
    UNK:  float = 0.5
    OCC:  float = 1.0

    def __init__(
        self,
        mapex_dir: str,
        device: str = "cuda",
        map_size: int = 512,        # side length of the top-down map (px); must be multiple of 16
        map_margin_m: float = 3.0,  # extra metres of unknown border beyond observed extent
        min_frontier_size: int = 10,
        gain_scale: float = 1e4,    # scale raw variance [0-0.083] → gain; floor is 2 × filter_min_gain
        log_level: int = logging.INFO,
    ):
        """
        Args:
            mapex_dir:         Root of the MapEx clone
                               (contains scripts/ and pretrained_models/).
            device:            "cuda" or "cpu".
            map_size:          Pixel size of the square top-down map fed to LaMa.
                               Must be a multiple of 16.  Default 512.
            map_margin_m:      Metres of unknown padding around the observed
                               bounding box.  Larger values let LaMa hallucinate
                               more of the unseen environment.
            min_frontier_size: Minimum pixel-cluster size to accept as a frontier.
            gain_scale:        Multiplier for the per-frontier LaMa variance before
                               it is stored as Frontier.gain.  Ensures gains pass
                               filter_min_gain (default 1 in the config).
            log_level:         Python logging level.
        """
        assert map_size % 16 == 0, f"map_size must be a multiple of 16, got {map_size}"

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.device = device
        self.map_size = map_size
        self.map_margin_m = map_margin_m
        self.min_frontier_size = min_frontier_size
        self.gain_scale = gain_scale

        # Make MapEx/scripts importable so lama_pred_utils resolves.
        scripts_dir = str(Path(mapex_dir) / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from lama_pred_utils import (
            load_lama_model,
            convert_obsimg_to_model_input,
            get_lama_transform,
        )
        self._convert_input = convert_obsimg_to_model_input

        weights_dir = Path(mapex_dir) / "pretrained_models" / "weights"
        ensemble_dirs = [
            str(weights_dir / "lama_ensemble" / f"train_{i}") for i in (1, 2, 3)
        ]

        self.logger.info("Loading 3 LaMa ensemble models from %s …", weights_dir)
        self._models = [load_lama_model(d, device=device) for d in ensemble_dirs]
        # default_map_eval: pads image to nearest multiple of 16, converts to float.
        # For map_size=512 (already ×16) no padding is needed.
        self._transform = get_lama_transform("default_map_eval", map_size)
        self.logger.info("MapExFrontierDetector ready (%d models).", len(self._models))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        free_pts: np.ndarray,
        occ_pts: Optional[np.ndarray],
        W_T_C: np.ndarray,
        z_filter: Optional[Tuple[float, float]] = None,
    ) -> List[Frontier]:
        """
        Detect frontiers from the current partial 3-D occupancy map.

        Args:
            free_pts:  (N, 3) float32 — observed free voxel centres (world frame).
            occ_pts:   (M, 3) float32 — observed occupied voxel centres (world frame).
            W_T_C:     (4, 4)         — camera pose in world frame (W_T_C[:3,3] = cam pos).
            z_filter:  (z_min, z_max) metres — height range to keep when projecting
                       3-D voxels to 2-D.  Filters out floor/ceiling clutter.
                       Pass None to include all heights.

        Returns:
            List of Frontier objects (may be empty).
        """
        if free_pts is None or len(free_pts) == 0:
            self.logger.warning("MapEx: no free points — skip detection.")
            return []

        cam_pos = W_T_C[:3, 3]

        # Step 1 — project 3-D voxels to 2-D occupancy map
        obs_map, origin_xy, m_per_px = self._project_to_2d(free_pts, occ_pts, z_filter)
        self.logger.info(
            "MapEx: 2D map %s  origin=(%.2f,%.2f)  m/px=%.4f  "
            "free=%.1f%%  unk=%.1f%%  occ=%.1f%%",
            obs_map.shape, origin_xy[0], origin_xy[1], m_per_px,
            100.0 * (obs_map == self.FREE).mean(),
            100.0 * (obs_map == self.UNK).mean(),
            100.0 * (obs_map == self.OCC).mean(),
        )

        # Step 2 — LaMa ensemble → per-pixel variance
        var_map = self._lama_variance(obs_map)

        # Step 3 — frontier pixel detection
        centers_px = self._frontier_pixels(obs_map)
        self.logger.info("MapEx: %d frontier regions detected.", len(centers_px))
        if not centers_px:
            return []

        # Step 4 — lift to 3-D Frontier objects
        frontiers = self._make_frontiers(centers_px, var_map, origin_xy, m_per_px, cam_pos)
        self.logger.info("MapEx: returning %d Frontier objects.", len(frontiers))
        return frontiers

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _project_to_2d(
        self,
        free_pts: np.ndarray,
        occ_pts: Optional[np.ndarray],
        z_filter: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Project 3-D voxel clouds to a square (map_size × map_size) occupancy map.

        Returns:
            obs_map:   (map_size, map_size) float32; 0=free, 0.5=unknown, 1=occupied.
            origin_xy: (2,) world XY of the top-left corner (x_min, y_min).
            m_per_px:  metres per pixel.
        """
        def _z_filter(pts: np.ndarray) -> np.ndarray:
            if pts is None or len(pts) == 0:
                return np.empty((0, 3), dtype=np.float32)
            if z_filter is None:
                return pts
            z_min, z_max = z_filter
            return pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)]

        fp = _z_filter(free_pts)
        op = _z_filter(occ_pts) if occ_pts is not None else np.empty((0, 3), dtype=np.float32)

        fp2 = fp[:, :2]
        op2 = op[:, :2]
        all2 = np.vstack([fp2, op2]) if len(op2) else fp2

        if len(all2) == 0:
            return (
                np.full((self.map_size, self.map_size), self.UNK, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                1.0,
            )

        # Square bounding box with margin
        cx = (all2[:, 0].min() + all2[:, 0].max()) / 2.0
        cy = (all2[:, 1].min() + all2[:, 1].max()) / 2.0
        half = max(
            (all2[:, 0].max() - all2[:, 0].min()) / 2.0,
            (all2[:, 1].max() - all2[:, 1].min()) / 2.0,
        ) + self.map_margin_m

        x_min = cx - half
        y_min = cy - half
        m_per_px = (2.0 * half) / self.map_size
        origin_xy = np.array([x_min, y_min], dtype=np.float64)

        obs_map = np.full((self.map_size, self.map_size), self.UNK, dtype=np.float32)

        def _w2px(xy: np.ndarray):
            col = np.clip(((xy[:, 0] - x_min) / m_per_px).astype(int), 0, self.map_size - 1)
            row = np.clip(((xy[:, 1] - y_min) / m_per_px).astype(int), 0, self.map_size - 1)
            return row, col

        if len(fp2) > 0:
            r, c = _w2px(fp2)
            obs_map[r, c] = self.FREE
        if len(op2) > 0:
            r, c = _w2px(op2)
            obs_map[r, c] = self.OCC

        return obs_map, origin_xy, m_per_px

    def _lama_variance(self, obs_map: np.ndarray) -> np.ndarray:
        """
        Run LaMa ensemble on obs_map; return per-pixel predictive variance.

        The variance across the 3 ensemble models is the information-gain proxy
        used by MapEx's 'visvarprob' scoring mode.
        """
        import torch

        # Grayscale map → 3-channel float image in [0, 1]
        obs_3ch = np.stack([obs_map, obs_map, obs_map], axis=2).astype(np.float32)

        preds = []
        with torch.no_grad():
            for model in self._models:
                inp, _ = self._convert_input(obs_3ch, self._transform, self.device)
                out = model(inp)
                # inpainted: (1, 3, H_out, W_out) — take mean across the 3 channels
                pred = out["inpainted"][0].mean(dim=0).cpu().numpy()   # (H_out, W_out)
                # Crop back to map_size in case the transform added padding
                pred = pred[: self.map_size, : self.map_size]
                preds.append(pred)

        preds_arr = np.stack(preds, axis=0)     # (3, map_size, map_size)
        var_map   = preds_arr.var(axis=0)       # (map_size, map_size)

        self.logger.debug(
            "LaMa var: min=%.5f  max=%.5f  mean=%.5f",
            var_map.min(), var_map.max(), var_map.mean(),
        )
        return var_map

    def _frontier_pixels(self, obs_map: np.ndarray) -> List[np.ndarray]:
        """
        Detect frontier regions: free cells adjacent to unknown cells.

        Returns list of (row, col) pixel centres of each valid region.
        """
        from scipy.ndimage import convolve, label, generate_binary_structure

        # 8-connected kernel: count unknown neighbours of each cell
        kernel = np.ones((3, 3), dtype=np.int32)
        kernel[1, 1] = 0

        adj_unk = convolve((obs_map == self.UNK).astype(np.int32), kernel) > 0
        edge_cells = adj_unk & (obs_map == self.FREE)

        structure = generate_binary_structure(2, 2)   # 8-connected
        regions, n_regions = label(edge_cells, structure)

        centers = []
        for i in range(1, n_regions + 1):
            pts = np.argwhere(regions == i)
            if len(pts) < self.min_frontier_size:
                continue
            centroid = pts.mean(axis=0)
            closest  = pts[np.linalg.norm(pts - centroid, axis=1).argmin()]
            centers.append(closest)   # (row, col) ndarray

        return centers

    def _make_frontiers(
        self,
        pixel_centers: List[np.ndarray],
        var_map: np.ndarray,
        origin_xy: np.ndarray,
        m_per_px: float,
        cam_pos: np.ndarray,
    ) -> List[Frontier]:
        """Convert 2-D pixel frontier centres to 3-D Frontier objects."""
        frontiers = []
        for rc in pixel_centers:
            row, col = int(rc[0]), int(rc[1])

            # Back-project pixel to world XY; use camera Z as height proxy
            world_x = origin_xy[0] + col * m_per_px
            world_y = origin_xy[1] + row * m_per_px
            world_z = float(cam_pos[2])
            world_pt = np.array([world_x, world_y, world_z], dtype=np.float64)

            # View direction: camera centre → frontier (unit vector)
            vd = world_pt - cam_pos
            vd_norm = float(np.linalg.norm(vd))
            if vd_norm < 1e-6:
                continue
            vd = vd / vd_norm

            # Gain: mean LaMa variance in a 3×3 window around the frontier pixel,
            # scaled so it comfortably exceeds filter_min_gain (default = 1).
            r0, r1 = max(0, row - 1), min(var_map.shape[0], row + 2)
            c0, c1 = max(0, col - 1), min(var_map.shape[1], col + 2)
            raw_var = float(var_map[r0:r1, c0:c1].mean())
            gain = max(raw_var * self.gain_scale, 2.0)

            f = Frontier()
            f.pos3d       = world_pt
            f.view_direction = vd
            f.gain        = gain
            f.u_gain      = gain
            f.direct_angle = float(np.arctan2(vd[1], vd[0]))
            f.pixel_pos   = np.array([col, row], dtype=np.float64)
            f.set_valid()
            frontiers.append(f)

        return frontiers
