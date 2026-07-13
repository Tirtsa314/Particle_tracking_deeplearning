""" 
author: Tirtsa den Haan 
06-07-2026
"""
import numpy as np
from Training_data_generation_triangles_structured.Image_creation.particle_position.cluster_position import cluster_positions
from Training_data_generation_triangles_structured.Rotations import rotation_matrix_xyz
from scipy.spatial import ConvexHull


def convex_hull_2d(points):
    pts = np.asarray(points, float)
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def outlines(
    cx, cy, rx, ry, rz,
    side_length_px, thickness_px,
):
    """
    Returns projected triangular-prism outline in image pixel coords (x,y).

    Convention:
    - cx, cy are image coordinates: x right, y down
    - rx is rotation around x
    - ry is rotation around y
    - rz is rotation around z, the in-plane angle
    - output is outline_xy with columns [x, y]
    """

    Rmat = rotation_matrix_xyz(rx, ry, rz)

    Rtri = side_length_px / np.sqrt(3.0)

    zt = thickness_px / 2.0
    zb = -thickness_px / 2.0

    # Triangle in object coordinates.
    # One vertex points along +x when rz = 0.
    tri_xy = np.array([
        [ Rtri, 0.0],
        [-Rtri / 2,  np.sqrt(3) * Rtri / 2],
        [-Rtri / 2, -np.sqrt(3) * Rtri / 2],
    ], dtype=float)

    top_xyz = np.column_stack([tri_xy, np.full(3, zt)])
    bot_xyz = np.column_stack([tri_xy, np.full(3, zb)])
    V_xyz = np.vstack([top_xyz, bot_xyz])

    # object -> image/world
    V = (Rmat @ V_xyz.T).T

    P_xy = V[:, :2].copy()
    P_xy[:, 0] += cx
    P_xy[:, 1] += cy

    hull_xy = convex_hull_2d(P_xy)

    return hull_xy


class TrianglePrismSampler:
    def __init__(self, H, W, margin, side_length_px, thickness_px,
                 rot_sampler_fn=None, pos_sampler_fn=None, scale=2.0):
        self.H, self.W = H, W
        self.margin = margin
        self.side = float(side_length_px)
        self.thickness = float(thickness_px)

        self.outlines_px = []

        if rot_sampler_fn is None or pos_sampler_fn is None:
            rot_sampler_fn, pos_sampler_fn = cluster_positions(scale=scale)

        self._rot_sampler_fn = rot_sampler_fn
        self._pos_sampler_fn = pos_sampler_fn

        self._has_pose = False
        self._used_pos = False
        self._used_rot = False

    def reset(self):
        self.outlines_px.clear()
        self._has_pose = False
        self._used_pos = False
        self._used_rot = False

    def _sample_pose(self):
        cx, cy = self._pos_sampler_fn()
        rx, ry, rz = self._rot_sampler_fn()


        outline = outlines(
            cx, cy, rx, ry, rz,
            self.side, self.thickness,
        )

        self.outlines_px.append(outline)

        self._last_position_xy = (cx, cy)
        self._last_position_yx = (cy, cx)
        self._last_rotation = (rx, ry, rz)

        self._has_pose = True
        self._used_pos = False
        self._used_rot = False

    def _ensure_sampled(self):
        if not self._has_pose:
            self._sample_pose()

    def _clear_if_done(self):
        if self._used_pos and self._used_rot:
            self._has_pose = False
            self._used_pos = False
            self._used_rot = False

    def pos_sampler(self):
        self._ensure_sampled()
        pos = self._last_position_yx
        self._used_pos = True
        self._clear_if_done()
        return pos

    def rot_sampler(self):
        self._ensure_sampled()
        rot = self._last_rotation
        self._used_rot = True
        self._clear_if_done()
        return rot

