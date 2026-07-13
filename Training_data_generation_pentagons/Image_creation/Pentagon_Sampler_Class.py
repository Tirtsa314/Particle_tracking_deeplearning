
import numpy as np
from Training_data_generation_pentagons.Image_creation.particle_position.cluster_position import cluster_positions



def convex_hull_2d(points):
    """
    Monotonic chain convex hull.
    points: (N,2) float
    returns hull as (M,2) in CCW order, without repeating the first point.
    """
    pts = np.asarray(points, float)
    # sort by x then y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

def outlines(
    cx, cy, rx, ry, rz,
    side_length_px, thickness_px,
):
    """
    Returns projected pentagonal-prism outline in image pixel coords (x,y).

    This matches the Pentagonal.get() convention:
    - regular pentagon
    - one vertex along +x
    - rotation matrix Rz @ Ry @ Rx
    - projection onto image xy plane
    """

    rotation = (rx, ry, rz)
    rx, ry, rz = rotation

    cxr, sxr = np.cos(rx), np.sin(rx)
    cyr, syr = np.cos(ry), np.sin(ry)
    czr, szr = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cxr, -sxr],
        [0, sxr,  cxr],
    ], dtype=float)

    Ry = np.array([
        [ cyr, 0, syr],
        [   0, 1,   0],
        [-syr, 0, cyr],
    ], dtype=float)

    Rz = np.array([
        [czr, -szr, 0],
        [szr,  czr, 0],
        [  0,    0, 1],
    ], dtype=float)

    # object frame -> image/world frame
    Rmat = Rz @ Ry @ Rx

    # -------------------------------
    # Regular pentagon object vertices
    # -------------------------------
    Rpent = side_length_px / (2 * np.sin(np.pi / 5))

    angles = 2 * np.pi * np.arange(5) / 5.0

    pent_xy = np.column_stack([
        Rpent * np.cos(angles),
        Rpent * np.sin(angles),
    ])

    zt = thickness_px / 2.0
    zb = -thickness_px / 2.0

    top_xyz = np.column_stack([
        pent_xy,
        np.full(5, zt),
    ])

    bot_xyz = np.column_stack([
        pent_xy,
        np.full(5, zb),
    ])

    V_xyz = np.vstack([top_xyz, bot_xyz])  # 10 vertices

    # Rotate vertices
    V = (Rmat @ V_xyz.T).T

    # Project to image xy
    P_xy = V[:, :2].copy()
    P_xy[:, 0] += cx
    P_xy[:, 1] += cy

    # Convex hull of projected 3D prism
    hull_xy = convex_hull_2d(P_xy)

    return hull_xy

class PentagonalPrismSampler:
    def __init__(self, H, W, margin, side_length_px, thickness_px,
                 rz_sigma_deg=20, rx_sigma_deg=2, ry_sigma_deg=2,
                 rot_sampler_fn=None, pos_sampler_fn=None, scale=2.0):
        self.H, self.W = H, W
        self.margin = margin
        self.side = float(side_length_px)
        self.thickness = float(thickness_px)

        self.rz_sigma = np.deg2rad(rz_sigma_deg)
        self.rx_sigma = np.deg2rad(rx_sigma_deg)
        self.ry_sigma = np.deg2rad(ry_sigma_deg)

        self.outlines_px = []
        self._last_position = None
        self._last_rotation = None

        # self._V0 = prism_vertices_3d(self.side, self.thickness)

        if rot_sampler_fn is None or pos_sampler_fn is None:
            rot_sampler_fn, pos_sampler_fn, _ = cluster_positions(scale=scale)
        self._rot_sampler_fn = rot_sampler_fn
        self._pos_sampler_fn = pos_sampler_fn

        # flag to ensure we don't resample twice per object
        self._fresh = False

    def reset(self):
        self.outlines_px.clear()
        self._fresh = False

    def _ensure_sampled(self):
        if self._fresh:
            return

        # sample new pose
        cx, cy = self._pos_sampler_fn()
        rx, ry, rz = self._rot_sampler_fn()

        side_length_px = self.side
        thickness_px   = self.thickness

        outline = outlines(
        cx, cy, rx, ry, rz,
        side_length_px, thickness_px,
    )
        self.outlines_px.append(outline)
        # self._last_position = (cy, cx)  # (y, x) for DeepTrack  

        self._last_position_xy = (cx, cy)   # for outlines / labels (x,y)
        self._last_position_yx = (cy, cx)   # for DeepTrack position (y,x)
        self._last_rotation = (rx, ry, rz)
        self._fresh = True
        # print("pos xy:", self._last_position_xy, "pos yx:", self._last_position_yx)
        # print("outline first point (x,y):", self.outlines_px[-1][0])

    def pos_sampler(self):
        self._ensure_sampled()
        return self._last_position_yx

    def rot_sampler(self):
        self._ensure_sampled()
        rot = self._last_rotation
        self._fresh = False     # next call gives a new sample
        return rot

    # optional: call this after an object is fully generated if needed
    def mark_consumed(self):
        self._fresh = False

#%%
# #testing outlines:
# # outlines = outlines(64, 64, 0, 0, 0, 30, 10)
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_outline(
#     cx, cy,
#     rx, ry, rz,
#     side_length_px,
#     thickness_px,
#     W=256, H=256
# ):
#     """
#     Uses your outlines(...) function and visualizes the result.
#     """

#     poly = outlines(
#         cx, cy,
#         rx, ry, rz,
#         side_length_px,
#         thickness_px
#     )

#     poly = np.asarray(poly, float)
#     poly_closed = np.vstack([poly, poly[0]])  # close the polygon

#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.set_xlim(0, W)
#     ax.set_ylim(H, 0)  # image coordinates (y down)
#     ax.set_aspect("equal")

#     ax.set_title(
#         f"pos=({cx},{cy})  rot=({rx:.2f},{ry:.2f},{rz:.2f})"
#     )

#     # Draw frame
#     ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], linewidth=1)

#     # Draw outline
#     ax.plot(poly_closed[:, 0], poly_closed[:, 1], linewidth=2)
#     ax.scatter(poly[:, 0], poly[:, 1], s=40)  # vertices
#     ax.scatter([cx], [cy], s=60)              # center

#     plt.show()
#%%
# plot_outline(s
#     cx=64,
#     cy=64,
#     rx=np.pi+np.pi/6,
#     ry=0.0,
#     rz=0.0,
#     side_length_px=30,
#     thickness_px=10,
#     W=128,
#     H=128
# )
#%%





#%%
#previous method:

# def rot_matrix(rx, ry, rz):
#     cx, sx = np.cos(rx), np.sin(rx)
#     cy, sy = np.cos(ry), np.sin(ry)
#     cz, sz = np.cos(rz), np.sin(rz)

#     Rx = np.array([[1, 0, 0],
#                    [0, cx, -sx],
#                    [0, sx,  cx]], dtype=float)
#     Ry = np.array([[ cy, 0, sy],
#                    [  0, 1,  0],
#                    [-sy, 0, cy]], dtype=float)
#     Rz = np.array([[cz, -sz, 0],
#                    [sz,  cz, 0],
#                    [ 0,   0, 1]], dtype=float)
#     return Rz @ Ry @ Rx


# def equilateral_triangle_xy(side):
#     h = np.sqrt(3) / 2 * side
#     tri = np.array([
#         [-side/2, -h/3],
#         [ side/2, -h/3],
#         [ 0.0,     2*h/3]
#     ], dtype=float)

#     # tri[:,1] *= -1  # flip y once for image coords FIX!!!
#     return tri

# def prism_vertices_3d(side, thickness):
#     tri = equilateral_triangle_xy(side)  # (3,2)
#     zt, zb = thickness/2.0, -thickness/2.0
#     top = np.c_[tri, np.full(3, zt)]
#     bot = np.c_[tri, np.full(3, zb)]
#     return np.vstack([top, bot])         # (6,3)













# class TrianglePrismSampler:
#     def __init__(self, H, W, margin, side_length_px, thickness_px,
#                  rz_sigma_deg=20, rx_sigma_deg=2, ry_sigma_deg=2):
#         self.H, self.W = H, W
#         self.margin = margin
#         self.side = float(side_length_px)
#         self.thickness = float(thickness_px)

#         self.rz_sigma = np.deg2rad(rz_sigma_deg)
#         self.rx_sigma = np.deg2rad(rx_sigma_deg)
#         self.ry_sigma = np.deg2rad(ry_sigma_deg)

#         self.outlines_px = []   # list of (M,2) arrays, CCW outline vertices
#         self._last_position = None
#         self._last_rotation = None

#         # precompute object-space prism vertices
#         self._V0 = prism_vertices_3d(self.side, self.thickness)  # (6,3)

#     def reset(self):
#         self.outlines_px.clear()

#     def sample(self):
#         # cy = np.random.uniform(self.margin, self.H - self.margin)
#         # cx = np.random.uniform(self.margin, self.W - self.margin)

#         # rx = np.random.randn() * self.rx_sigma
#         # ry = np.random.randn() * self.ry_sigma
#         # rz = np.random.randn() * self.rz_sigma

#         rot_sampler, pos_sampler = cluster_positions()
#         cx, cy = pos_sampler()
#         rx, ry, rz = rot_sampler()


#         # force 60 degrees HERE (before rot_matrix)
#         rz = np.pi/3

#         F = np.diag([1.0, -1.0, 1.0])
#         R = F @ rot_matrix(rx, ry, rz) @ F

#         Vr = (R @ self._V0.T).T
#         V2 = Vr[:, :2]

#         V2[:, 0] += cx
#         V2[:, 1] += cy

#         outline = convex_hull_2d(V2)
#         self.outlines_px.append(outline)

#         self._last_position = (cy, cx)
#         self._last_rotation = (rx, ry, rz)

#     def pos_sampler(self):
#         self.sample()
#         return self._last_position

#     def rot_sampler(self):
#         return self._last_rotation

# %%
