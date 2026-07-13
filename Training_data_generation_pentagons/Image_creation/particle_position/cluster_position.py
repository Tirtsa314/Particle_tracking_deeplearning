
import numpy as np
import math
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def resample_polyline(points, n_samples=250):
    """
    Resample a polyline to equally spaced points along arc length.
    points: (K,2)
    returns: (n_samples,2)
    """
    points = np.asarray(points, dtype=float)
    seg = np.diff(points, axis=0)
    seglen = np.sqrt((seg**2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(seglen)])

    if s[-1] == 0:
        return np.repeat(points[:1], n_samples, axis=0)

    s_new = np.linspace(0, s[-1], n_samples)
    x_new = np.interp(s_new, s, points[:, 0])
    y_new = np.interp(s_new, s, points[:, 1])
    return np.column_stack([x_new, y_new])


def smooth_curve(points, window=9, n_passes=2):
    """
    Simple moving-average smoothing.
    """
    pts = np.asarray(points, dtype=float).copy()
    if window < 3:
        return pts

    kernel = np.ones(window, dtype=float) / window

    for _ in range(n_passes):
        x = np.convolve(pts[:, 0], kernel, mode="same")
        y = np.convolve(pts[:, 1], kernel, mode="same")
        pts[:, 0] = x
        pts[:, 1] = y

    return pts


def make_random_filament_curve(
    W,
    H,
    rng,
    n_knots=7,
    step_mean=180,
    step_std=40,
    turn_sigma=0.7,
    margin=50,
    n_curve_samples=250,
):
    """
    Correlated random walk -> smooth filament-like curve.
    """
    # random start
    p = np.array([
        rng.uniform(margin, W - margin),
        rng.uniform(margin, H - margin)
    ], dtype=float)

    angle = rng.uniform(0, 2 * np.pi)

    pts = [p.copy()]

    for _ in range(n_knots - 1):
        # gradual turning, not fully random direction
        angle += rng.normal(0, turn_sigma)

        step = max(20, rng.normal(step_mean, step_std))
        p = p + step * np.array([np.cos(angle), np.sin(angle)])

        # keep inside image
        p[0] = np.clip(p[0], margin, W - margin)
        p[1] = np.clip(p[1], margin, H - margin)

        pts.append(p.copy())

    pts = np.asarray(pts)

    # resample + smooth
    curve = resample_polyline(pts, n_samples=n_curve_samples)
    curve = smooth_curve(curve, window=11, n_passes=2)

    return curve

def curve_bend_strength(curve, smooth_passes=2):
    """
    curve: (M,2) array of sampled curve points
    returns bend strength in [0,1] for each curve sample
    """
    if len(curve) < 3:
        return np.zeros(len(curve), dtype=float)

    v1 = curve[1:-1] - curve[:-2]
    v2 = curve[2:]   - curve[1:-1]

    n1 = np.linalg.norm(v1, axis=1) + 1e-12
    n2 = np.linalg.norm(v2, axis=1) + 1e-12

    cosang = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)

    # angle between consecutive segments
    ang = np.arccos(cosang)   # 0 = straight, larger = more bend

    # pad so output has same length as curve
    bend = np.pad(ang, (1, 1), mode="edge")

    # optional smoothing
    kernel = np.array([1, 2, 1], dtype=float)
    kernel /= kernel.sum()
    for _ in range(smooth_passes):
        bend = np.convolve(bend, kernel, mode="same")

    # normalize to [0,1]
    bmin, bmax = bend.min(), bend.max()
    if bmax > bmin:
        bend = (bend - bmin) / (bmax - bmin)
    else:
        bend = np.zeros_like(bend)

    return bend















def sample_bezier(P0, P1, P2, P3, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * P0 + 3 * ((1 - t) ** 2) * t * P1 + 3 * (1 - t) * (t ** 2) * P2 + (t ** 3) * P3


# def make_grid_sites(W, H, min_dist, margin=17, jitter_frac=0.0, rng=None):
#     """
#     Create grid sites that guarantee a minimum distance.

#     If jitter_frac > 0, the grid spacing is enlarged so that even after jitter
#     the minimum distance is still at least min_dist.
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     if not (0.0 <= jitter_frac < 0.5):
#         raise ValueError("jitter_frac must satisfy 0 <= jitter_frac < 0.5")

#     if jitter_frac == 0:
#         cell = float(min_dist)
#         jitter = 0.0
#     else:
#         cell = float(min_dist) / (1.0 - 2.0 * jitter_frac)
#         jitter = jitter_frac * cell

#     x0 = margin + jitter
#     x1 = W - margin - jitter
#     y0 = margin + jitter
#     y1 = H - margin - jitter

#     xs = np.arange(x0, x1 + 1e-9, cell)
#     ys = np.arange(y0, y1 + 1e-9, cell)

#     XX, YY = np.meshgrid(xs, ys, indexing="xy")
#     sites = np.column_stack([XX.ravel(), YY.ravel()])

#     if jitter > 0:
#         sites += rng.uniform(-jitter, jitter, size=sites.shape)

#     return sites

def make_grid_sites(W, H, min_dist, margin=17, jitter_frac=0.0, rng=None):
    """
    Create hexagonal / staggered grid sites.

    This gives a honeycomb-like arrangement of particle centers:
    every other row is shifted by half a lattice spacing.

    min_dist is approximately the nearest-neighbor center distance.
    If jitter_frac > 0, the lattice spacing is enlarged so that
    the minimum distance is still roughly protected.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 <= jitter_frac < 0.5):
        raise ValueError("jitter_frac must satisfy 0 <= jitter_frac < 0.5")

    # Enlarge spacing to leave room for jitter
    if jitter_frac == 0:
        a = float(min_dist)
        jitter_radius = 0.0
    else:
        a = float(min_dist) / (1.0 - 2.0 * jitter_frac)
        jitter_radius = jitter_frac * a

    # Hexagonal row spacing
    dx = a
    dy = np.sqrt(3) / 2 * a

    x_min = margin + jitter_radius
    x_max = W - margin - jitter_radius
    y_min = margin + jitter_radius
    y_max = H - margin - jitter_radius

    sites = []

    y = y_min
    row = 0

    while y <= y_max:
        # shift every other row
        x_offset = 0.5 * dx if row % 2 == 1 else 0.0

        x = x_min + x_offset
        while x <= x_max:
            sites.append([x, y])
            x += dx

        y += dy
        row += 1

    sites = np.asarray(sites, dtype=float)

    if jitter_radius > 0 and len(sites) > 0:
        # jitter in a disk, so max displacement is controlled
        angles = rng.uniform(0, 2 * np.pi, size=len(sites))
        radii = jitter_radius * np.sqrt(rng.uniform(0, 1, size=len(sites)))

        sites[:, 0] += radii * np.cos(angles)
        sites[:, 1] += radii * np.sin(angles)

    return sites

def _choose_from_free(free_mask, weights, k, rng):
    """
    Choose up to k free indices, optionally weighted.
    """
    free_idx = np.where(free_mask)[0]
    if len(free_idx) == 0 or k <= 0:
        return np.array([], dtype=int)

    k = min(k, len(free_idx))

    w = np.asarray(weights, dtype=float)[free_idx]
    w = np.clip(w, 0, None)

    if np.all(w == 0) or not np.isfinite(w).all():
        chosen = rng.choice(free_idx, size=k, replace=False)
    else:
        p = w / w.sum()
        chosen = rng.choice(free_idx, size=k, replace=False, p=p)

    return chosen


def add_background_from_grid(sites, free_mask, n_bg, rng):
    chosen = _choose_from_free(free_mask, np.ones(len(sites)), n_bg, rng)
    free_mask[chosen] = False
    return sites[chosen]


def add_blob_from_grid(sites, free_mask, pts_per_blob, spread, W, H, rng):
    """
    Pick grid sites around a random blob center with Gaussian weighting.
    """
    cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    d2 = (sites[:, 0] - cx) ** 2 + (sites[:, 1] - cy) ** 2
    weights = np.exp(-0.5 * d2 / (spread ** 2))
    chosen = _choose_from_free(free_mask, weights, pts_per_blob, rng)
    free_mask[chosen] = False
    return sites[chosen]



def add_filament_from_grid(
    sites,
    free_mask,
    pts_per_fil,
    thickness_base,
    thickness_bend_gain,
    W,
    H,
    rng,
    n_curve_samples=250,
):
    """
    Filament whose thickness increases at bends.
    
    thickness_base: minimum thickness on straighter parts
    thickness_bend_gain: extra thickness added at strongest bends
    """
    P0 = rng.random((1, 2)) * [W, H]
    P3 = rng.random((1, 2)) * [W, H]
    P1 = P0 + (rng.normal(size=(1, 2)) * 0.2 + [0.3, 0.3]) * [W, H]
    P2 = P3 + (rng.normal(size=(1, 2)) * 0.2 - [0.3, 0.3]) * [W, H]

    curve = sample_bezier(P0, P1, P2, P3, n_curve_samples)   # (M,2)
    bend = curve_bend_strength(curve)                         # (M,)

    free_idx = np.where(free_mask)[0]
    if len(free_idx) == 0:
        return np.empty((0, 2))

    free_sites = sites[free_idx]   # (N,2)

    # distance from each free site to every curve sample
    diff = free_sites[:, None, :] - curve[None, :, :]
    d2_all = np.sum(diff * diff, axis=2)                     # (N,M)

    # nearest curve sample for each site
    nearest_j = np.argmin(d2_all, axis=1)                    # (N,)
    d2_min = d2_all[np.arange(len(free_sites)), nearest_j]   # (N,)

    # local thickness: bigger at bends
    sigma_local = thickness_base + thickness_bend_gain * bend[nearest_j]

    # width weighting
    weights_local = np.exp(-0.5 * d2_min / (sigma_local ** 2))

    # optional: also slightly prefer bends even aside from width
    # weights_local *= (1.0 + 0.5 * bend[nearest_j])

    weights = np.zeros(len(sites), dtype=float)
    weights[free_idx] = weights_local

    chosen = _choose_from_free(free_mask, weights, pts_per_fil, rng)
    free_mask[chosen] = False
    return sites[chosen]


def generate_positions_grid(
    W=2048,
    H=2048,
    n_bg=1,
    n_blobs=1,
    pts_per_blob=1,
    blob_spread=25,
    n_filaments=1,
    pts_per_fil=1,
    # fil_thickness=15,
    thickness_base=8,
    thickness_bend_gain=14,
    min_dist=10,
    margin=17,
    jitter_frac=0.0,
    seed=None,
):
    """
    Grid-based clustered position generator.

    Minimum distance is enforced by construction.
    """
    rng = np.random.default_rng(seed)

    sites = make_grid_sites(
        W=W,
        H=H,
        min_dist=min_dist,
        margin=margin,
        jitter_frac=jitter_frac,
        rng=rng,
    )

    free_mask = np.ones(len(sites), dtype=bool)
    chunks = []

    # background
    if n_bg > 0:
        chunks.append(add_background_from_grid(sites, free_mask, n_bg, rng))

    # blobs
    for _ in range(n_blobs):
        chunks.append(
            add_blob_from_grid(
                sites, free_mask,
                pts_per_blob=pts_per_blob,
                spread=blob_spread,
                W=W, H=H,
                rng=rng
            )
        )

    # # filaments
    # for _ in range(n_filaments):
    #     chunks.append(
    #         add_filament_from_grid(
    #             sites, free_mask,
    #             pts_per_fil=pts_per_fil,
    #             thickness=fil_thickness,
    #             W=W, H=H,
    #             rng=rng
    #         )
    #     )
    for _ in range(n_filaments):
        chunks.append(
        add_filament_from_grid(
            sites, free_mask,
            pts_per_fil=pts_per_fil,
            thickness_base=thickness_base,
            thickness_bend_gain=thickness_bend_gain,
            W=W, H=H,
            rng=rng
            )
        )   

    if len(chunks) == 0:
        return np.empty((0, 2))

    positions = np.vstack([c for c in chunks if len(c) > 0])

    # shuffle final order
    if len(positions) > 0:
        positions = positions[rng.permutation(len(positions))]

    return positions




def make_position_sampler(pos_array):
    i = {"k": 0}
    def sampler():
        k = i["k"]
        i["k"] = k + 1
        return tuple(pos_array[k % len(pos_array)])
    return sampler



import math
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def regular_polygon_object(n_sides=5, side_px=30.0, theta0=0.0):
    """
    Regular polygon centered at (0,0).

    side_px is the side length in pixels.
    """
    R = side_px / (2 * np.sin(np.pi / n_sides))  # center -> vertex distance

    angles = theta0 + 2 * np.pi * np.arange(n_sides) / n_sides

    verts = np.column_stack([
        R * np.cos(angles),
        R * np.sin(angles),
    ])

    return Polygon(verts)


def polygon_placement(x: float, y: float, theta: float, base_poly: Polygon) -> Polygon:
    p = rotate(base_poly, theta, origin=(0, 0), use_radians=True)
    return translate(p, xoff=x, yoff=y)


def overlap_polygon(i_poly: Polygon, neighbor_polys: list[Polygon]) -> float:
    cost = 0.0
    for pj in neighbor_polys:
        inter = i_poly.intersection(pj)
        if not inter.is_empty:
            cost += inter.area
    return cost

def build_neighbor_list(pos: np.ndarray, cutoff: float = 35.0) -> list[list[int]]:
    neigh = []
    cutoff2 = cutoff * cutoff
    for i in range(len(pos)):
        d = pos - pos[i]
        dist2 = np.sum(d*d, axis=1)
        idx = np.where((dist2 > 0) & (dist2 < cutoff2))[0].tolist()
        neigh.append(idx)
    return neigh

def optimize_angles_sequential(
    positions,
    rotations,
    N,
    n_iter=1,
    cutoff=None,
    n_sides=5,
    side_px=None,
    n_candidates=50,
):
    """
    Sequential greedy in-plane angle optimization for regular polygons.

    For pentagons, the overlap geometry repeats every 72 degrees,
    so we only need to search 0 to 2*pi/5.
    """
    thetas = np.asarray(rotations, dtype=float).copy()

    base_poly = regular_polygon_object(
        n_sides=n_sides,
        side_px=side_px,
    )

    if cutoff is None:
        R = side_px / (2 * np.sin(np.pi / n_sides))
        cutoff = 2 * R + 5.0 

    # Symmetry period:
    # triangle -> 120 degrees
    # pentagon -> 72 degrees
    period = 2 * np.pi / n_sides
    candidates = np.linspace(0, period, n_candidates, endpoint=False) #angles to try

    polys = [ #list of shapely polygon objects with current pos and rot
        polygon_placement(positions[i, 0], positions[i, 1], thetas[i], base_poly)
        for i in range(N)
    ]

    cutoff2 = cutoff * cutoff #squared since faster with distance squared instead of taking sqrt

    for _ in range(n_iter): #looping over optimization passes
        for i in range(N): #looping over each particle
            d = positions - positions[i] #vector from current particle to all other particles
            dist2 = np.sum(d * d, axis=1) #squared distance to all other particles, shape (N-1,)
            neighbour_index = np.where((dist2 > 0) & (dist2 < cutoff2))[0].tolist() #shape (M,) indices of neighbors within cutoff distance

            if not neighbour_index:
                continue

            neighbor_polys = [polys[j] for j in neighbour_index] #list of shapely polygon objects for neighbors

            best_theta = thetas[i] #current angle is the best so far
            best_poly = polys[i] #current polygon is the best so far
            best_cost = overlap_polygon(best_poly, neighbor_polys) #current overlap cost is the best so far

            for angle in candidates: 
                pi = polygon_placement( #create test polygon for every particle i, shapely object
                    positions[i, 0],
                    positions[i, 1],
                    angle,
                    base_poly,
                )
                c = overlap_polygon(pi, neighbor_polys) #cost of overlapping polygons

                if c < best_cost: #looks for lowest cost
                    best_cost = c
                    best_theta = angle
                    best_poly = pi

            thetas[i] = best_theta #update angle particle i
            polys[i] = best_poly #update polygon particle i

    return thetas


def make_rotation_sampler(rx_arr, ry_arr, rz_arr):
    i = {"k": 0}
    N = len(rz_arr)

    def sampler():
        k = i["k"]
        i["k"] = k + 1
        return (
            float(rx_arr[k % N]),
            float(ry_arr[k % N]),
            float(rz_arr[k % N]),
        )
    return sampler

def cluster_positions(scale, side_um, um_per_pixel):
    N_sides = 5
    UM_PER_PIXEL = um_per_pixel
    SIDE_UM = side_um
    SIDE_PX = scale * SIDE_UM / UM_PER_PIXEL
    PENTAGON_R_PX = SIDE_PX / (2 * np.sin(np.pi / N_sides))
    PENTAGON_INNER_R_PX = SIDE_PX / (2 * np.tan(np.pi / N_sides))
    n_bg = np.random.randint(20, 70)
    n_blobs = np.random.randint(1, 3)
    pts_per_blob = np.random.randint(10, 60)

    n_filaments = 1
    pts_per_fil = np.random.randint(20, 40)

    print("n_bg =", n_bg)
    print("n_blobs =", n_blobs, "pts_per_blob =", pts_per_blob)
    print("n_filaments =", n_filaments, "pts_per_fil =", pts_per_fil)
    print("requested total ≈", n_bg + n_blobs * pts_per_blob + n_filaments * pts_per_fil)

    positions = generate_positions_grid(
        W=512*scale,
        H=512*scale,

        n_bg=n_bg,
        n_blobs=n_blobs,
        pts_per_blob=pts_per_blob,
        blob_spread=20 * scale,

        n_filaments=n_filaments,
        pts_per_fil=pts_per_fil,
        thickness_base=5 * scale,
        thickness_bend_gain=30 * scale,

        # Important for pentagons:
        # side = 21.9 px, apothem = 15.1 px, circumradius = 18.6 px
        # min_dist=16 is probably too small for pentagons unless you want strong overlap.
        min_dist=PENTAGON_INNER_R_PX * 2,

        # margin should be larger than the pentagon circumradius
        margin=int(np.ceil(PENTAGON_R_PX + 3)) * scale,

        jitter_frac=0.1,
    )

    print("actual generated =", len(positions))

    positions = np.asarray(positions)
    N = len(positions)

    # Small out-of-plane tilts
    rx_arr = np.random.randn(N) * np.deg2rad(10)
    ry_arr = np.random.randn(N) * np.deg2rad(10)

    # Initial in-plane pentagon rotation.
    # A regular pentagon repeats every 72 degrees.
    rz0 = np.random.uniform(0, 2 * np.pi / N_sides, size=N)

    rz_arr = optimize_angles_sequential(
        positions,
        rz0,
        N,
        n_iter=5,
        cutoff=2 * PENTAGON_R_PX + 5,
        n_sides=N_sides,
        side_px=SIDE_PX,
        n_candidates=50,
    )

    # For a normal DeepTrack rotation=(rx, ry, rz),
    # the optimized in-plane angle should go into rz.
    rot_sampler = make_rotation_sampler(rx_arr, ry_arr, rz_arr)
    pos_sampler = make_position_sampler(positions)

    return rot_sampler, pos_sampler, N







# def cluster_positions(
#     angle_deg=0,
#     axis="rz",
#     W=512,
#     H=512, scale=2.0
# ):
#     """
#     Debug version: return exactly one triangle at the image center.

#     Parameters
#     ----------
#     angle_deg : float
#         Rotation angle in degrees.
#     axis : str
#         Which rotation component gets the angle: "rx", "ry", or "rz".
#     W, H : int
#         Image size in pixels.
#     scale : float
#         Scaling factor for the image.

#     Returns
#     -------
#     rot_sampler, pos_sampler, N
#     """

#     # one particle in the center
#     positions = np.array([[W * scale / 2, H * scale / 2]], dtype=float)
#     N = 1

#     angle_rad = np.deg2rad(angle_deg)

#     # all zero first
#     rx_arr = np.zeros(N)
#     ry_arr = np.full(N, np.deg2rad(10))
#     rz_arr = np.full(N, np.deg2rad(10))


#     rot_sampler = make_rotation_sampler(rx_arr, ry_arr, rz_arr)
#     pos_sampler = make_position_sampler(positions)


#     return rot_sampler, pos_sampler, N



































































# #%%
# """Simulating how deeptrack would plot:"""

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon
# from matplotlib.collections import PatchCollection

# def triangle_vertices(side_px=30):
#     """Equilateral triangle centered at (0,0), pointing up."""
#     h = np.sqrt(3) / 2 * side_px
#     return np.array([
#         (-side_px/2, -h/3),
#         ( side_px/2, -h/3),
#         (0, 2*h/3)
#     ], dtype=float)

# def rotate2d(pts, theta):
#     c, s = np.cos(theta), np.sin(theta)
#     R = np.array([[c, -s],
#                   [s,  c]])
#     return pts @ R.T

# def plot_triangles(positions, thetas=None, side_px=30, W=2048, H=2048,
#                    filled=False, alpha=0.9, linewidth=0.8, s=1.0):
#     """
#     positions: (N,2) array in pixel coords (x,y)
#     thetas:    (N,) radians. If None => all 0
#     """
#     positions = np.asarray(positions, float)
#     N = len(positions)
#     if thetas is None:
#         thetas = np.zeros(N, float)
#     else:
#         thetas = np.asarray(thetas, float)
#         if thetas.shape[0] != N:
#             raise ValueError("thetas must have same length as positions")

#     base = triangle_vertices(side_px=side_px)

#     patches = []
#     for (x, y), th in zip(positions, thetas):
#         pts = rotate2d(base, th) * s
#         pts[:, 0] += x
#         pts[:, 1] += y
#         patches.append(MplPolygon(pts, closed=True))

#     fig, ax = plt.subplots(figsize=(7,7))
#     ax.set_xlim(0, W)
#     ax.set_ylim(H, 0)  # invert Y to match image coordinates
#     ax.set_aspect("equal")
#     ax.set_title(f"{N} triangles (side={side_px}px)")

#     if filled:
#         coll = PatchCollection(patches, match_original=True, alpha=alpha, linewidth=linewidth)
#         ax.add_collection(coll)
#     else:
#         for p in patches:
#             p.set_fill(False)
#             p.set_linewidth(linewidth)
#             p.set_alpha(alpha)
#             ax.add_patch(p)

#     plt.show()




# #%%
# # Example usage:
# #plot_triangles(positions, thetas, side_px=30, W=2048, H=2048, filled=False)
# """ """
# #%%
# """Also simulating with deeptrack:"""
# import matplotlib.pyplot as plt
# import trackpy as tp
# from Training_data_generation.Image_creation.Image_generator import generate_image
# from Training_data_generation.Image_creation.Triangle_Sampler_Class import TrianglePrismSampler

# #%%

# positions = generate_positions(
# W=2048, H=2048,
# n_bg=400,
# n_blobs=6, pts_per_blob=120, blob_spread=90,
# n_filaments=2, pts_per_fil=400, fil_thickness=25, min_dist=15,
# )

# positions = np.asarray(positions)   # shape (N, 2)
# N = len(positions)
# # Make per-particle rx, ry, rz0 arrays (same distribution as rotation_xy_more)
# rx0 = np.random.randn(N) * np.deg2rad(40)
# ry0 = np.random.randn(N) * np.deg2rad(40)
# rz0 = np.random.randn(N) * np.deg2rad(40)
# amount_particles = len(positions)
# thetas_z = optimize_angles_sequential(positions, rz0 , amount_particles, n_iter=3, cutoff=35.0)
# rot_sampler = make_rotation_sampler(rx0, ry0, thetas_z)
# pos_sampler = make_position_sampler(positions)

# sampler = TrianglePrismSampler(
#     H=2048,
#     W=2048,
#     margin=50,
#     side_length_px=30,
#     thickness_px=8,
# )

# image_of_particles = generate_image(
#     sampler,
#     pos_sampler,
#     rot_sampler,
#     len(positions),
#     um_per_pixel=0.32,
#     image_size=2048,
#     side_length_um=10.0,
#     thickness_um=2.5,
#     noise_sigma=0.048,
# )
# sampler.reset()

# img = image_of_particles.update().resolve()
# img_u8 = (img * 255).astype(np.uint8)

# #%%

# plt.figure()
# vmin, vmax = np.percentile(img, (1, 99))
# plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
# plt.axis("off")
# plt.show()
# #%%
# RUN_DEEPTRACK = True  # set True only when you want the slow image

# print("N positions:", len(positions))
# print("N thetas:", len(thetas))

# plot_triangles(positions, thetas, side_px=30, W=2048, H=2048,
#               filled=False, alpha=1.0, linewidth=1.2)

# if RUN_DEEPTRACK:
#     rot_sampler = make_rotation_sampler(thetas)
#     pos_sampler = make_position_sampler(positions)

#     image_of_particles = generate_image(
#         sampler=sampler,
#         pos_sampler=pos_sampler,
#         rot_sampler=rot_sampler,
#     )
#     sampler.reset()

#     img = image_of_particles.update().resolve()
#     img_u8 = (img * 255).astype(np.uint8)

#     plt.figure(figsize=(7,7))
#     plt.imshow(img_u8, cmap="gray")
#     plt.axis("off")
#     plt.show()
# # # %%
