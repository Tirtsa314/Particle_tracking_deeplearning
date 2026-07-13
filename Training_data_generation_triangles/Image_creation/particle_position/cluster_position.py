#%%
import numpy as np
import math
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


# def sample_bezier(P0, P1, P2, P3, n):
#     t = np.random.rand(n, 1)
#     return ((1-t)**3)*P0 + 3*((1-t)**2)*t*P1 + 3*(1-t)*(t**2)*P2 + (t**3)*P3

# def filament_points(W, H, n_points=300, thickness=25):
#     # random cubic bezier control points
#     P0 = np.random.rand(1,2) * [W,H]
#     P3 = np.random.rand(1,2) * [W,H]
#     P1 = P0 + (np.random.randn(1,2) * 0.2 + [0.3,0.3]) * [W,H]
#     P2 = P3 + (np.random.randn(1,2) * 0.2 - [0.3,0.3]) * [W,H]

#     pts = sample_bezier(P0, P1, P2, P3, n_points)

#     # jitter (thickness)
#     pts += np.random.randn(n_points,2) * thickness
#     return pts

# def blob_points(W, H, n_clusters=8, pts_per=60, spread=60):
#     centers = np.random.rand(n_clusters,2) * [W,H]
#     pts = []
#     for c in centers:
#         pts.append(c + np.random.randn(pts_per,2) * spread)
#     return np.vstack(pts)

# """


# Grid things


# """


# def grid(W, H, min_dist):

#     x0 = 0
#     x1 = W 
#     y0 = 0
#     y1 = H 

#     xs = np.arange(x0, x1 + 1e-9, min_dist)
#     ys = np.arange(y0, y1 + 1e-9, min_dist)

#     XX, YY = np.meshgrid(xs, ys, indexing="xy")
#     sites = np.column_stack([XX.ravel(), YY.ravel()])

#     return sites


# def add_background_from_grid(sites, free_mask, n_bg, rng):
#     chosen = _choose_from_free(free_mask, np.ones(len(sites)), n_bg, rng)
#     free_mask[chosen] = False
#     return sites[chosen]


# def add_blob_from_grid(sites, free_mask, pts_per_blob, spread, W, H, rng):
#     """
#     Pick grid sites around a random blob center with Gaussian weighting.
#     """
#     cx, cy = rng.uniform(0, W), rng.uniform(0, H)
#     d2 = (sites[:, 0] - cx) ** 2 + (sites[:, 1] - cy) ** 2
#     weights = np.exp(-0.5 * d2 / (spread ** 2))
#     chosen = _choose_from_free(free_mask, weights, pts_per_blob, rng)
#     free_mask[chosen] = False
#     return sites[chosen]


# def add_filament_from_grid(sites, free_mask, pts_per_fil, thickness, W, H, rng, n_curve_samples=250):
#     """
#     Pick grid sites near a random cubic Bézier curve.
#     """
#     P0 = rng.random((1, 2)) * [W, H]
#     P3 = rng.random((1, 2)) * [W, H]
#     P1 = P0 + (rng.normal(size=(1, 2)) * 0.2 + [0.3, 0.3]) * [W, H]
#     P2 = P3 + (rng.normal(size=(1, 2)) * 0.2 - [0.3, 0.3]) * [W, H]

#     curve = sample_bezier(P0, P1, P2, P3, n_curve_samples)  # (M,2)

#     free_idx = np.where(free_mask)[0]
#     if len(free_idx) == 0:
#         return np.empty((0, 2))

#     free_sites = sites[free_idx]  # (N,2)

#     # distance from each free grid site to nearest curve sample
#     diff = free_sites[:, None, :] - curve[None, :, :]
#     d2_min = np.min(np.sum(diff * diff, axis=2), axis=1)

#     weights_local = np.exp(-0.5 * d2_min / (thickness ** 2))
#     weights = np.zeros(len(sites), dtype=float)
#     weights[free_idx] = weights_local

#     chosen = _choose_from_free(free_mask, weights, pts_per_fil, rng)
#     free_mask[chosen] = False
#     return sites[chosen]


# def generate_positions(W=2048, H=2048,
#                        n_bg=1,#n_bg=200
#                        n_blobs=1, pts_per_blob=1, blob_spread=1,#n_blobs=8, pts_per_blob=60, blob_spread=60,
#                        n_filaments=1, pts_per_fil=1, fil_thickness=1,
#                        min_dist=None):

#     bg = np.random.rand(n_bg,2) * [W,H]
#     blobs = blob_points(W,H,n_blobs,pts_per_blob,blob_spread)

#     fils = []
#     for _ in range(n_filaments):
#         fils.append(filament_points(W,H,pts_per_fil,fil_thickness))
#     fils = np.vstack(fils) if fils else np.empty((0,2))

#     positions = np.vstack([bg, blobs, fils])

#     # clip to bounds
#     positions = positions[
#         (positions[:,0] >= 17) & (positions[:,0] < (W-17)) &
#         (positions[:,1] >= 17) & (positions[:,1] < (H-17))
#     ]

#     return positions


# # def enforce_min_distance(positions, min_dist):
# #     accepted = []
# #     min2 = float(min_dist)**2

# #     for p in positions:
# #         if not accepted:
# #             accepted.append(p)
# #             continue

# #         d = np.asarray(accepted) - p
# #         if np.all(np.sum(d*d, axis=1) >= min2):
# #             accepted.append(p)

# #     return np.asarray(accepted)




# def make_position_sampler(pos_array):
#     i = {"k": 0}
#     def sampler():
#         k = i["k"]
#         i["k"] = k + 1
#         return tuple(pos_array[k % len(pos_array)])
#     return sampler



# def euclidean_distance(p1, p2):
#     return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# def triangle_placement(x: float, y: float, theta: float, base_tri: Polygon) -> Polygon:
#     p = rotate(base_tri, theta, origin=(0, 0), use_radians=True)
#     return translate(p, xoff=x, yoff=y)

# def triangle_object(side_px=30):
#     R = side_px / np.sqrt(3)

#     return Polygon([
#         ( R, 0.0),
#         (-R/2,  np.sqrt(3)*R/2),
#         (-R/2, -np.sqrt(3)*R/2),
#     ])

# def overlap_triangle(i_poly: Polygon, neighbor_polys: list[Polygon]) -> float:
#     cost = 0.0
#     for pj in neighbor_polys:
#         inter = i_poly.intersection(pj)
#         if not inter.is_empty:
#             cost += inter.area
#     return cost

# def build_neighbor_list(pos: np.ndarray, cutoff: float = 35.0) -> list[list[int]]:
#     neigh = []
#     cutoff2 = cutoff * cutoff
#     for i in range(len(pos)):
#         d = pos - pos[i]
#         dist2 = np.sum(d*d, axis=1)
#         idx = np.where((dist2 > 0) & (dist2 < cutoff2))[0].tolist()
#         neigh.append(idx)
#     return neigh

# def optimize_angles_sequential(positions, rotations, N, n_iter=1, cutoff=35.0):
#     """
#     Sequential greedy optimization:
#     when optimizing particle i, only consider neighbors with index < i
#     (i.e., already re-rotated/updated particles).
#     """
#     n_candidates = 100
#     candidates = np.linspace(0, 2*np.pi, n_candidates, endpoint=False)
#     thetas = np.asarray(rotations, dtype=float).copy()

#     base_tri = triangle_object()

#     # Initialize placed polygons with initial thetas
#     polys = [
#         triangle_placement(positions[i, 0], positions[i, 1], thetas[i], base_tri)
#         for i in range(N)
#     ]

#     cutoff2 = cutoff * cutoff

#     for _ in range(n_iter):
#         for i in range(N):
#             # build neighbor indices ON THE FLY, but only "past" neighbors (j < i)
#             d = positions[:i] - positions[i]          # shape (i,2)
#             dist2 = np.sum(d * d, axis=1)
#             nbr_idx = np.where(dist2 < cutoff2)[0].tolist()  # these are indices in [0, i-1]

#             if not nbr_idx:
#                 continue

#             neighbor_polys = [polys[j] for j in nbr_idx]  # already-updated polys

#             best_theta = thetas[i]
#             best_poly  = polys[i]
#             best_cost  = overlap_triangle(best_poly, neighbor_polys)

#             for t in candidates:
#                 pi = triangle_placement(positions[i, 0], positions[i, 1], t, base_tri)
#                 c = overlap_triangle(pi, neighbor_polys)
#                 if c < best_cost:
#                     best_cost = c
#                     best_theta = t
#                     best_poly = pi

#             # IMPORTANT: update immediately so later particles see it
#             thetas[i] = best_theta
#             polys[i] = best_poly

#     return thetas


# def make_rotation_sampler(rx_arr, ry_arr, rz_arr):
#     i = {"k": 0}
#     N = len(rz_arr)

#     def sampler():
#         k = i["k"]
#         i["k"] = k + 1
#         return (
#             float(rx_arr[k % N]),
#             float(ry_arr[k % N]),
#             float(rz_arr[k % N]),
#         )
#     return sampler

# def cluster_positions():

#     positions = generate_positions(
#     W=2048, H=2048,
#     # n_bg=5,
#     # n_blobs=2, pts_per_blob=10, blob_spread=50,
#     # n_filaments=1, pts_per_fil=5, fil_thickness=40, min_dist=20,
#     n_bg = np.random.randint(1, 3), #number background particles
#     n_blobs = np.random.randint(4, 8),
#     pts_per_blob = np.random.randint(200, 700),
#     blob_spread = 60,
#     n_filaments = np.random.randint(3, 4),
#     pts_per_fil = np.random.randint(500, 700),
#     fil_thickness = 50,
#     min_dist=20,
# )


#     positions = np.asarray(positions)   # shape (N, 2)
#     N = len(positions)
#     # Make per-particle rx, ry, rz0 arrays (same distribution as rotation_xy_more)
#     rx0 = np.random.randn(N) * np.deg2rad(360)
#     ry0 = np.random.randn(N) * np.deg2rad(30)
#     rz0 = np.random.randn(N) * np.deg2rad(30)
#     amount_particles = len(positions)
#     thetas_x = optimize_angles_sequential(positions, rx0 , amount_particles, n_iter=3, cutoff=35.0)
#     rot_sampler = make_rotation_sampler(thetas_x, ry0, rz0)
#     pos_sampler = make_position_sampler(positions)

#     return rot_sampler, pos_sampler, N



#%%

#%%

import numpy as np


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


def make_grid_sites(
    W,
    H,
    min_dist,
    margin=17,
    jitter_frac=0.0,
    rng=None,
    grid_rotation_deg=0.0,
):
    """
    Create hexagonal / staggered grid sites.

    min_dist is approximately the nearest-neighbor center-to-center distance.
    Every other row is shifted by half a spacing.

    grid_rotation_deg rotates the entire grid around the image center.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 <= jitter_frac < 0.5):
        raise ValueError("jitter_frac must satisfy 0 <= jitter_frac < 0.5")

    if jitter_frac == 0:
        a = float(min_dist)
        jitter_radius = 0.0
    else:
        a = float(min_dist) / (1.0 - 2.0 * jitter_frac)
        jitter_radius = jitter_frac * a

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
        x_offset = 0.5 * dx if row % 2 == 1 else 0.0

        x = x_min + x_offset
        while x <= x_max:
            sites.append([x, y])
            x += dx

        y += dy
        row += 1

    sites = np.asarray(sites, dtype=float)

    # Optional random/local jitter
    if jitter_radius > 0 and len(sites) > 0:
        angles = rng.uniform(0, 2 * np.pi, size=len(sites))
        radii = jitter_radius * np.sqrt(rng.uniform(0, 1, size=len(sites)))

        sites[:, 0] += radii * np.cos(angles)
        sites[:, 1] += radii * np.sin(angles)

    # Rotate whole grid around the image center
    if grid_rotation_deg != 0.0 and len(sites) > 0:
        angle = np.deg2rad(grid_rotation_deg)
        c = np.cos(angle)
        s = np.sin(angle)

        center = np.array([W / 2, H / 2], dtype=float)

        shifted = sites - center

        R = np.array([
            [c, -s],
            [s,  c],
        ])

        sites = shifted @ R.T + center

        # Keep only sites still inside allowed region
        keep = (
            (sites[:, 0] >= margin) &
            (sites[:, 0] <= W - margin) &
            (sites[:, 1] >= margin) &
            (sites[:, 1] <= H - margin)
        )

        sites = sites[keep]

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


# def add_filament_from_grid(sites, free_mask, pts_per_fil, thickness, W, H, rng, n_curve_samples=250):
#     """
#     Pick grid sites near a random cubic Bézier curve.
#     """
#     P0 = rng.random((1, 2)) * [W, H]
#     P3 = rng.random((1, 2)) * [W, H]
#     P1 = P0 + (rng.normal(size=(1, 2)) * 0.2 + [0.3, 0.3]) * [W, H]
#     P2 = P3 + (rng.normal(size=(1, 2)) * 0.2 - [0.3, 0.3]) * [W, H]

#     curve = sample_bezier(P0, P1, P2, P3, n_curve_samples)  # (M,2)

#     free_idx = np.where(free_mask)[0]
#     if len(free_idx) == 0:
#         return np.empty((0, 2))

#     free_sites = sites[free_idx]  # (N,2)

#     # distance from each free grid site to nearest curve sample
#     diff = free_sites[:, None, :] - curve[None, :, :]
#     d2_min = np.min(np.sum(diff * diff, axis=2), axis=1)

#     weights_local = np.exp(-0.5 * d2_min / (thickness ** 2))
#     weights = np.zeros(len(sites), dtype=float)
#     weights[free_idx] = weights_local

#     chosen = _choose_from_free(free_mask, weights, pts_per_fil, rng)
#     free_mask[chosen] = False
#     return sites[chosen]

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
    random_grid_rotation_deg=0.0,
    seed=None,
):
    """
    Grid-based clustered position generator.

    Minimum distance is enforced by construction.
    """
    rng = np.random.default_rng(seed)

    grid_rotation_deg = rng.uniform(
    -random_grid_rotation_deg,
    random_grid_rotation_deg
)

    print("grid rotation deg =", grid_rotation_deg)

    sites = make_grid_sites(
        W=W,
        H=H,
        min_dist=min_dist,
        margin=margin,
        jitter_frac=jitter_frac,
        rng=rng,
        grid_rotation_deg=grid_rotation_deg,
    )

    free_mask = np.ones(len(sites), dtype=bool)
    chunks = []

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

    # background
    if n_bg > 0:
        chunks.append(add_background_from_grid(sites, free_mask, n_bg, rng))

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

def triangle_placement(x: float, y: float, theta: float, base_tri: Polygon) -> Polygon:
    p = rotate(base_tri, theta, origin=(0, 0), use_radians=True)
    return translate(p, xoff=x, yoff=y)

def triangle_object(side_px=30):
    R = side_px / np.sqrt(3)

    return Polygon([
        ( R, 0.0),
        (-R/2,  np.sqrt(3)*R/2),
        (-R/2, -np.sqrt(3)*R/2),
    ])

def overlap_triangle(i_poly: Polygon, neighbor_polys: list[Polygon]) -> float:
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





def triangle_object(side_px=31.25):
    """
    Equilateral triangle centered at (0, 0).
    This base triangle points along +x before rotation.
    """
    R = side_px / np.sqrt(3)

    return Polygon([
        ( R, 0.0),
        (-R / 2,  np.sqrt(3) * R / 2),
        (-R / 2, -np.sqrt(3) * R / 2),
    ])


def triangle_placement(x, y, theta, base_tri):
    """
    Place a triangle at position (x, y) with in-plane angle theta.
    theta is in radians.
    """
    p = rotate(base_tri, theta, origin=(0, 0), use_radians=True)
    return translate(p, xoff=x, yoff=y)


def generate_triangle_crystal_limited_overlap_px(
    W=1024,
    H=1024,
    side_px=31.25,
    gap_px=1.0,
    margin=None,
    vacancy_fraction=0.04,
    pos_jitter_px=0.25,
    rot_jitter_deg=2.0,
    n_domains=4,
    domain_angle_std_deg=4.0,
    domain_shift_px=1.0,
    line_defect=True,
    line_defect_width_px=8.0,
    line_defect_remove_probability=0.5,
    max_overlap_fraction=0.10,
    max_tries=40,
    seed=None,
):
    """
    Generate a close-packed crystal of triangular particles in pixel coordinates.

    Returns
    -------
    positions : (N, 2) array
        Particle center positions in pixels.
    theta2d : (N,) array
        In-plane rotations in radians.
    domain_id : (N,) array
        Domain labels.
    """

    rng = np.random.default_rng(seed)

    if margin is None:
        margin = side_px

    # Base triangle points along +x, so pi/2 makes it point upward.
    theta_up = np.pi / 2
    theta_down = theta_up + np.pi

    a = side_px + gap_px
    h = np.sqrt(3) / 2 * a

    candidates = []

    # Generate larger region, then crop.
    j_min = int(-2 * margin / h) - 5
    j_max = int((H + 2 * margin) / h) + 5

    i_min = -j_max - 5
    i_max = int((W + 2 * margin) / a) + j_max + 5

    for j in range(j_min, j_max):
        for i in range(i_min, i_max):

            x0 = a * (i + 0.5 * j)
            y0 = h * j

            # Down triangle centroid
            x_down = x0 + 0.5 * a
            y_down = y0 + h / 3

            # Up triangle centroid
            x_up = x0 + a
            y_up = y0 + 2 * h / 3

            candidates.append([x_down, y_down, theta_down])
            candidates.append([x_up, y_up, theta_up])
    candidates = np.asarray(candidates, dtype=float)

    # Crop by center first.
    keep = (
        (candidates[:, 0] >= margin) &
        (candidates[:, 0] <= W - margin) &
        (candidates[:, 1] >= margin) &
        (candidates[:, 1] <= H - margin)
    )

    candidates = candidates[keep]

    positions = candidates[:, :2].copy()
    theta2d = candidates[:, 2].copy()

    # Crystal domains / grains
    if n_domains > 1:
        domain_centers = np.column_stack([
            rng.uniform(margin, W - margin, n_domains),
            rng.uniform(margin, H - margin, n_domains),
        ])

        dists = np.linalg.norm(
            positions[:, None, :] - domain_centers[None, :, :],
            axis=2
        )

        domain_id = np.argmin(dists, axis=1)

        domain_angles = rng.normal(
            0,
            np.deg2rad(domain_angle_std_deg),
            n_domains
        )

        domain_shifts = rng.normal(
            0,
            domain_shift_px,
            size=(n_domains, 2)
        )

        for d in range(n_domains):
            mask = domain_id == d

            angle = domain_angles[d]
            c, s = np.cos(angle), np.sin(angle)
            Rmat = np.array([[c, -s], [s, c]])

            local = positions[mask] - domain_centers[d]
            positions[mask] = local @ Rmat.T + domain_centers[d]

            positions[mask] += domain_shifts[d]

            # IMPORTANT:
            # image y-axis points downward, so visual rotation sign is flipped
            theta2d[mask] -= angle

    else:
        domain_id = np.zeros(len(positions), dtype=int)

    # Position jitter
    positions[:, 0] += rng.normal(0, pos_jitter_px, len(positions))
    positions[:, 1] += rng.normal(0, pos_jitter_px, len(positions))

    # Rotation jitter
    theta2d += rng.normal(0, np.deg2rad(rot_jitter_deg), len(theta2d))

    # Random vacancies
    keep = rng.random(len(positions)) > vacancy_fraction
    positions = positions[keep]
    theta2d = theta2d[keep]
    domain_id = domain_id[keep]

    # Optional line defect
    if line_defect and len(positions) > 0:
        x = positions[:, 0]
        y = positions[:, 1]

        y0 = rng.uniform(0.25 * H, 0.75 * H)
        slope = rng.normal(0, 0.2)

        line_y = y0 + slope * (x - W / 2)
        dist_to_line = np.abs(y - line_y) / np.sqrt(1 + slope**2)

        near_line = dist_to_line < line_defect_width_px
        remove = near_line & (
            rng.random(len(positions)) < line_defect_remove_probability
        )

        keep = ~remove
        positions = positions[keep]
        theta2d = theta2d[keep]
        domain_id = domain_id[keep]

    # Now accept particles only if overlap is not too large.
    base_tri = triangle_object(side_px=side_px)
    triangle_area = base_tri.area
    max_overlap_area = max_overlap_fraction * triangle_area

    R_particle = side_px / np.sqrt(3)
    cell_size = 2.5 * R_particle + 2 * pos_jitter_px + gap_px

    accepted_positions = []
    accepted_thetas = []
    accepted_domains = []
    accepted_polys = []

    grid = {}

    order = rng.permutation(len(positions))

    for idx in order:
        x0, y0 = positions[idx]
        th0 = theta2d[idx]
        dom = domain_id[idx]

        placed = False

        for attempt in range(max_tries):

            if attempt == 0:
                x = x0
                y = y0
                th = th0
            else:
                x = x0 + rng.normal(0, pos_jitter_px)
                y = y0 + rng.normal(0, pos_jitter_px)
                th = th0 + rng.normal(0, np.deg2rad(rot_jitter_deg))

            poly = triangle_placement(x, y, th, base_tri)

            # Require full triangle to be inside the image.
            minx, miny, maxx, maxy = poly.bounds
            if minx < 0 or maxx > W or miny < 0 or maxy > H:
                continue

            cx = int(np.floor(x / cell_size))
            cy = int(np.floor(y / cell_size))

            nearby_ids = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby_ids.extend(grid.get((cx + dx, cy + dy), []))

            too_much_overlap = False

            for j in nearby_ids:
                inter = poly.intersection(accepted_polys[j])

                if not inter.is_empty:
                    overlap_area = inter.area

                    if overlap_area > max_overlap_area:
                        too_much_overlap = True
                        break

            if not too_much_overlap:
                new_id = len(accepted_positions)

                accepted_positions.append([x, y])
                accepted_thetas.append(th)
                accepted_domains.append(dom)
                accepted_polys.append(poly)

                grid.setdefault((cx, cy), []).append(new_id)

                placed = True
                break

        # If not placed, skip it. This naturally creates defects.

    positions_out = np.asarray(accepted_positions, dtype=float)
    theta_out = np.asarray(accepted_thetas, dtype=float)
    domain_out = np.asarray(accepted_domains, dtype=int)

    return positions_out, theta_out, domain_out


# def cluster_positions():
#     n_bg = np.random.randint(20, 200)
#     n_blobs = np.random.randint(1, 3)
#     pts_per_blob = np.random.randint(10, 100)
#     n_filaments = np.random.randint(0, 2)
#     pts_per_fil = np.random.randint(50, 300)
#     thickness_base=np.random.randint(pts_per_fil*0.2, pts_per_fil*0.4)



#     print("n_bg =", n_bg)
#     print("n_blobs =", n_blobs, "pts_per_blob =", pts_per_blob)
#     print("n_filaments =", n_filaments, "pts_per_fil =", pts_per_fil)
#     print("requested total ≈", n_bg + n_blobs * pts_per_blob + n_filaments * pts_per_fil)

#     positions = generate_positions_grid(
#         W=1024, H=1024,
#         n_bg=n_bg,
#         n_blobs=n_blobs,
#         pts_per_blob=pts_per_blob,
#         blob_spread=np.random.randint(pts_per_blob*0.6, pts_per_blob*0.8),
#         n_filaments=n_filaments,
#         pts_per_fil=pts_per_fil,
#         # fil_thickness=15,
#         thickness_base=thickness_base,
#         thickness_bend_gain=thickness_base*0.5,
#         min_dist=16,
#         margin=17,
#         jitter_frac=0.2
#         )

#     print("actual generated =", len(positions))

#     positions = np.asarray(positions)   # shape (N, 2)
#     N = len(positions)
#     # Make per-particle rx, ry, rz0 arrays (same distribution as rotation_xy_more)
#     rx0 = np.random.randn(N) * np.deg2rad(360)
#     ry0 = np.random.randn(N) * np.deg2rad(20)
#     rz0 = np.random.randn(N) * np.deg2rad(20)
#     amount_particles = len(positions)
#     thetas_x = optimize_angles_sequential(positions, rx0 , amount_particles, n_iter=5, cutoff=35.0)
#     rot_sampler = make_rotation_sampler(thetas_x, ry0, rz0) #theta_x   rx0 is actually rz0 in my code so ignore this with debugging!
#     pos_sampler = make_position_sampler(positions)

#     return rot_sampler, pos_sampler, N


























# def cluster_positions():
#     """
#     Replacement for your old cluster_positions().

#     Returns
#     -------
#     rot_sampler, pos_sampler, N
#     """

#     W = 512
#     H = 512

#     um_per_pixel = 0.32
#     side_length_um = 10.0
#     side_px = side_length_um / um_per_pixel   # 10 µm / 0.32 µm px^-1 = 31.25 px

#     positions, theta2d, domain_id = generate_triangle_crystal_limited_overlap_px(
#         W=W,
#         H=H,
#         side_px=side_px,

#         # Crystal packing
#         gap_px=3,

#         # Defects/disorder
#         vacancy_fraction=0,
#         pos_jitter_px=1,
#         rot_jitter_deg=np.random.randint(0, 5),

#         # Domains
#         n_domains=np.random.randint(2, 4),
#         domain_angle_std_deg= np.random.uniform(2.0, 10.0),
#         domain_shift_px=np.random.uniform(1, 4),

#         # Line defect
#         line_defect=False,
#         line_defect_width_px=np.random.uniform(5.0, 12.0),
#         line_defect_remove_probability=np.random.uniform(0.3, 0.7),

#         # Important overlap rule
#         max_overlap_fraction=0.20,

#         max_tries=40,
#         seed=None,
#     )

#     # positions, theta2d, domain_id = generate_triangle_crystal_limited_overlap_px(
#     #     W=W,
#     #     H=H,
#     #     side_px=side_px,

#     #     gap_px=2.0,

#     #     vacancy_fraction=0.0,
#     #     pos_jitter_px=0.0,
#     #     rot_jitter_deg=0.0,

#     #     n_domains=1,
#     #     domain_angle_std_deg=0.0,
#     #     domain_shift_px=0.0,

#     #     line_defect=False,

#     #     max_overlap_fraction=0.2,
#     #     max_tries=40,
#     #     seed=0,
#     # )

#     N = len(positions)

#     print("crystal particles =", N)

#     # Small out-of-plane tilts
#     rx_arr = np.random.randn(N) * np.deg2rad(10)
#     ry_arr = np.random.randn(N) * np.deg2rad(10)

#     # In-plane triangle angle
#     rz_arr = theta2d

#     rot_sampler = make_rotation_sampler(rz_arr, ry_arr, rx_arr)
#     pos_sampler = make_position_sampler(positions)

#     return rot_sampler, pos_sampler, N



# def cluster_positions(
#     angle_deg=0,
#     axis="rz",
#     W=512,
#     H=512,
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

#     Returns
#     -------
#     rot_sampler, pos_sampler, N
#     """

#     # one particle in the center
#     positions = np.array([[W / 2, H / 2]], dtype=float)
#     N = 1

#     angle_rad = np.deg2rad(angle_deg)

#     # all zero first
#     rx_arr = np.zeros(N)
#     ry_arr = np.full(N, np.deg2rad(10))
#     rz_arr = np.full(N, np.deg2rad(10))


#     rot_sampler = make_rotation_sampler(rz_arr, ry_arr, rx_arr)
#     pos_sampler = make_position_sampler(positions)


#     return rot_sampler, pos_sampler, N



"""

FIXED FROM HERE





"""


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


def optimize_angles_sequential(
    positions, 
    rotations, 
    N, 
    n_iter=1, 
    cutoff=None,
    n_sides=3,
    side_px=None,
    n_candidates=100,
    TRIANGLE_INNER_R_PX=None,
):
    """
    Sequential greedy optimization:
    when optimizing particle i, only consider neighbors with index < i
    (i.e., already re-rotated/updated particles).
    """

    thetas = np.asarray(rotations, dtype=float).copy()

    base_poly = regular_polygon_object(
        n_sides=n_sides,
        side_px=side_px,
    )


    if cutoff is None:
        cutoff = 2 * TRIANGLE_INNER_R_PX + 5.0 



    period = 2 * np.pi / n_sides
    candidates = np.linspace(0, period, n_candidates, endpoint=False)

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


















def cluster_positions(scale, side_um, um_per_pixel):
    N_sides = 3
    UM_PER_PIXEL = um_per_pixel
    SIDE_UM = side_um
    SIDE_PX = scale * SIDE_UM / UM_PER_PIXEL
    TRIANGLE_R_PX = SIDE_PX / (2 * np.sin(np.pi / N_sides))
    TRIANGLE_INNER_R_PX = SIDE_PX / (2 * np.tan(np.pi / N_sides))


    n_bg = np.random.randint(30, 60)

    n_blobs = np.random.randint(1, 3)
    # n_blobs = 0
    pts_per_blob = np.random.randint(20, 30)

    n_filaments = 0
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
        blob_spread=30 * scale,

        n_filaments=n_filaments,
        pts_per_fil=pts_per_fil,
        thickness_base=5 * scale,
        thickness_bend_gain=30 * scale,

        # Important for triangles:
        # side = 21.9 px, apothem = 15.1 px, circumradius = 18.6 px
        # min_dist=16 is probably too small for triangles unless you want strong overlap.
        min_dist=TRIANGLE_INNER_R_PX * 2 + 6*scale,

        # margin should be larger than the triangle circumradius
        margin=int(np.ceil(TRIANGLE_R_PX + 3)) * scale,

        jitter_frac=0.0,
        random_grid_rotation_deg=20.0,
    )

    print("actual generated =", len(positions))

    positions = np.asarray(positions)
    N = len(positions)

    # Small out-of-plane tilts
    rx_arr = np.random.randn(N) * np.deg2rad(10)   # out-of-plane tilt
    ry_arr = np.random.randn(N) * np.deg2rad(10)   # out-of-plane tilt


    # Initial in-plane triangle rotation.
    # A regular triangle repeats every 120 degrees.
    rz0 = np.random.uniform(0, 2 * np.pi / N_sides, size=N)

    rz_arr = optimize_angles_sequential(
        positions,
        rz0,
        N,
        n_iter=10,
        cutoff=2 * TRIANGLE_R_PX + 5,
        n_sides=N_sides,
        side_px=SIDE_PX,
        n_candidates=200,
    )

    # For a normal DeepTrack rotation=(rx, ry, rz),
    # the optimized in-plane angle should go into rz.
    rot_sampler = make_rotation_sampler(rx_arr, ry_arr, rz_arr)
    pos_sampler = make_position_sampler(positions)

    return rot_sampler, pos_sampler, N





def clusterpositions_test_one(scale, side_um, um_per_pixel):
    N_sides = 3
    UM_PER_PIXEL = um_per_pixel
    SIDE_UM = side_um
    SIDE_PX = scale * SIDE_UM / UM_PER_PIXEL
    TRIANGLE_R_PX = SIDE_PX / (2 * np.sin(np.pi / N_sides))
    TRIANGLE_INNER_R_PX = SIDE_PX / (2 * np.tan(np.pi / N_sides))

    positions = np.array([[256*scale, 256*scale]])

    rx_arr = np.array([0])
    ry_arr = np.array([0])
    rz_arr = np.array([np.deg2rad(0)])

    rot_sampler = make_rotation_sampler(rx_arr, ry_arr, rz_arr)
    pos_sampler = make_position_sampler(positions)

    return rot_sampler, pos_sampler, len(positions)


