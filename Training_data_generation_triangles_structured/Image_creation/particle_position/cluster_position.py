#%%
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate




def regular_polygon_object(n_sides=5, side_px=30.0, theta0=0.0):

    R = side_px / (2 * np.sin(np.pi / n_sides))  

    angles = theta0 + 2 * np.pi * np.arange(n_sides) / n_sides

    verts = np.column_stack([
        R * np.cos(angles),
        R * np.sin(angles),
    ])

    return Polygon(verts)

def resample_polyline(points, n_samples=250):

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

    # smoothing
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

    if jitter_radius > 0 and len(sites) > 0:
        angles = rng.uniform(0, 2 * np.pi, size=len(sites))
        radii = jitter_radius * np.sqrt(rng.uniform(0, 1, size=len(sites)))

        sites[:, 0] += radii * np.cos(angles)
        sites[:, 1] += radii * np.sin(angles)

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

        keep = (
            (sites[:, 0] >= margin) &
            (sites[:, 0] <= W - margin) &
            (sites[:, 1] >= margin) &
            (sites[:, 1] <= H - margin)
        )

        sites = sites[keep]

    return sites


def _choose_from_free(free_mask, weights, k, rng):

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

    R = side_px / np.sqrt(3)

    return Polygon([
        ( R, 0.0),
        (-R / 2,  np.sqrt(3) * R / 2),
        (-R / 2, -np.sqrt(3) * R / 2),
    ])


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

    positions = np.array([[256*scale, 256*scale]])
    rx_arr = np.array([np.deg2rad(20)])
    ry_arr = np.array([np.deg2rad(0)])
    rz_arr = np.array([np.deg2rad(0)])

    rot_sampler = make_rotation_sampler(rx_arr, ry_arr, rz_arr)
    pos_sampler = make_position_sampler(positions)

    return rot_sampler, pos_sampler, len(positions)


