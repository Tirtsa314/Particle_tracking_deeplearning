
#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import nd2
from pprint import pprint
import pandas as pd
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import networkx as nx
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize


def to_uint8(img, mode="minmax", p_low=1, p_high=99):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img.copy()

    img = img.astype(np.float32)

    if mode == "divide256":
        img8 = np.clip(img / 256.0, 0, 255).astype(np.uint8)

    elif mode == "minmax":
        mn, mx = img.min(), img.max()
        if mx <= mn:
            img8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img8 = ((img - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    elif mode == "percentile":
        lo, hi = np.percentile(img, [p_low, p_high])
        if hi <= lo:
            img8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img8 = ((img - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

    else:
        raise ValueError("mode must be 'minmax', 'percentile', or 'divide256'")

    return img8


def make_starts(length, tile, stride):
    if length < tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))
    last = length - tile

    if starts[-1] != last:
        starts.append(last)

    return sorted(set(starts))

import json
def detect_particles_tiled(
    model,
    frame_path,
    conf=0.05,
    tile=512,
    stride=256,
    min_dist=17,
    uint8_mode="percentile",
    p_low=1,
    p_high=99,
    save_csv_path=None,
    verbose=True,
):
    """
    Run tiled YOLO detection on one grayscale image and return particle centers.

    Parameters
    ----------
    model : ultralytics.YOLO
        Already loaded YOLO model.
    frame_path : str
        Path to the image file.
    conf : float
        Confidence threshold for YOLO prediction.
    tile : int
        Tile size.
    stride : int
        Tile stride.
    min_dist : float
        Minimum allowed distance between accepted detections.
    uint8_mode : str
        Mode for to_uint8().
    p_low, p_high : float
        Percentiles for percentile conversion.
    save_csv_path : str or None
        If given, saves detections to CSV.
    verbose : bool
        Print progress.

    Returns
    -------
    det_df : pandas.DataFrame
        Columns: x, y, det_id, tile_x0, tile_y0
    arr8 : np.ndarray
        8-bit grayscale image used for detection/display
    frame_rgb : np.ndarray
        3-channel image sent to YOLO
    """
    img_raw = np.array(Image.open(frame_path))
    arr8 = to_uint8(img_raw, mode=uint8_mode, p_low=p_low, p_high=p_high)
    frame_rgb = np.stack([arr8, arr8, arr8], axis=-1)

    H, W = frame_rgb.shape[:2]
    ys = make_starts(H, tile, stride)
    xs = make_starts(W, tile, stride)

    if verbose:
        print("img_raw:", img_raw.shape, img_raw.dtype, img_raw.min(), img_raw.max())
        print("arr8   :", arr8.shape, arr8.dtype, arr8.min(), arr8.max())
        print("frame  :", frame_rgb.shape, frame_rgb.dtype)
        print("y starts:", ys)
        print("x starts:", xs)
        print("total tiles:", len(ys) * len(xs))

    all_rows = []

    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            y1 = y0 + tile
            x1 = x0 + tile
            tile_img = frame_rgb[y0:y1, x0:x1]

            results = model.predict(
                source=tile_img,
                conf=conf,
                imgsz=tile,
                save=False,
                verbose=False,
            )

            result = results[0]

            if result.masks is not None:
                polys = result.masks.xy

                for det_id, poly_xy in enumerate(polys):
                    poly_xy = np.asarray(poly_xy)

                    if len(poly_xy) < 3:
                        continue

                    cx_tile, cy_tile = polygon_centroid(poly_xy)

                    cx = cx_tile + x0
                    cy = cy_tile + y0

                    too_close = False
                    for row in all_rows:
                        old_x, old_y = row[0], row[1]
                        dist = np.sqrt((cx - old_x) ** 2 + (cy - old_y) ** 2)
                        if dist < min_dist:
                            too_close = True
                            break

                    if not too_close:
                        

                        poly_global = poly_xy.copy()
                        poly_global[:, 0] += x0
                        poly_global[:, 1] += y0

                        all_rows.append([
                            cx,
                            cy,
                            det_id,
                            x0,
                            y0,
                            json.dumps(poly_global.tolist())
                        ])

            if verbose:
                print(f"tile ({yi}, {xi}) done")

    det_df = pd.DataFrame(
    all_rows,
    columns=["x", "y", "det_id", "tile_x0", "tile_y0", "poly_json"]
)

    if save_csv_path is not None:
        det_df.to_csv(save_csv_path, index=False)

    return det_df, arr8, frame_rgb

def compute_coord_num_from_df(det_df, cutoff):
    particles_xy = det_df[["x", "y"]].to_numpy()

    if len(particles_xy) == 0:
        return np.nan, np.array([])

    tree = cKDTree(particles_xy)
    coord_num = tree.query_ball_point(particles_xy, r=cutoff, return_length=True) - 1
    mean_coordination = np.mean(coord_num)

    return mean_coordination, coord_num

def compute_coord_num(particles_csv, cutoff, min_persist=2):
    particles = pd.read_csv(particles_csv)

    particles_xy = particles[['x', 'y']].values

    if len(particles_xy) == 0:
        return np.nan, np.array([])

    particles_tree = cKDTree(particles_xy)

    coord_num = particles_tree.query_ball_point(particles_xy, r=cutoff, return_length=True) - 1
    mean_coordination = np.mean(coord_num)

    return mean_coordination, coord_num


def radial_profile_from_tree_single_particle(
    det_df,
    particle_index,
    um_per_px=0.32,
    dr_um=2.0,
    r_max_um=100.0,
    image_shape=None,
    boundary_safe=True,
):
    """
    Radial density profile around one particle using one pre-built cKDTree.

    Returns density in particles / µm² and normalized g(r).
    """

    coords_px = det_df[["x", "y"]].to_numpy()
    coords_um = coords_px * um_per_px

    tree = cKDTree(coords_um)

    center = coords_um[particle_index]
    center_x, center_y = center

    # avoid edge effects by limiting r_max to nearest image boundary
    if image_shape is not None and boundary_safe:
        H, W = image_shape
        W_um = W * um_per_px
        H_um = H * um_per_px

        dist_to_edge = min(
            center_x,
            W_um - center_x,
            center_y,
            H_um - center_y,
        )

        r_max_um = min(r_max_um, dist_to_edge)

    # radii / cutoffs
    r_edges = np.arange(0, r_max_um + dr_um, dr_um)

    cumulative_counts = []

    for r in r_edges[1:]:
        neighbours = tree.query_ball_point(center, r=r)

        # remove the selected particle itself
        count = len(neighbours) - 1

        cumulative_counts.append(count)

    cumulative_counts = np.array(cumulative_counts)

    # convert cumulative counts into shell counts
    shell_counts = np.diff(np.r_[0, cumulative_counts])

    r_inner = r_edges[:-1]
    r_outer = r_edges[1:]
    r_mid = 0.5 * (r_inner + r_outer)

    shell_area = np.pi * (r_outer**2 - r_inner**2)

    radial_density = shell_counts / shell_area

    # average density of full image, for g(r)-like normalization
    if image_shape is not None:
        H, W = image_shape
        total_area_um2 = (H * um_per_px) * (W * um_per_px)
        average_density = len(det_df) / total_area_um2
        g_r = radial_density / average_density
    else:
        average_density = np.nan
        g_r = np.full_like(radial_density, np.nan)

    profile_df = pd.DataFrame({
        "r_um": r_mid,
        "r_inner_um": r_inner,
        "r_outer_um": r_outer,
        "cumulative_count": cumulative_counts,
        "shell_count": shell_counts,
        "shell_area_um2": shell_area,
        "density": radial_density,
        "g_r": g_r,
    })

    return profile_df

def polygon_centroid(poly):
    """
    Area-weighted centroid of a polygon.
    poly: array with shape (N, 2)
    """
    poly = np.asarray(poly, dtype=float)
    x = poly[:, 0]
    y = poly[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    A = 0.5 * np.sum(cross)

    if abs(A) < 1e-9:
        return x.mean(), y.mean()

    cx = np.sum((x + x_next) * cross) / (6 * A)
    cy = np.sum((y + y_next) * cross) / (6 * A)

    return cx, cy

# ----------------------------
# paths
# -----------------------------
csv_path = r"C:\Users\DenHaan\Downloads\tiled_detections.csv"
frame_path = r"C:\Users\DenHaan\Downloads\lastframe.png"
cutoff = 100

# -----------------------------
# load existing detections
# -----------------------------
det_df = pd.read_csv(csv_path)

# compute neighbours from the same detections
mean_coordination, coord_num = compute_coord_num_from_df(det_df, cutoff)
det_df["coord_num"] = coord_num

print("mean coordination =", mean_coordination)
print(det_df.head())

# load image for plotting
img_raw = np.array(Image.open(frame_path))
arr8 = to_uint8(img_raw, mode="percentile", p_low=1, p_high=99)

# -----------------------------
# plot neighbour map on image
# -----------------------------
#%%

um_per_px = 0.32

det_df["x_um"] = det_df["x"] * um_per_px
det_df["y_um"] = det_df["y"] * um_per_px

H, W = arr8.shape
W_um = W * um_per_px
H_um = H * um_per_px


fig, ax = plt.subplots(figsize=(10, 10))

sc = ax.scatter(
    det_df["x_um"],
    det_df["y_um"],
    c=det_df["coord_num"],
    s=25,
    cmap="plasma",
    edgecolors="white",
    linewidths=0.3
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Number of neighbours within cutoff")

ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title(f"Neighbour map (cutoff = {cutoff * um_per_px:.2f} µm)")
ax.set_aspect("equal")

# optional: same limits as full image size
ax.set_xlim(0, W_um)
ax.set_ylim(0, H_um)

plt.show()
# %%

det_df = det_df.reset_index(drop=True)
particle_index = det_df["coord_num"].idxmax()

print("chosen particle_index =", particle_index)
print(det_df.loc[particle_index])
particle_index = det_df["coord_num"].idxmax()   # or choose manually

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(arr8, cmap="gray")

# all detections
ax.scatter(
    det_df["x"],
    det_df["y"],
    s=15,
    c=det_df["coord_num"],
    cmap="plasma",
    edgecolors="white",
    linewidths=0.3
)

# selected particle
x0 = det_df.loc[particle_index, "x"]
y0 = det_df.loc[particle_index, "y"]

ax.scatter(
    x0,
    y0,
    s=250,
    facecolors="none",
    edgecolors="cyan",
    linewidths=2.5
)

# optional label
ax.text(
    x0 + 10, y0 + 10,
    str(particle_index),
    color="cyan",
    fontsize=12,
    weight="bold"
)

ax.set_title(f"Overlay of selected particle index = {particle_index}")
ax.set_axis_off()

plt.show()
#%%
