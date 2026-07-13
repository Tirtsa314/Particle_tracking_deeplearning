
#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import nd2
from pprint import pprint
import pandas as pd
from matplotlib.patches import Circle
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from pathlib import Path as FilePath
from matplotlib.path import Path as MplPath

def to_uint8(img, mode="minmax", p_low=0, p_high=100):

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


from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_full_detections_red(arr8, det_df, figsize=(12, 12), alpha=0.35):
    # make grayscale image RGB so the red overlay shows nicely
    bg = np.stack([arr8, arr8, arr8], axis=-1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(bg)

    for _, row in det_df.iterrows():
        poly = np.array(json.loads(row["poly_json"]))

        if len(poly) < 3:
            continue

        patch = Polygon(
            poly,
            closed=True,
            facecolor=(1, 0, 0, alpha),   # red transparent fill
            edgecolor=(1, 0, 0, 0.45),     # red edge
            linewidth=0.5
        )
        ax.add_patch(patch)

    ax.set_title(f"Full image with {len(det_df)} detections")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


model = YOLO(r"C:\Users\DenHaan\Downloads\best_09-04_20epoch_ratio1.pt")

#%% 
"""Uitlezen nd2:
"""


# nd2_path = r"c:\Users\Public\Hydrazine 010.nd2"
nd2_path = r"c:\Users\Public\Hydrazine 003(good).nd2"
file_name = FilePath(nd2_path).stem
print(file_name)


FRAME_INDEX = -1   # -1 = last frame, 0 = first frame, 10 = frame 10, etc.

# Optional: inspect ND2 metadata
with nd2.ND2File(nd2_path) as f:
    print("shape:", f.shape)
    print("sizes:", f.sizes)
    print("voxel size:", f.voxel_size())

    print("\n=== text_info ===")
    pprint(f.text_info)

    print("\n=== experiment ===")
    pprint(f.experiment)

    print("\n=== first frame metadata ===")
    pprint(f.frame_metadata(0))


# Read ND2 as lazy dask array
framesarr = nd2.imread(nd2_path, dask=True)

print("framesarr shape:", framesarr.shape)
print("framesarr ndim :", framesarr.ndim)

# Select frame safely
if framesarr.ndim == 2:
    # ND2 contains only one 2D image
    raw_frame = framesarr

elif framesarr.ndim == 3:
    # ND2 contains multiple frames: shape probably (T, Y, X)
    raw_frame = framesarr[FRAME_INDEX]

elif framesarr.ndim == 4:
    # ND2 may have time + channel or z
    # common shape: (T, C, Y, X) or (T, Z, Y, X)
    raw_frame = framesarr[FRAME_INDEX, 0]

else:
    raise ValueError(f"Unexpected ND2 shape: {framesarr.shape}")

# Convert from dask array to numpy array
try:
    img16 = raw_frame.compute()
except AttributeError:
    img16 = np.asarray(raw_frame)

img16 = np.squeeze(img16)

print("selected img16 shape:", img16.shape)

if img16.ndim != 2:
    raise ValueError(f"Expected a 2D image, got shape {img16.shape}")

arr8 = to_uint8(img16, mode="percentile", p_low=1, p_high=99)
frame = np.stack([arr8, arr8, arr8], axis=-1)

print("img16:", img16.shape, img16.dtype, img16.min(), img16.max())
print("arr8 :", arr8.shape, arr8.dtype, arr8.min(), arr8.max())
print("frame:", frame.shape, frame.dtype)

#%%


mpp_train = 0.32
side_train_um = 10.0

# mpp_real = 0.65
# side_real_um = 10.7
mpp_real = 0.32
side_real_um = 10.0

train_side_px = side_train_um / mpp_train
real_side_px = side_real_um / mpp_real

side_ref_um = 10.0
mpp_ref = 0.32
side_ref_px = side_ref_um / mpp_ref      # 31.25 px

side_real_px = side_real_um / mpp_real   # for your new case: 1.0 / 0.06 = 16.67 px

physical_scale = side_real_um / side_ref_um
pixel_scale = side_real_px / side_ref_px
#%%

SCALE = train_side_px / real_side_px


CONF = 0.05
TILE = 512
STRIDE = 256   # 256 = half-overlap, so you get the "middle" predictions too


H, W = frame.shape[:2]

frame_scaled = np.array(
    Image.fromarray(frame).resize(
        (int(W * SCALE), int(H * SCALE)),
        resample=Image.BICUBIC
    )
)

Hs, Ws = frame_scaled.shape[:2]

print("arr8 :", arr8.shape, arr8.dtype, arr8.min(), arr8.max())
print("frame:", frame.shape, frame.dtype)

# float copy for accumulation
frame_f = frame.astype(np.float32)
delta_accum = np.zeros_like(frame_f, dtype=np.float32)

# ---------------------------
# helper: fixed tile starts
# ---------------------------
def make_starts(length, tile, stride):
    if length < tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))

ys = make_starts(Hs, TILE, STRIDE)
xs = make_starts(Ws, TILE, STRIDE)

print("y starts:", ys)
print("x starts:", xs)
print("total tiles:", len(ys) * len(xs))

all_rows = []

for yi, y0 in enumerate(ys):
    for xi, x0 in enumerate(xs):
        y1 = y0 + TILE
        x1 = x0 + TILE

        tile = frame_scaled[y0:y1, x0:x1]

        results = model.predict(
            source=tile,
            conf=CONF,
            imgsz=TILE,
            save=False,
            verbose=False
        )

        result = results[0]
        annotated_tile = result.plot(
            boxes=False,
            labels=False,
            conf=False
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # left: raw tile
        axes[0].imshow(tile[..., 0], cmap="gray")
        axes[0].set_title(f"Before detection\nTile ({yi}, {xi})")
        axes[0].axis("off")

        # right: YOLO overlay
        axes[1].imshow(annotated_tile)
        axes[1].set_title(f"After detection\nTile ({yi}, {xi})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        if result.masks is not None:
            polys = result.masks.xy

            for det_id, poly_xy in enumerate(polys):
                poly_xy = np.asarray(poly_xy)

                # skip broken detections
                if len(poly_xy) < 3:
                    continue

                # center of polygon inside the tile
                cx_tile = poly_xy[:, 0].mean()
                cy_tile = poly_xy[:, 1].mean()

                # shift to full-image coordinates
                cx_scaled = cx_tile + x0
                cy_scaled = cy_tile + y0

                cx = cx_scaled / SCALE
                cy = cy_scaled / SCALE

                min_dist = 10* pixel_scale
                too_close = False

                for row in all_rows:
                    old_x = row[1]   # x is now column 1
                    old_y = row[2]   # y is now column 2
                    dist = np.sqrt((cx - old_x)**2 + (cy - old_y)**2)
                    if dist < min_dist:
                        too_close = True
                        break

                if not too_close:
                    # YOLO polygon is in tile coordinates of the scaled image
                    poly_scaled_global = poly_xy.copy()
                    poly_scaled_global[:, 0] += x0
                    poly_scaled_global[:, 1] += y0

                    # convert polygon back to original arr8 coordinates
                    poly_original = poly_scaled_global / SCALE
                    particle_id = len(all_rows)

                    all_rows.append([
                    particle_id,
                    cx,
                    cy,
                    json.dumps(poly_original.tolist())
                ])

        print(f"tile ({yi}, {xi}) done")





#%%

det_df = pd.DataFrame(
    all_rows,
    columns=[
        "particle_id",
        "x",
        "y",
        "poly_json",
    ]
)
plot_full_detections_red(arr8, det_df, alpha=0.45)

#%%
# Overlay of exactly the detections that will be saved in the CSV

fig, ax = plt.subplots(figsize=(10, 10))

# show original frame
ax.imshow(arr8, cmap="gray")

# show detected centers from det_df
ax.scatter(
    det_df["x"],
    det_df["y"],
    s=12,
    facecolors="none",
    edgecolors="red",
    linewidths=0.8,
    label="CSV detections"
)

ax.set_title(f"Overlay of CSV detections: {file_name}\nN = {len(det_df)} particles")
ax.axis("off")
ax.legend(loc="upper right")

plt.show()

out_dir = FilePath(r"C:\particle_csv_files")
out_dir.mkdir(exist_ok=True)

out_path = out_dir / f"{file_name}.csv"

det_df.to_csv(out_path, index=False)

print("Saved to:", out_path)

print(det_df.head())



# %%
plt.figure(figsize=(10, 10))
plt.imshow(arr8, cmap="gray")
plt.scatter(
    det_df["x"],
    det_df["y"],
    s=0.1,
    c=np.arange(len(det_df)),
    cmap="hsv",
)
plt.axis("off")
plt.title("Detected centers as points")
plt.show()

# %%

def polygon_centroid(poly):
    """
    Area-weighted centroid of polygon.
    poly shape: (N, 2), columns x,y.
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

import cv2

def polygon_to_mask(shape, poly):
    """
    Fast polygon rasterization using OpenCV.
    poly is in local/crop coordinates, columns x,y.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    pts = np.asarray(poly, dtype=np.float32)
    pts = np.round(pts).astype(np.int32)

    cv2.fillPoly(mask, [pts], 1)

    return mask.astype(bool)


def triangle_vertices_from_centroid(cx, cy, side_px, theta):
    """
    Equilateral triangle from centroid.
    theta is direction from center to first vertex.
    """
    R = side_px / np.sqrt(3)

    angles = theta + np.array([
        0,
        2*np.pi/3,
        4*np.pi/3
    ])

    xs = cx + R * np.cos(angles)
    ys = cy + R * np.sin(angles)

    return np.column_stack([xs, ys])


def triangle_prism_band_masks(shape, cx, cy, side_px, theta, thickness_px):
    """
    Make a triangular edge band centered around the given triangle size.

    middle triangle = the given side_px
    outer triangle  = middle triangle expanded outward by thickness_px
    inner triangle  = middle triangle inset inward by thickness_px

    So the band thickness exists on BOTH sides of the given triangle edge.
    """

    # If an equilateral triangle edge is moved outward/inward by distance t,
    # the side length changes by 2 * sqrt(3) * t.
    side_mid = side_px
    side_outer = side_px + 2 * np.sqrt(3) * thickness_px
    side_inner = side_px - 2 * np.sqrt(3) * thickness_px

    if side_mid <= 1 or side_inner <= 1:
        return None

    outer_verts = triangle_vertices_from_centroid(cx, cy, side_outer, theta)
    mid_verts = triangle_vertices_from_centroid(cx, cy, side_mid, theta)
    inner_verts = triangle_vertices_from_centroid(cx, cy, side_inner, theta)

    outer_mask = polygon_to_mask(shape, outer_verts)
    mid_mask = polygon_to_mask(shape, mid_verts)
    inner_mask = polygon_to_mask(shape, inner_verts)

    # Full band around the middle edge
    full_band = outer_mask & (~inner_mask)

    # Outside part of the band
    outer_band = outer_mask & (~mid_mask)

    # Inside part of the band
    inner_band = mid_mask & (~inner_mask)

    return outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts


def plot_triangle_prism_fit_near_poly(arr8, poly_xy, res, alpha_poly=0.25):
    """
    Show:
    - image crop
    - YOLO polygon overlay
    - fitted triangle prism outer/mid/inner outlines
    - fitted center
    """
    y_min, y_max, x_min, x_max = res["crop_box"]

    crop = arr8[y_min:y_max, x_min:x_max]

    poly_local = np.asarray(poly_xy, dtype=float).copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min

    poly_closed = np.vstack([poly_local, poly_local[0]])

    outer = res["outer_vertices"].copy()
    mid = res["mid_vertices"].copy()
    inner = res["inner_vertices"].copy()

    for verts in [outer, mid, inner]:
        verts[:, 0] -= x_min
        verts[:, 1] -= y_min

    outer_c = np.vstack([outer, outer[0]])
    mid_c = np.vstack([mid, mid[0]])
    inner_c = np.vstack([inner, inner[0]])

    x_ref = res["x_refined"] - x_min
    y_ref = res["y_refined"] - y_min

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(crop, cmap="gray")

    # YOLO polygon fill
    poly_mask = res["poly_mask_local"]
    overlay = np.zeros((*poly_mask.shape, 4), dtype=float)
    overlay[..., 0] = 1.0
    overlay[..., 3] = alpha_poly * poly_mask.astype(float)
    ax.imshow(overlay)

    # YOLO polygon outline
    ax.plot(
        poly_closed[:, 0],
        poly_closed[:, 1],
        linewidth=2,
        label="YOLO poly outline"
    )

    # fitted triangular prism
    ax.plot(outer_c[:, 0], outer_c[:, 1], linewidth=2, label="outer fitted triangle")
    ax.plot(mid_c[:, 0], mid_c[:, 1], linewidth=2, linestyle="--", label="middle")
    ax.plot(inner_c[:, 0], inner_c[:, 1], linewidth=2, label="inner fitted triangle")

    ax.scatter(x_ref, y_ref, s=80, marker="x", label="fitted center")

    ax.set_title(""
        # f"Triangle prism fit\n"
        # f"dark_score={res['dark_score']:.3f}, "
        # f"mean outline dist={res['mean_outline_distance']:.2f}px, "
        # # f"far={res['far_fraction']:.2f}"
    )

    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.optimize import minimize

#%%
import numpy as np
from scipy.ndimage import gaussian_filter

def polygon_centroid(poly):
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


def compute_initial_middle_brightness_threshold(
    arr8,
    all_polys_xy,
    blur_sigma=0.5,
    global_norm_percentiles=(1, 99),
    middle_radius_px=2.0,
):
    """
    Computes brightness threshold from the average brightness
    around the initial YOLO centroid of every detected triangle.

    Parameters
    ----------
    arr8 : 2D uint8 image
    all_polys_xy : list of arrays
        Each element has shape (N, 2), with x,y polygon vertices.
    blur_sigma : float
        Same smoothing used in fitting.
    global_norm_percentiles : tuple
        Percentiles for normalizing the whole image brightness.
    middle_radius_px : float
        Radius around each initial centroid used to measure middle brightness.

    Returns
    -------
    threshold : float
        Average normalized brightness of all initial triangle middles.
    middle_brightnesses : np.ndarray
        Individual middle brightness values.
    brightness_norm : 2D array
        Whole-image normalized brightness map.
    """

    arr = np.asarray(arr8, dtype=float)

    if blur_sigma is not None and blur_sigma > 0:
        arr_smooth = gaussian_filter(arr, sigma=blur_sigma)
    else:
        arr_smooth = arr

    lo, hi = np.percentile(arr_smooth, global_norm_percentiles)
    brightness_norm = np.clip((arr_smooth - lo) / (hi - lo + 1e-9), 0, 1)

    H, W = brightness_norm.shape
    yy, xx = np.ogrid[:H, :W]

    middle_brightnesses = []

    for poly_xy in all_polys_xy:
        poly_xy = np.asarray(poly_xy, dtype=float)

        if len(poly_xy) < 3:
            continue

        cx, cy = polygon_centroid(poly_xy)

        disk = (xx - cx)**2 + (yy - cy)**2 <= (middle_radius_px)**2

        if disk.sum() == 0:
            continue

        middle_brightnesses.append(np.mean(brightness_norm[disk]))

    middle_brightnesses = np.asarray(middle_brightnesses)

    if len(middle_brightnesses) == 0:
        raise ValueError("No valid middle brightness values found.")

    threshold = np.mean(middle_brightnesses)

    return threshold, middle_brightnesses, brightness_norm



# %%

def fit_triangle_prism_grid_search_near_poly(
    arr8,
    poly_xy,
    um_per_px=0.32,
    side_um=10.0,
    thickness_um=2.0,

    # Make the fitted edge band thicker than the real edge
    band_extra_um=1.0,

    # Search range
    max_center_shift_um=8.0,

    # Coarse grid steps
    center_step_px=3.0,
    theta_step_deg=5.0,

    # Image scoring
    blur_sigma=0.5,
    dark_top_fraction=1.0,
    bright_top_fraction=1.0,

    # Loss weights
    w_dark=1.0,
    w_bright=1.0,

    # Hard rejection limits
    max_yolo_outside_fraction=0.20,

    bright_bad_threshold=None,
    w_bright_fraction=1.0,

    verbose=True,
):
    """
    Brute-force grid-search version of the triangle prism fit.

    It searches over:
    - center x
    - center y
    - rotation theta
    - side length

    using simple nested for loops.

    Returns a result dictionary compatible with your
    plot_triangle_prism_fit_near_poly(...) function.
    """

    arr8 = np.asarray(arr8)
    poly_xy = np.asarray(poly_xy, dtype=float)

    if arr8.ndim != 2:
        raise ValueError("arr8 must be a 2D grayscale image")

    if poly_xy.ndim != 2 or poly_xy.shape[1] != 2:
        raise ValueError("poly_xy must have shape (N, 2)")

    if len(poly_xy) < 3:
        raise ValueError("poly_xy must contain at least 3 points")

    # -------------------------
    # physical units to pixels
    # -------------------------
    side_px = side_um / um_per_px

  
    # Here we make the fitted edge band larger
    thickness_px = (thickness_um + band_extra_um) / um_per_px

    max_center_shift_px = max_center_shift_um / um_per_px


    # -------------------------
    # initial center from YOLO polygon
    # -------------------------
    x0_global, y0_global = polygon_centroid(poly_xy)

    # -------------------------
    # crop around particle
    # -------------------------
    x_min0 = int(np.floor(poly_xy[:, 0].min()))
    x_max0 = int(np.ceil(poly_xy[:, 0].max()))
    y_min0 = int(np.floor(poly_xy[:, 1].min()))
    y_max0 = int(np.ceil(poly_xy[:, 1].max()))

    pad = int(np.ceil(
        side_px / 2
        + thickness_px
        + max_center_shift_px
        + 8
    ))

    H, W = arr8.shape

    x_min = max(0, x_min0 - pad)
    x_max = min(W, x_max0 + pad + 1)
    y_min = max(0, y_min0 - pad)
    y_max = min(H, y_max0 + pad + 1)

    img_crop = arr8[y_min:y_max, x_min:x_max].astype(float)
    crop_shape = img_crop.shape

    # local polygon coordinates
    poly_local = poly_xy.copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min

    x0 = x0_global - x_min
    y0 = y0_global - y_min

    # -------------------------
    # YOLO mask and outline distance
    # -------------------------
    poly_mask = polygon_to_mask(crop_shape, poly_local)

    poly_eroded = binary_erosion(poly_mask)
    poly_outline = poly_mask ^ poly_eroded

    if poly_outline.sum() == 0:
        poly_outline = poly_mask.copy()

    dist_to_outline = distance_transform_edt(~poly_outline)

    yolo_area = poly_mask.sum()

    if yolo_area < 10:
        raise ValueError("YOLO polygon mask is too small")

    # -------------------------
    # darkness and brightness images
    # -------------------------
    img_smooth = gaussian_filter(img_crop, sigma=blur_sigma)

    lo, hi = np.percentile(img_smooth, [1, 99])
    img_norm = np.clip((img_smooth - lo) / (hi - lo + 1e-9), 0, 1)

    darkness = 1.0 - img_norm
    brightness = img_norm

    def top_fraction_mean(vals, fraction):
        vals = np.asarray(vals)

        if len(vals) == 0:
            return np.nan

        k = int(fraction * len(vals))
        k = max(5, k)
        k = min(len(vals), k)

        # largest k values
        top_vals = np.partition(vals, len(vals) - k)[len(vals) - k:]
        return np.mean(top_vals)

    def evaluate_candidate(cx, cy, theta, side_px):
        out = triangle_prism_band_masks(
            crop_shape,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta,
            thickness_px=thickness_px,
        )

        if out is None:
            return None

        outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

        if full_band.sum() < 10:
            return None

        # Check how much YOLO polygon is outside fitted triangle
        fitted_outer_mask = polygon_to_mask(crop_shape, outer_verts)
        yolo_outside = poly_mask & (~fitted_outer_mask)
        yolo_outside_fraction = yolo_outside.sum() / yolo_area

        if yolo_outside_fraction > max_yolo_outside_fraction:
            return None

        # Dark/bright scores inside thick triangular edge band
        band_dark_vals = darkness[full_band]
        band_bright_vals = brightness[full_band]

        dark_score = top_fraction_mean(band_dark_vals, dark_top_fraction)
        bright_score = top_fraction_mean(band_bright_vals, bright_top_fraction)

        if bright_bad_threshold is None:
            bright_fraction = 0.0
        else:
            bright_fraction = np.mean(band_bright_vals >= bright_bad_threshold)

        loss = (
            -w_dark * dark_score
            + w_bright * bright_score
            + w_bright_fraction * bright_fraction
        )

        return {
            "loss": loss,
            "dark_score": dark_score,
            "bright_score": bright_score,
            "bright_fraction": bright_fraction,
            "yolo_outside_fraction": yolo_outside_fraction,
            "outer_band": outer_band,
            "inner_band": inner_band,
            "full_band": full_band,
            "outer_verts": outer_verts,
            "mid_verts": mid_verts,
            "inner_verts": inner_verts,
        }

    # -------------------------
    # make coarse search grids
    # -------------------------
    cx_values = np.arange(
        x0 - max_center_shift_px,
        x0 + max_center_shift_px + 1e-9,
        center_step_px
    )

    cy_values = np.arange(
        y0 - max_center_shift_px,
        y0 + max_center_shift_px + 1e-9,
        center_step_px
    )

    theta_values = np.deg2rad(
        np.arange(0, 120, theta_step_deg)
    )


    if verbose:
        n_total = (
            len(cx_values)
            * len(cy_values)
            * len(theta_values)
        )

        print("Grid search:")
        print("  cx values:", len(cx_values))
        print("  cy values:", len(cy_values))
        print("  theta values:", len(theta_values))
        print("  total candidates:", n_total)
        print("  effective fitting thickness px:", thickness_px)

    # -------------------------
    # brute-force loop
    # -------------------------
    best_info = None
    best_params = None
    tested = 0
    valid = 0
    best_trace = []

    for cx in cx_values:
        for cy in cy_values:
            for theta in theta_values:
                tested += 1

                info = evaluate_candidate(cx, cy, theta, side_px)

                if info is None:
                    continue

                valid += 1

                if best_info is None or info["loss"] < best_info["loss"]:
                    best_info = info
                    best_params = (cx, cy, theta, side_px)

                    best_trace.append({
                        "tested": tested,
                        "valid": valid,
                        "loss": info["loss"],
                        "cx": cx,
                        "cy": cy,
                        "theta": theta,
                        "theta_deg": np.rad2deg(theta),
                        "side_px": side_px,
                        "dark_score": info["dark_score"],
                        "bright_score": info["bright_score"],
                        "outside": info["yolo_outside_fraction"],
                        "bright_fraction": info["bright_fraction"],
                    })

    if best_info is None:
        raise RuntimeError("No valid brute-force candidate found.")

    cx_best, cy_best, theta_best, side_px_best = best_params

    if verbose:
        print("Finished grid search")
        print("  tested:", tested)
        print("  valid:", valid)
        print("  best loss:", best_info["loss"])
        print("  best theta deg:", np.rad2deg(theta_best))
        print("  best side px:", side_px_best)

    # -------------------------
    # convert vertices back to global coordinates
    # -------------------------
    outer_global = best_info["outer_verts"].copy()
    mid_global = best_info["mid_verts"].copy()
    inner_global = best_info["inner_verts"].copy()

    for verts in [outer_global, mid_global, inner_global]:
        verts[:, 0] += x_min
        verts[:, 1] += y_min

    x_refined = cx_best + x_min
    y_refined = cy_best + y_min

    d_final = dist_to_outline[best_info["full_band"]]

    return {
        "x_refined": x_refined,
        "y_refined": y_refined,
        "theta": theta_best,
        "side_px": side_px_best,
        "side_um": side_px_best * um_per_px,
        "thickness_px": thickness_px,
        "loss": best_info["loss"],
        "dark_score": best_info["dark_score"],
        "bright_score": best_info["bright_score"],
        "mean_outline_distance": np.mean(d_final),
        "yolo_outside_fraction": best_info["yolo_outside_fraction"],
        "outer_vertices": outer_global,
        "mid_vertices": mid_global,
        "inner_vertices": inner_global,
        "crop_box": (y_min, y_max, x_min, x_max),
        "poly_local": poly_local,
        "poly_mask_local": poly_mask,
        "poly_outline_local": poly_outline,
        "darkness_crop": darkness,
        "brightness_crop": brightness,
        "outer_band_local": best_info["outer_band"],
        "inner_band_local": best_info["inner_band"],
        "full_band_local": best_info["full_band"],
        "tested": tested,
        "valid": valid,
        "best_trace": best_trace,
    }

#%%
import json
import numpy as np

all_polys_xy = [
    np.array(json.loads(s), dtype=float)
    for s in det_df["poly_json"]
]

middle_brightness_threshold, middle_brightnesses, brightness_norm = (
    compute_initial_middle_brightness_threshold(
        arr8,
        all_polys_xy,
        blur_sigma=0.5,
        middle_radius_px=2.0 * pixel_scale,
        global_norm_percentiles=(1, 99),
    )
)

print("Middle brightness threshold:", middle_brightness_threshold)
print("Mean:", middle_brightnesses.mean())
print("Std:", middle_brightnesses.std())

# %%

for particle_id in range(0, 400):

    print("\nparticle_id:", particle_id)

    try:

        row = det_df[det_df["particle_id"] == particle_id].iloc[0]
        poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)

        res_grid = fit_triangle_prism_grid_search_near_poly(
            arr8,
            poly_xy,
            um_per_px=mpp_real,
            side_um=10.0*physical_scale ,
            thickness_um=0.5*physical_scale ,

            # makes the triangle edge band larger
            band_extra_um=0.2*physical_scale,

            max_center_shift_um=8.0*physical_scale,

            # coarse brute-force steps
            center_step_px=5.0*pixel_scale,
            theta_step_deg=10.0,

            dark_top_fraction=1.0,
            bright_top_fraction=1.0,

            w_dark=1.0,
            w_bright=0.2,

            max_yolo_outside_fraction=0.05,
            bright_bad_threshold=middle_brightness_threshold,
            w_bright_fraction=2.0,


            verbose=True,
        )

        print("fitted side px:", res_grid["side_px"])
        print("fitted side um:", res_grid["side_um"])
        print("theta deg:", np.rad2deg(res_grid["theta"]))
        print("loss:", res_grid["loss"])
        print("valid candidates:", res_grid["valid"], "/", res_grid["tested"])

        plot_triangle_prism_fit_near_poly(arr8, poly_xy, res_grid)

    except Exception as e:
        print("Skipping particle", particle_id)
        print("Reason:", e)
        continue
# %%
def refine_triangle_prism_from_grid_result(
    arr8,
    poly_xy,
    res_grid,
    um_per_px=0.32,
    side_um=10.0,
    thickness_um=0.5,

    # local refinement window around grid result
    refine_center_window_px=6.0,
    refine_theta_window_deg=15.0,

    # NEW: allow side length to change a little
    refine_side_window_px=2.0,

    # image scoring
    blur_sigma=0.5,
    dark_top_fraction=1.0,
    bright_top_fraction=1.0,

    # loss weights
    w_dark=1.0,
    w_bright=0.2,

    # hard rejection
    max_yolo_outside_fraction=0.05,

    maxiter=120,
    verbose=True,
):
    """
    Local Powell refinement after coarse grid search.

    Starts from res_grid and refines:
    - center x
    - center y
    - theta
    - side length

    The given side_um is treated as the expected/nominal side length,
    but side_px can move within refine_side_window_px.
    """

    arr8 = np.asarray(arr8)
    poly_xy = np.asarray(poly_xy, dtype=float)

    side_px_nominal = side_um / um_per_px
    thickness_px = thickness_um / um_per_px

    # -------------------------
    # starting point from grid result
    # -------------------------
    x_start_global = res_grid["x_refined"]
    y_start_global = res_grid["y_refined"]
    theta_start = res_grid["theta"]

    # use grid result side if it exists, otherwise use nominal side
    side_start = res_grid.get("side_px", side_px_nominal)

    side_min = max(1.0, side_start - refine_side_window_px)
    side_max = side_start + refine_side_window_px

    # -------------------------
    # crop around particle
    # -------------------------
    x_min0 = int(np.floor(poly_xy[:, 0].min()))
    x_max0 = int(np.ceil(poly_xy[:, 0].max()))
    y_min0 = int(np.floor(poly_xy[:, 1].min()))
    y_max0 = int(np.ceil(poly_xy[:, 1].max()))

    # use side_max because side can grow during refinement
    pad = int(np.ceil(
        side_max / 2
        + thickness_px
        + refine_center_window_px
        + 8
    ))

    H, W = arr8.shape

    x_min = max(0, x_min0 - pad)
    x_max = min(W, x_max0 + pad + 1)
    y_min = max(0, y_min0 - pad)
    y_max = min(H, y_max0 + pad + 1)

    img_crop = arr8[y_min:y_max, x_min:x_max].astype(float)
    crop_shape = img_crop.shape

    # local coordinates
    poly_local = poly_xy.copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min

    x_start = x_start_global - x_min
    y_start = y_start_global - y_min

    # -------------------------
    # YOLO mask
    # -------------------------
    poly_mask = polygon_to_mask(crop_shape, poly_local)
    yolo_area = poly_mask.sum()

    if yolo_area < 10:
        raise ValueError("YOLO polygon mask is too small")

    poly_eroded = binary_erosion(poly_mask)
    poly_outline = poly_mask ^ poly_eroded

    if poly_outline.sum() == 0:
        poly_outline = poly_mask.copy()

    dist_to_outline = distance_transform_edt(~poly_outline)

    # -------------------------
    # darkness / brightness image
    # -------------------------
    img_smooth = gaussian_filter(img_crop, sigma=blur_sigma)

    lo, hi = np.percentile(img_smooth, [1, 99])
    img_norm = np.clip((img_smooth - lo) / (hi - lo + 1e-9), 0, 1)

    darkness = 1.0 - img_norm
    brightness = img_norm

    def top_fraction_mean(vals, fraction):
        vals = np.asarray(vals)

        if len(vals) == 0:
            return np.nan

        k = int(fraction * len(vals))
        k = max(5, k)
        k = min(len(vals), k)

        top_vals = np.partition(vals, len(vals) - k)[len(vals) - k:]
        return np.mean(top_vals)

    def evaluate_candidate(cx, cy, theta, side_px):
        # triangle has 120 degree symmetry
        theta_mod = theta % (2 * np.pi / 3)

        out = triangle_prism_band_masks(
            crop_shape,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta_mod,
            thickness_px=thickness_px,
        )

        if out is None:
            return None

        outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

        if full_band.sum() < 10:
            return None

        # containment check
        fitted_outer_mask = polygon_to_mask(crop_shape, outer_verts)
        yolo_outside = poly_mask & (~fitted_outer_mask)
        yolo_outside_fraction = yolo_outside.sum() / yolo_area

        if yolo_outside_fraction > max_yolo_outside_fraction:
            return None

        band_dark_vals = darkness[full_band]
        band_bright_vals = brightness[full_band]

        dark_score = top_fraction_mean(band_dark_vals, dark_top_fraction)
        bright_score = top_fraction_mean(band_bright_vals, bright_top_fraction)

    
        loss = (
            -w_dark * dark_score
            + w_bright * bright_score
        )

        return {
            "loss": loss,
            "dark_score": dark_score,
            "bright_score": bright_score,
            "yolo_outside_fraction": yolo_outside_fraction,
            "outer_band": outer_band,
            "inner_band": inner_band,
            "full_band": full_band,
            "outer_verts": outer_verts,
            "mid_verts": mid_verts,
            "inner_verts": inner_verts,
            "theta_mod": theta_mod,
        }

    def objective(params):
        cx, cy, theta, side_px = params

        info = evaluate_candidate(cx, cy, theta, side_px)

        if info is None:
            return 1e6

        return info["loss"]

    # -------------------------
    # Powell bounds around grid result
    # -------------------------
    theta_window = np.deg2rad(refine_theta_window_deg)

    bounds = [
        (x_start - refine_center_window_px, x_start + refine_center_window_px),
        (y_start - refine_center_window_px, y_start + refine_center_window_px),
        (theta_start - theta_window, theta_start + theta_window),
        (side_min, side_max),
    ]

    res = minimize(
        objective,
        x0=np.array([x_start, y_start, theta_start, side_start]),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "xtol": 0.5,
            "ftol": 1e-4,
        },
    )

    cx_best, cy_best, theta_best_raw, side_px_best = res.x

    final_info = evaluate_candidate(
        cx_best,
        cy_best,
        theta_best_raw,
        side_px_best,
    )
    if final_info is None:
        print("Powell ended on invalid candidate, using grid result instead.")
        return res_grid

    theta_best = final_info["theta_mod"]

    if verbose:
        print("Powell refinement:")
        print("  success:", res.success)
        print("  message:", res.message)
        print("  loss:", final_info["loss"])
        print("  theta deg:", np.rad2deg(theta_best))
        print("  side px:", side_px_best)
        print("  side um:", side_px_best * um_per_px)
        print("  dark score:", final_info["dark_score"])
        print("  bright score:", final_info["bright_score"])
        print("  outside fraction:", final_info["yolo_outside_fraction"])

    # -------------------------
    # convert vertices back to global coordinates
    # -------------------------
    outer_global = final_info["outer_verts"].copy()
    mid_global = final_info["mid_verts"].copy()
    inner_global = final_info["inner_verts"].copy()

    for verts in [outer_global, mid_global, inner_global]:
        verts[:, 0] += x_min
        verts[:, 1] += y_min

    x_refined = cx_best + x_min
    y_refined = cy_best + y_min

    d_final = dist_to_outline[final_info["full_band"]]

    return {
        "x_refined": x_refined,
        "y_refined": y_refined,
        "theta": theta_best,
        "side_px": side_px_best,
        "side_um": side_px_best * um_per_px,
        "side_px_nominal": side_px_nominal,
        "side_start_px": side_start,
        "side_start_um": side_start * um_per_px,
        "thickness_px": thickness_px,
        "loss": final_info["loss"],
        "dark_score": final_info["dark_score"],
        "bright_score": final_info["bright_score"],
        "mean_outline_distance": np.mean(d_final),
        "yolo_outside_fraction": final_info["yolo_outside_fraction"],
        "outer_vertices": outer_global,
        "mid_vertices": mid_global,
        "inner_vertices": inner_global,
        "crop_box": (y_min, y_max, x_min, x_max),
        "poly_local": poly_local,
        "poly_mask_local": poly_mask,
        "poly_outline_local": poly_outline,
        "darkness_crop": darkness,
        "brightness_crop": brightness,
        "outer_band_local": final_info["outer_band"],
        "inner_band_local": final_info["inner_band"],
        "full_band_local": final_info["full_band"],
        "powell_result": res,
    }
# %%


for particle_id in range(100,400):

    row = det_df[det_df["particle_id"] == particle_id].iloc[0]
    poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)

    res_grid = fit_triangle_prism_grid_search_near_poly(
        arr8,
        poly_xy,
        um_per_px=mpp_real,
        side_um=10.0,
        thickness_um=0.5,

        # makes the triangle edge band larger
        band_extra_um=0.5,

        max_center_shift_um=5.0,

        # coarse brute-force steps
        center_step_px=10.0,
        theta_step_deg=10.0,

        dark_top_fraction=1.0,
        bright_top_fraction=1.0,

        w_dark=1.0,
        w_bright=0.2,

        max_yolo_outside_fraction=0.05,

        verbose=True,
    )

    print("fitted side px:", res_grid["side_px"])
    print("fitted side um:", res_grid["side_um"])
    print("theta deg:", np.rad2deg(res_grid["theta"]))
    print("loss:", res_grid["loss"])
    print("valid candidates:", res_grid["valid"], "/", res_grid["tested"])

    plot_triangle_prism_fit_near_poly(arr8, poly_xy, res_grid)

    res_precise = refine_triangle_prism_from_grid_result(
        arr8,
        poly_xy,
        res_grid,
        um_per_px=mpp_real,
        side_um=10.0,
        thickness_um=0.5,

        refine_center_window_px=5.0,
        refine_theta_window_deg=15.0,

        # allow side to move by plus/minus 2 pixels
        refine_side_window_px=2.0,

        blur_sigma=0.5,
        dark_top_fraction=1.0,
        bright_top_fraction=0.2,

        w_dark=1.0,
        w_bright=1.0,

        max_yolo_outside_fraction=0.05,
        maxiter=120,
        verbose=True,
    )

    print("precise side px:", res_precise["side_px"])
    print("precise side um:", res_precise["side_um"])
    print("theta deg:", np.rad2deg(res_precise["theta"]))

    plot_triangle_prism_fit_near_poly(arr8, poly_xy, res_precise)


#%%

def _plot_triangle_fit_on_axis(ax, arr8, poly_xy, res, title="", alpha_poly=0.25):
    """
    Internal helper: plot one triangle fit on an existing axis.
    """

    y_min, y_max, x_min, x_max = res["crop_box"]
    crop = arr8[y_min:y_max, x_min:x_max]

    ax.imshow(crop, cmap="gray")

    # YOLO polygon in local crop coordinates
    poly_local = np.asarray(poly_xy, dtype=float).copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min
    poly_closed = np.vstack([poly_local, poly_local[0]])

    # YOLO polygon fill
    poly_mask = res.get("poly_mask_local", None)
    if poly_mask is not None and poly_mask.shape == crop.shape:
        overlay = np.zeros((*poly_mask.shape, 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = alpha_poly * poly_mask.astype(float)
        ax.imshow(overlay)
    else:
        ax.fill(poly_local[:, 0], poly_local[:, 1], alpha=alpha_poly)

    # YOLO polygon outline
    ax.plot(
        poly_closed[:, 0],
        poly_closed[:, 1],
        color="tab:blue",
        linewidth=2,
        label="YOLO outline"
    )

    # fitted triangle outlines
    outer = res["outer_vertices"].copy()
    mid = res["mid_vertices"].copy()
    inner = res["inner_vertices"].copy()

    for verts in [outer, mid, inner]:
        verts[:, 0] -= x_min
        verts[:, 1] -= y_min

    outer_c = np.vstack([outer, outer[0]])
    mid_c = np.vstack([mid, mid[0]])
    inner_c = np.vstack([inner, inner[0]])

    ax.plot(
        outer_c[:, 0], outer_c[:, 1],
        color="tab:orange",
        linewidth=2,
        label="outer fitted triangle"
    )

    ax.plot(
        mid_c[:, 0], mid_c[:, 1],
        color="tab:green",
        linewidth=2,
        linestyle="--",
        label="middle fitted triangle"
    )

    ax.plot(
        inner_c[:, 0], inner_c[:, 1],
        color="tab:red",
        linewidth=2,
        label="inner fitted triangle"
    )

    # fitted center
    x_ref = res["x_refined"] - x_min
    y_ref = res["y_refined"] - y_min

    ax.scatter(
        x_ref,
        y_ref,
        s=80,
        marker="x",
        color="tab:blue",
        label="fitted center"
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_grid_and_refined_fit_near_poly(arr8, poly_xy, res_grid, res_final):
    """
    Show the coarse grid result and the final Powell-refined result side by side.
    """

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    _plot_triangle_fit_on_axis(
        axes[0],
        arr8,
        poly_xy,
        res_grid,
        title=(
            "Coarse fit\n"
            # f"loss={res_grid['loss']:.3f}, "
            # f"theta={np.rad2deg(res_grid['theta']):.1f}°"
        )
    )

    _plot_triangle_fit_on_axis(
        axes[1],
        arr8,
        poly_xy,
        res_final,
        title=(
            "Powell refinement\n"
            # f"loss={res_final['loss']:.3f}, "
            # f"theta={np.rad2deg(res_final['theta']):.1f}°"
        )
    )

    axes[1].legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.show()


# %%



# %%
def refine_all_particles_to_df(
    det_df,
    arr8,
    um_per_px=0.32,
    side_um=10.0,
    grid_thickness_um=0.5,
    grid_band_extra_um=0.5,
    grid_max_center_shift_um=8.0,
    grid_center_step_px=2.0,
    grid_theta_step_deg=10.0,
    grid_blur_sigma=0.5,
    grid_dark_top_fraction=1.0,
    grid_bright_top_fraction=1.0,
    grid_w_dark=1.0,
    grid_w_bright=0.2,
    grid_max_yolo_outside_fraction=0.05,
    bright_bad_threshold=middle_brightness_threshold,
    w_bright_fraction=2.0,

    refine_thickness_um=0.5,
    refine_center_window_px=5.0,
    refine_theta_window_deg=15.0,
    refine_side_window_px=2.0,
    refine_blur_sigma=0.5,
    refine_dark_top_fraction=1.0,
    refine_bright_top_fraction=0.2,
    refine_w_dark=1.0,
    refine_w_bright=1.0,
    refine_max_yolo_outside_fraction=0.05,
    refine_maxiter=120,

    verbose=True,

    show_fit_plots=False,
    plot_stage="side_by_side",   # "grid", "final", "both", or "side_by_side"
    plot_every=1,
):

    """
    Run grid fit + Powell refinement for all detections in det_df.

    Returns
    -------
    fit_df : pandas.DataFrame
        Dataframe containing old center, refined center,
        triangle outlines, fit parameters, and status.
    """

    results = []

    n_total = len(det_df)

    for i, row in det_df.iterrows():
        particle_id = row["particle_id"]

        if verbose:
            print("\n" + "=" * 60)
            print(f"Particle {particle_id}  ({i+1}/{n_total})")

        poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)
        do_plot = show_fit_plots and ((i % plot_every) == 0)

        try:
            # -------------------------
            # coarse grid fit
            # -------------------------
            res_grid = fit_triangle_prism_grid_search_near_poly(
                arr8,
                poly_xy,
                um_per_px=um_per_px,
                side_um=side_um,
                thickness_um=grid_thickness_um,
                band_extra_um=grid_band_extra_um,
                max_center_shift_um=grid_max_center_shift_um,
                center_step_px=grid_center_step_px,
                theta_step_deg=grid_theta_step_deg,
                blur_sigma=grid_blur_sigma,
                dark_top_fraction=grid_dark_top_fraction,
                bright_top_fraction=grid_bright_top_fraction,
                w_dark=grid_w_dark,
                w_bright=grid_w_bright,
                max_yolo_outside_fraction=grid_max_yolo_outside_fraction,
                bright_bad_threshold=bright_bad_threshold,
                w_bright_fraction=w_bright_fraction,
                verbose=False,
            )

            # -------------------------
            # precise refinement
            # -------------------------
            res_final = refine_triangle_prism_from_grid_result(
                arr8,
                poly_xy,
                res_grid,
                um_per_px=um_per_px,
                side_um=side_um,
                thickness_um=refine_thickness_um,
                refine_center_window_px=refine_center_window_px,
                refine_theta_window_deg=refine_theta_window_deg,
                refine_side_window_px=refine_side_window_px,
                blur_sigma=refine_blur_sigma,
                dark_top_fraction=refine_dark_top_fraction,
                bright_top_fraction=refine_bright_top_fraction,
                w_dark=refine_w_dark,
                w_bright=refine_w_bright,
                max_yolo_outside_fraction=refine_max_yolo_outside_fraction,
                maxiter=refine_maxiter,
                verbose=False,
            )

            if do_plot:
                if plot_stage == "grid":
                    plot_triangle_prism_fit_near_poly(arr8, poly_xy, res_grid)

                elif plot_stage == "final":
                    plot_triangle_prism_fit_near_poly(arr8, poly_xy, res_final)

                elif plot_stage == "both":
                    print("Showing coarse fit")
                    plot_triangle_prism_fit_near_poly(arr8, poly_xy, res_grid)

                    print("Showing Powell refinement")
                    plot_triangle_prism_fit_near_poly(arr8, poly_xy, res_final)

                elif plot_stage == "side_by_side":
                    plot_grid_and_refined_fit_near_poly(
                        arr8,
                        poly_xy,
                        res_grid,
                        res_final,
                    )

                else:
                    raise ValueError(
                        "plot_stage must be 'grid', 'final', 'both', or 'side_by_side'"
                    )

            results.append({
                "particle_id": particle_id,

                # original detection center
                "x_old": row["x"],
                "y_old": row["y"],

                # original YOLO polygon
                "poly_json": row["poly_json"],

                # refined center
                "x_refined": res_final["x_refined"],
                "y_refined": res_final["y_refined"],

                # fitted triangle parameters
                "theta_rad": res_final["theta"],
                "theta_deg": np.rad2deg(res_final["theta"]),
                "side_px": res_final["side_px"],
                "side_um": res_final["side_um"],
                "thickness_px": res_final["thickness_px"],

                # scores
                "loss": res_final["loss"],
                "dark_score": res_final["dark_score"],
                "bright_score": res_final["bright_score"],
                "mean_outline_distance": res_final["mean_outline_distance"],
                "yolo_outside_fraction": res_final["yolo_outside_fraction"],

                # fitted triangle outlines as JSON
                "outer_vertices_json": json.dumps(res_final["outer_vertices"].tolist()),
                "mid_vertices_json": json.dumps(res_final["mid_vertices"].tolist()),
                "inner_vertices_json": json.dumps(res_final["inner_vertices"].tolist()),

                # status
                "success": True,
                "error": "",
            })

            # if verbose:
            #     print("  success")
            #     print(f"  old center: ({row['x']:.2f}, {row['y']:.2f})")
            #     print(f"  new center: ({res_final['x_refined']:.2f}, {res_final['y_refined']:.2f})")
            #     print(f"  theta deg: {np.rad2deg(res_final['theta']):.2f}")
            #     print(f"  side px  : {res_final['side_px']:.2f}")

        except Exception as e:
            results.append({
                "particle_id": particle_id,

                "x_old": row["x"],
                "y_old": row["y"],

                "poly_json": row["poly_json"],

                "x_refined": np.nan,
                "y_refined": np.nan,

                "theta_rad": np.nan,
                "theta_deg": np.nan,
                "side_px": np.nan,
                "side_um": np.nan,
                "thickness_px": np.nan,

                "loss": np.nan,
                "dark_score": np.nan,
                "bright_score": np.nan,
                "mean_outline_distance": np.nan,
                "yolo_outside_fraction": np.nan,

                "outer_vertices_json": "",
                "mid_vertices_json": "",
                "inner_vertices_json": "",

                "success": False,
                "error": str(e),
            })

            if verbose:
                print("  FAILED:", e)

    fit_df = pd.DataFrame(results)
    return fit_df


# %%
# %%
def plot_old_and_new_centers(arr8, fit_df, draw_shift_lines=True):
    """
    Plot full image with:
    - old centers
    - refined centers
    - optional line between them
    """

    ok = fit_df["success"] == True
    df_ok = fit_df[ok].copy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr8, cmap="gray")

    # old centers
    ax.scatter(
        df_ok["x_old"],
        df_ok["y_old"],
        s=1,
        facecolors="none",
        edgecolors="red",
        linewidths=0.8,
        label="old centers"
    )

    # new centers
    ax.scatter(
        df_ok["x_refined"],
        df_ok["y_refined"],
        s=1,
        c="cyan",
        marker="x",
        linewidths=0.8,
        label="refined centers"
    )

    if draw_shift_lines:
        for _, row in df_ok.iterrows():
            ax.plot(
                [row["x_old"], row["x_refined"]],
                [row["y_old"], row["y_refined"]],
                linewidth=0.5,
                alpha=0.5,
                color="yellow"
            )

    ax.set_title(
        f"Old vs refined centers\n"
        f"successful fits: {len(df_ok)} / {len(fit_df)}"
    )
    ax.axis("off")
    ax.legend(loc="upper right")
    plt.show()
# %%
# %%
fit_df = refine_all_particles_to_df(
    det_df=det_df,
    arr8=arr8,
    um_per_px=mpp_real,
    side_um=10.0*physical_scale,

    # grid fit settings
    grid_thickness_um=0.5*physical_scale,
    grid_band_extra_um=0.2*physical_scale,
    grid_max_center_shift_um=8.0*physical_scale,
    grid_center_step_px=5.0*pixel_scale,
    grid_theta_step_deg=10.0,
    grid_blur_sigma=0.5,
    grid_dark_top_fraction=1.0,
    grid_bright_top_fraction=0.8,
    grid_w_dark=1.0,
    grid_w_bright=1.0,
    grid_max_yolo_outside_fraction=0.05,
    bright_bad_threshold=middle_brightness_threshold,
    w_bright_fraction=2.0,

    # refinement settings
    refine_thickness_um=0.5*physical_scale,
    refine_center_window_px=5.0*pixel_scale,
    refine_theta_window_deg=15.0,
    refine_side_window_px=2.0*pixel_scale,
    refine_blur_sigma=0.5,
    refine_dark_top_fraction=1.0,
    refine_bright_top_fraction=0.2,
    refine_w_dark=1.0,
    refine_w_bright=1.0,
    refine_max_yolo_outside_fraction=0.05,
    refine_maxiter=120,

    verbose=True,

    show_fit_plots=True,
    plot_stage="side_by_side",
    plot_every=1,
)
#%%
fit_df.head()
# %%
# %%
out_dir = FilePath(r"C:\particle_csv_files")
out_dir.mkdir(exist_ok=True)

out_fit_path = out_dir / f"{file_name}_refined_triangle_fits.csv"
fit_df.to_csv(out_fit_path, index=False)

print("Saved refined fits to:", out_fit_path)

# %%
plot_old_and_new_centers(arr8, fit_df, draw_shift_lines=True)
# %%
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_all_crops_with_centers(
    arr8,
    fit_df,
    crop_size=220,      # larger than before
    stride=180,         # smaller than crop_size = overlapping crops
    draw_shift_lines=True,
    ncols=4,
):
    """
    Plot the whole image as many crops, with old and refined centers overlaid.

    Parameters
    ----------
    arr8 : 2D image
    fit_df : DataFrame
        Must contain x_old, y_old, x_refined, y_refined
        and optionally 'success'.
    crop_size : int
        Size of each crop in pixels.
    stride : int
        Step between crop origins. If < crop_size, crops overlap.
    draw_shift_lines : bool
        Whether to draw yellow lines from old to new centers.
    ncols : int
        Number of subplot columns.
    """

    H, W = arr8.shape

    # keep only successful fits if available
    if "success" in fit_df.columns:
        df_plot = fit_df[fit_df["success"] == True].copy()
    else:
        df_plot = fit_df.copy()

    # make crop start positions
    x_starts = list(range(0, max(1, W - crop_size + 1), stride))
    y_starts = list(range(0, max(1, H - crop_size + 1), stride))

    # ensure last crop reaches image edge
    if len(x_starts) == 0 or x_starts[-1] != max(0, W - crop_size):
        x_starts.append(max(0, W - crop_size))
    if len(y_starts) == 0 or y_starts[-1] != max(0, H - crop_size):
        y_starts.append(max(0, H - crop_size))

    x_starts = sorted(set(x_starts))
    y_starts = sorted(set(y_starts))

    crop_boxes = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(W, x0 + crop_size)
            y1 = min(H, y0 + crop_size)
            crop_boxes.append((x0, x1, y0, y1))

    n_crops = len(crop_boxes)
    nrows = math.ceil(n_crops / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (x0, x1, y0, y1), idx in zip(axes, crop_boxes, range(n_crops)):
        crop = arr8[y0:y1, x0:x1]
        ax.imshow(crop, cmap="gray")

        # select particles whose old OR new center lies inside crop
        in_crop = (
            (
                (df_plot["x_old"] >= x0) & (df_plot["x_old"] < x1) &
                (df_plot["y_old"] >= y0) & (df_plot["y_old"] < y1)
            )
            |
            (
                (df_plot["x_refined"] >= x0) & (df_plot["x_refined"] < x1) &
                (df_plot["y_refined"] >= y0) & (df_plot["y_refined"] < y1)
            )
        )

        df_crop = df_plot[in_crop].copy()

        x_old_local = df_crop["x_old"] - x0
        y_old_local = df_crop["y_old"] - y0
        x_new_local = df_crop["x_refined"] - x0
        y_new_local = df_crop["y_refined"] - y0

        # old centers
        ax.scatter(
            x_old_local,
            y_old_local,
            s=14,
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
            label="old"
        )

        # new centers
        ax.scatter(
            x_new_local,
            y_new_local,
            s=14,
            c="cyan",
            marker="x",
            linewidths=0.8,
            label="new"
        )

        # shift lines
        if draw_shift_lines:
            for xo, yo, xn, yn in zip(x_old_local, y_old_local, x_new_local, y_new_local):
                ax.plot([xo, xn], [yo, yn], color="yellow", linewidth=0.5, alpha=0.6)

        ax.set_title(f"crop {idx+1}\nN={len(df_crop)}")
        ax.axis("off")

    # hide unused axes
    for ax in axes[n_crops:]:
        ax.axis("off")

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

# %%
plot_all_crops_with_centers(
    arr8,
    fit_df,
    crop_size=220,   # bigger crops
    stride=180,      # overlap a bit
    draw_shift_lines=True,
    ncols=4,
)
# %%
