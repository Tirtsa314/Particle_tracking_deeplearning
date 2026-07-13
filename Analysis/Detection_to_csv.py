
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

model = YOLO(r"C:\Users\DenHaan\Downloads\best_09-04_20epoch_ratio1.pt")

#%% 
"""Uitlezen nd2:
"""


nd2_path = r"c:\Users\Public\Hydrazine 003(good).nd2"
file_name = Path(nd2_path).stem
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

mpp_real = 0.32
side_real_um = 10.0

train_side_px = side_train_um / mpp_train
real_side_px = side_real_um / mpp_real

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

                min_dist = 10
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

out_dir = Path(r"C:\particle_csv_files")
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.optimize import minimize


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


def polygon_to_mask(shape, poly):
    """
    Convert polygon coordinates to boolean mask.
    poly is in local/crop coordinates.
    """
    H, W = shape
    yy, xx = np.mgrid[:H, :W]

    points = np.column_stack([
        xx.ravel() + 0.5,
        yy.ravel() + 0.5
    ])

    path = Path(poly)
    mask = path.contains_points(points).reshape(H, W)

    return mask


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
    Make a triangular prism-like 2D band.

    outer triangle = full particle size
    inner triangle = inset by thickness
    band = outer - inner

    Also returns outer half and inner half separately.
    """

    side_mid = side_px - 2 * np.sqrt(3) * (thickness_px / 2)
    side_inner = side_px - 2 * np.sqrt(3) * thickness_px

    if side_mid <= 1 or side_inner <= 1:
        return None

    outer_verts = triangle_vertices_from_centroid(cx, cy, side_px, theta)
    mid_verts = triangle_vertices_from_centroid(cx, cy, side_mid, theta)
    inner_verts = triangle_vertices_from_centroid(cx, cy, side_inner, theta)

    outer_mask = polygon_to_mask(shape, outer_verts)
    mid_mask = polygon_to_mask(shape, mid_verts)
    inner_mask = polygon_to_mask(shape, inner_verts)

    full_band = outer_mask & (~inner_mask)
    outer_band = outer_mask & (~mid_mask)
    inner_band = mid_mask & (~inner_mask)

    return outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts

def fit_triangle_prism_to_dark_edges_near_poly(
    arr8,
    poly_xy,
    um_per_px=0.32,
    side_um=10.0,
    thickness_um=2.0,
    max_center_shift_um=3.0,
    max_edge_dist_um=2.0,
    blur_sigma=1.0,
    dark_top_fraction=0.20,
    w_dist=0.8,
    w_far=1.5,
    w_center=0.05,
    side_slack_px=3.0,      # NEW
    allow_smaller=False,    # NEW
    max_far_fraction=0.45,
    n_theta_starts=12,
    maxiter=120,
):
    """
    Fit a fixed-size triangular prism/band to dark pixels near a YOLO polygon outline.

    Parameters
    ----------
    arr8 : 2D ndarray
        Grayscale image.
    poly_xy : ndarray, shape (N, 2)
        YOLO polygon outline in original image coordinates.
    um_per_px : float
        Microns per pixel.
    side_um : float
        Outer triangle side length in microns.
    thickness_um : float
        Triangle band thickness in microns.
    max_center_shift_um : float
        Maximum allowed movement from YOLO polygon centroid.
    max_edge_dist_um : float
        Fitted triangle band should stay near the YOLO polygon outline.
    blur_sigma : float
        Smoothing before darkness scoring.
    w_outer : float
        Weight for darkness in outer half of triangle band.
    w_inner : float
        Weight for darkness in inner half of triangle band.
    w_dist : float
        Penalty weight for being too far from polygon outline.
    w_far : float
        Penalty weight for fraction of band far from polygon outline.
    w_center : float
        Penalty weight for moving away from polygon centroid.
    max_far_fraction : float
        If too much of the band is far from the polygon outline, reject.
    n_theta_starts : int
        Number of starting angles for optimizer.
    """

    arr8 = np.asarray(arr8) #makes sure image is np array
    poly_xy = np.asarray(poly_xy, dtype=float) #makes sure polygon is np array

    if arr8.ndim != 2:
        raise ValueError("arr8 must be a 2D grayscale image")

    if poly_xy.ndim != 2 or poly_xy.shape[1] != 2:
        raise ValueError("poly_xy must have shape (N, 2)")

    if len(poly_xy) < 3:
        raise ValueError("poly_xy must contain at least 3 points")

    side_px_nominal = side_um / um_per_px #conversion side length to pixels

    if allow_smaller:
        side_px_min = max(1.0, side_px_nominal - side_slack_px) #minimal side length
    else:
        side_px_min = side_px_nominal

    side_px_max = side_px_nominal + side_slack_px #max side length

    thickness_px = thickness_um / um_per_px
    max_center_shift_px = max_center_shift_um / um_per_px
    max_edge_dist_px = max_edge_dist_um / um_per_px

    # Initial center from YOLO polygon
    x0_global, y0_global = polygon_centroid(poly_xy)

    # Crop around polygon for speed
    x_min0 = int(np.floor(poly_xy[:, 0].min()))
    x_max0 = int(np.ceil(poly_xy[:, 0].max()))
    y_min0 = int(np.floor(poly_xy[:, 1].min()))
    y_max0 = int(np.ceil(poly_xy[:, 1].max()))

    pad = int(np.ceil(
        side_px_max / 2
        + thickness_px
        + max_center_shift_px
        + max_edge_dist_px
        + 8
    ))

    H, W = arr8.shape

    x_min = max(0, x_min0 - pad)
    x_max = min(W, x_max0 + pad + 1)
    y_min = max(0, y_min0 - pad)
    y_max = min(H, y_max0 + pad + 1)

    img_crop = arr8[y_min:y_max, x_min:x_max].astype(float)
    crop_shape = img_crop.shape

    # Convert polygon and center to crop coordinates
    poly_local = poly_xy.copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min

    x0 = x0_global - x_min
    y0 = y0_global - y_min

    # Make polygon mask and polygon outline distance map
    poly_mask = polygon_to_mask(crop_shape, poly_local)

    poly_eroded = binary_erosion(poly_mask)
    poly_outline = poly_mask ^ poly_eroded

    if poly_outline.sum() == 0:
        poly_outline = poly_mask.copy()

    dist_to_outline = distance_transform_edt(~poly_outline)

    # Darkness image: dark pixels get high score
    img_smooth = gaussian_filter(img_crop, sigma=blur_sigma)

    lo, hi = np.percentile(img_smooth, [1, 99])
    img_norm = np.clip((img_smooth - lo) / (hi - lo + 1e-9), 0, 1)

    darkness = 1.0 - img_norm

    def objective(params):
        cx, cy, theta, side_px = params

        out = triangle_prism_band_masks(
            crop_shape,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta,
            thickness_px=thickness_px,
        )

        if out is None:
            return 1e6

        outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

        if full_band.sum() < 10:
            return 1e6

        # focus on the darkest part of the band
        band_vals = darkness[full_band]
        k = max(5, int(dark_top_fraction * len(band_vals)))
        top_vals = np.partition(band_vals, -k)[-k:]
        dark_score = np.mean(top_vals)

        # distance to YOLO outline
        d = dist_to_outline[full_band]
        far_fraction = np.mean(d > max_edge_dist_px)

        if far_fraction > max_far_fraction:
            return 1e6 + 1000 * far_fraction

        dist_penalty = np.mean(
            np.clip(d / (max_edge_dist_px + 1e-9) - 1.0, 0, None) ** 2
        )

        center_shift2 = (cx - x0)**2 + (cy - y0)**2
        center_penalty = center_shift2 / (max_center_shift_px**2 + 1e-9)

        loss = (
            -dark_score
            + w_dist * dist_penalty
            + w_far * far_fraction
            + w_center * center_penalty
        )

        return loss

    bounds = [
        (x0 - max_center_shift_px, x0 + max_center_shift_px),
        (y0 - max_center_shift_px, y0 + max_center_shift_px),
        (0, 2*np.pi/3),
        (side_px_min, side_px_max),
    ]

    best_res = None

    size_starts = np.linspace(side_px_min, side_px_max, 3)

    for theta0 in np.linspace(0, 2*np.pi/3, n_theta_starts, endpoint=False):
        for side0 in size_starts:
            res = minimize(
                objective,
                x0=np.array([x0, y0, theta0, side0]),
                method="Powell",
                bounds=bounds,
                options={
                    "maxiter": maxiter,
                    "xtol": 1e-2,
                    "ftol": 1e-3,
                },
            )

            if best_res is None or res.fun < best_res.fun:
                best_res = res

    cx_best, cy_best, theta_best, side_px_best = best_res.x

    out = triangle_prism_band_masks(
        crop_shape,
        cx=cx_best,
        cy=cy_best,
        side_px=side_px_best,
        theta=theta_best,
        thickness_px=thickness_px,
    )

    if out is None:
        raise RuntimeError("Final triangle band construction failed")

    outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

    # Final darkness score: focus on blackest part of the fitted band
    band_vals = darkness[full_band]

    k = max(5, int(dark_top_fraction * len(band_vals)))
    top_vals = np.partition(band_vals, -k)[-k:]

    dark_score = np.mean(top_vals)

    # Still keep these for diagnostics only
    outer_dark = darkness[outer_band].mean()
    inner_dark = darkness[inner_band].mean()

    d_final = dist_to_outline[full_band]
    mean_outline_distance = d_final.mean()
    far_fraction = np.mean(d_final > max_edge_dist_px)

    # Convert vertices back to global coordinates
    outer_global = outer_verts.copy()
    mid_global = mid_verts.copy()
    inner_global = inner_verts.copy()

    for verts in [outer_global, mid_global, inner_global]:
        verts[:, 0] += x_min
        verts[:, 1] += y_min

    x_refined = cx_best + x_min
    y_refined = cy_best + y_min

    return {
        "x_refined": x_refined,
        "y_refined": y_refined,
        "theta": theta_best,
        "side_px": side_px_best,
        "side_um": side_px_best * um_per_px,
        "side_px_nominal": side_px_nominal,
        "thickness_px": thickness_px,
        "loss": best_res.fun,
        "dark_score": dark_score,
        "outer_dark": outer_dark,
        "inner_dark": inner_dark,
        "mean_outline_distance": mean_outline_distance,
        "far_fraction": far_fraction,
        "outer_vertices": outer_global,
        "mid_vertices": mid_global,
        "inner_vertices": inner_global,
        "crop_box": (y_min, y_max, x_min, x_max),
        "poly_local": poly_local,
        "poly_mask_local": poly_mask,
        "poly_outline_local": poly_outline,
        "darkness_crop": darkness,
        "outer_band_local": outer_band,
        "inner_band_local": inner_band,
        "full_band_local": full_band,
    }

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

    ax.set_title(
        f"Triangle prism fit\n"
        f"dark_score={res['dark_score']:.3f}, "
        f"mean outline dist={res['mean_outline_distance']:.2f}px, "
        f"far={res['far_fraction']:.2f}"
    )

    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.show()


# %%
for particle_id in range(0,20):
    # particle_id = 40
    row = det_df[det_df["particle_id"] == particle_id].iloc[0]

    poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)

    res = fit_triangle_prism_to_dark_edges_near_poly(
        arr8,
        poly_xy,
        um_per_px=mpp_real,
        side_um=10.0,
        thickness_um=2.0,
        max_center_shift_um=5.0,
        max_edge_dist_um=1.5,
        dark_top_fraction=0.30,
        w_dist=0.3,
        w_far=0.3,
        w_center=0.1,
        side_slack_px=4.0,      # allow up to 3 px larger
        allow_smaller=False,    # only allow larger, not smaller
        n_theta_starts=12,
    )

    print("fitted side px:", res["side_px"])
    print("fitted side um:", res["side_um"])

    plot_triangle_prism_fit_near_poly(arr8, poly_xy, res)
# %%

def fit_triangle_prism_to_dark_pixels_near_poly(
    arr8,
    poly_xy,
    um_per_px=0.32,
    side_um=10.0,
    thickness_um=2.0,
    max_center_shift_um=3.0,
    blur_sigma=1.0,

    # triangle size fitting
    side_slack_px=5.0,
    allow_smaller=False,

    # dark-pixel mask settings
    dark_search_dist_um=5.0,
    dark_pixel_fraction=0.20,

    # fitting objective settings
    max_dark_dist_um=1.0,
    band_dark_fraction=0.70,
    w_dark_dist=0.15,
    w_far_dark=0.8,
    w_center=0.05,

    n_theta_starts=18,
    maxiter=150,
):
    """
    Fit triangular prism/band to actual dark pixels near a YOLO polygon.

    Difference from previous version:
    - YOLO polygon is NOT used as the outline target.
    - A dark-pixel mask is built near the YOLO polygon.
    - The fitted triangle band is attracted to those dark pixels.
    """

    arr8 = np.asarray(arr8)
    poly_xy = np.asarray(poly_xy, dtype=float)

    if arr8.ndim != 2:
        raise ValueError("arr8 must be a 2D grayscale image")

    if poly_xy.ndim != 2 or poly_xy.shape[1] != 2:
        raise ValueError("poly_xy must have shape (N, 2)")

    if len(poly_xy) < 3:
        raise ValueError("poly_xy must contain at least 3 points")

    # ----------------------------
    # Convert physical parameters
    # ----------------------------
    side_px_nominal = side_um / um_per_px

    if allow_smaller:
        side_px_min = max(1.0, side_px_nominal - side_slack_px)
    else:
        side_px_min = side_px_nominal

    side_px_max = side_px_nominal + side_slack_px

    thickness_px = thickness_um / um_per_px
    max_center_shift_px = max_center_shift_um / um_per_px
    dark_search_dist_px = dark_search_dist_um / um_per_px
    max_dark_dist_px = max_dark_dist_um / um_per_px

    # ----------------------------
    # Initial center from YOLO polygon
    # ----------------------------
    x0_global, y0_global = polygon_centroid(poly_xy)

    x_min0 = int(np.floor(poly_xy[:, 0].min()))
    x_max0 = int(np.ceil(poly_xy[:, 0].max()))
    y_min0 = int(np.floor(poly_xy[:, 1].min()))
    y_max0 = int(np.ceil(poly_xy[:, 1].max()))

    H, W = arr8.shape

    # Bigger crop because triangle may move and grow
    pad = int(np.ceil(
        side_px_max / np.sqrt(3)
        + thickness_px
        + max_center_shift_px
        + dark_search_dist_px
        + 10
    ))

    x_min = max(0, x_min0 - pad)
    x_max = min(W, x_max0 + pad + 1)
    y_min = max(0, y_min0 - pad)
    y_max = min(H, y_max0 + pad + 1)

    img_crop = arr8[y_min:y_max, x_min:x_max].astype(float)
    crop_shape = img_crop.shape

    # Local polygon coordinates
    poly_local = poly_xy.copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min

    x0 = x0_global - x_min
    y0 = y0_global - y_min

    # ----------------------------
    # Make YOLO rough region
    # ----------------------------
    poly_mask = polygon_to_mask(crop_shape, poly_local)

    # Region around YOLO polygon where we allow dark pixels to matter
    dist_to_poly_mask = distance_transform_edt(~poly_mask)
    search_region = dist_to_poly_mask <= dark_search_dist_px

    if search_region.sum() < 10:
        search_region = np.ones(crop_shape, dtype=bool)

    # ----------------------------
    # Build darkness image
    # ----------------------------
    img_smooth = gaussian_filter(img_crop, sigma=blur_sigma)

    lo, hi = np.percentile(img_smooth, [1, 99])
    img_norm = np.clip((img_smooth - lo) / (hi - lo + 1e-9), 0, 1)

    # high value = dark
    darkness = 1.0 - img_norm

    # ----------------------------
    # Build dark-pixel target mask
    # ----------------------------
    darkness_inside_search = darkness[search_region]

    threshold = np.quantile(
        darkness_inside_search,
        1.0 - dark_pixel_fraction
    )

    dark_mask = search_region & (darkness >= threshold)

    # fallback if thresholding somehow gives too few pixels
    if dark_mask.sum() < 10:
        threshold = np.quantile(darkness, 1.0 - dark_pixel_fraction)
        dark_mask = darkness >= threshold

    # Distance to actual dark pixels
    dist_to_dark = distance_transform_edt(~dark_mask)

    # ----------------------------
    # Objective
    # ----------------------------
    def objective(params):
        cx, cy, theta, side_px = params

        out = triangle_prism_band_masks(
            crop_shape,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta,
            thickness_px=thickness_px,
        )

        if out is None:
            return 1e6

        outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

        if full_band.sum() < 10:
            return 1e6

        # Darkness score over a large part of the band.
        # This avoids fitting only one black corner.
        band_vals = darkness[full_band]

        if band_dark_fraction >= 1.0:
            dark_score = np.mean(band_vals)
        else:
            k = max(5, int(band_dark_fraction * len(band_vals)))
            top_vals = np.partition(band_vals, -k)[-k:]
            dark_score = np.mean(top_vals)

        # Distance of triangle band to actual dark pixels
        d = dist_to_dark[full_band]

        dark_dist_penalty = np.mean(
            np.clip(d / (max_dark_dist_px + 1e-9), 0, 3) ** 2
        )

        far_dark_fraction = np.mean(d > max_dark_dist_px)

        # Weak center prior, only to stop it jumping to another nearby particle
        center_shift2 = (cx - x0)**2 + (cy - y0)**2
        center_penalty = center_shift2 / (max_center_shift_px**2 + 1e-9)

        loss = (
            -dark_score
            + w_dark_dist * dark_dist_penalty
            + w_far_dark * far_dark_fraction
            + w_center * center_penalty
        )

        return loss

    bounds = [
        (x0 - max_center_shift_px, x0 + max_center_shift_px),
        (y0 - max_center_shift_px, y0 + max_center_shift_px),
        (0, 2*np.pi/3),
        (side_px_min, side_px_max),
    ]

    best_res = None

    size_starts = np.linspace(side_px_min, side_px_max, 4)

    for theta0 in np.linspace(0, 2*np.pi/3, n_theta_starts, endpoint=False):
        for side0 in size_starts:
            res = minimize(
                objective,
                x0=np.array([x0, y0, theta0, side0]),
                method="Powell",
                bounds=bounds,
                options={
                    "maxiter": maxiter,
                    "xtol": 1e-2,
                    "ftol": 1e-3,
                },
            )

            if best_res is None or res.fun < best_res.fun:
                best_res = res

    cx_best, cy_best, theta_best, side_px_best = best_res.x

    out = triangle_prism_band_masks(
        crop_shape,
        cx=cx_best,
        cy=cy_best,
        side_px=side_px_best,
        theta=theta_best,
        thickness_px=thickness_px,
    )

    if out is None:
        raise RuntimeError("Final triangle band construction failed")

    outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

    # final diagnostics
    band_vals = darkness[full_band]

    if band_dark_fraction >= 1.0:
        dark_score = np.mean(band_vals)
    else:
        k = max(5, int(band_dark_fraction * len(band_vals)))
        top_vals = np.partition(band_vals, -k)[-k:]
        dark_score = np.mean(top_vals)

    d_final = dist_to_dark[full_band]
    mean_dark_distance = d_final.mean()
    far_dark_fraction = np.mean(d_final > max_dark_dist_px)

    outer_dark = darkness[outer_band].mean()
    inner_dark = darkness[inner_band].mean()

    # Convert vertices back to global coordinates
    outer_global = outer_verts.copy()
    mid_global = mid_verts.copy()
    inner_global = inner_verts.copy()

    for verts in [outer_global, mid_global, inner_global]:
        verts[:, 0] += x_min
        verts[:, 1] += y_min

    x_refined = cx_best + x_min
    y_refined = cy_best + y_min

    return {
        "x_refined": x_refined,
        "y_refined": y_refined,
        "theta": theta_best,
        "side_px": side_px_best,
        "side_um": side_px_best * um_per_px,
        "side_px_nominal": side_px_nominal,
        "side_px_min": side_px_min,
        "side_px_max": side_px_max,
        "thickness_px": thickness_px,
        "loss": best_res.fun,

        "dark_score": dark_score,
        "outer_dark": outer_dark,
        "inner_dark": inner_dark,
        "mean_dark_distance": mean_dark_distance,
        "far_dark_fraction": far_dark_fraction,
        "dark_threshold": threshold,

        "outer_vertices": outer_global,
        "mid_vertices": mid_global,
        "inner_vertices": inner_global,

        "crop_box": (y_min, y_max, x_min, x_max),
        "poly_local": poly_local,
        "poly_mask_local": poly_mask,
        "search_region_local": search_region,
        "dark_mask_local": dark_mask,
        "darkness_crop": darkness,
        "dist_to_dark_crop": dist_to_dark,

        "outer_band_local": outer_band,
        "inner_band_local": inner_band,
        "full_band_local": full_band,
    }
# %%
def plot_triangle_prism_fit_to_dark_pixels(
    arr8,
    poly_xy,
    res,
    alpha_poly=0.15,
    alpha_dark=0.35,
    show_dark_pixels=True,
):
    """
    Visualization for fit_triangle_prism_to_dark_pixels_near_poly().

    Shows:
    - image crop
    - YOLO polygon outline/fill
    - dark pixels used as target
    - fitted outer/mid/inner triangle
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
    poly_overlay = np.zeros((*poly_mask.shape, 4), dtype=float)
    poly_overlay[..., 0] = 1.0
    poly_overlay[..., 3] = alpha_poly * poly_mask.astype(float)
    ax.imshow(poly_overlay)

    # Dark pixels used for fitting
    if show_dark_pixels:
        dark_mask = res["dark_mask_local"]
        dark_overlay = np.zeros((*dark_mask.shape, 4), dtype=float)
        dark_overlay[..., 1] = 1.0
        dark_overlay[..., 2] = 1.0
        dark_overlay[..., 3] = alpha_dark * dark_mask.astype(float)
        ax.imshow(dark_overlay)

    # YOLO polygon outline
    ax.plot(
        poly_closed[:, 0],
        poly_closed[:, 1],
        linewidth=2,
        label="YOLO poly outline"
    )

    # fitted triangular prism
    ax.plot(
        outer_c[:, 0],
        outer_c[:, 1],
        linewidth=2,
        label="outer fitted triangle"
    )

    ax.plot(
        mid_c[:, 0],
        mid_c[:, 1],
        linewidth=2,
        linestyle="--",
        label="middle"
    )

    ax.plot(
        inner_c[:, 0],
        inner_c[:, 1],
        linewidth=2,
        label="inner fitted triangle"
    )

    ax.scatter(
        x_ref,
        y_ref,
        s=80,
        marker="x",
        label="fitted center"
    )

    ax.set_title(
        f"Triangle prism fit to dark pixels\n"
        f"dark_score={res['dark_score']:.3f}, "
        f"mean dark dist={res['mean_dark_distance']:.2f}px, "
        f"far_dark={res['far_dark_fraction']:.2f}\n"
        f"side={res['side_px']:.2f}px = {res['side_um']:.2f} µm"
    )

    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.show()
# %%


# for particle_id in range(0,200):
#     row = det_df[det_df["particle_id"] == particle_id].iloc[0]

#     poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)

#     res_dark = fit_triangle_prism_to_dark_pixels_near_poly(
#         arr8,
#         poly_xy,
#         um_per_px=mpp_real,
#         side_um=10.0,
#         thickness_um=2.0,

#         max_center_shift_um=3.0,

#         side_slack_px=5.0,
#         allow_smaller=False,

#         dark_search_dist_um=5.0,
#         dark_pixel_fraction=0.20,

#         max_dark_dist_um=1.0,
#         band_dark_fraction=0.70,

#         w_dark_dist=0.15,
#         w_far_dark=0.8,
#         w_center=0.05,

#         n_theta_starts=18,
#     )

#     print("fitted side px:", res_dark["side_px"])
#     print("fitted side um:", res_dark["side_um"])
#     print("mean dark distance px:", res_dark["mean_dark_distance"])
#     print("far dark fraction:", res_dark["far_dark_fraction"])

#     plot_triangle_prism_fit_to_dark_pixels(arr8, poly_xy, res_dark)
# %%

def points_inside_equilateral_triangle(points, cx, cy, side_px, theta, margin_px=0.0):
    """
    Check whether points are inside an equilateral triangle.

    margin_px > 0 means points must be safely inside.
    margin_px < 0 allows a small outside tolerance.
    """
    points = np.asarray(points, dtype=float)
    pts = points - np.array([cx, cy], dtype=float)

    R = side_px / np.sqrt(3)

    angles = theta + np.array([0, 2*np.pi/3, 4*np.pi/3])
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])

    proj = pts @ dirs.T

    inside = np.all(proj >= (-0.5 * R + margin_px), axis=1)

    return inside


def required_side_to_contain_points(points, cx, cy, theta, margin_px=1.0):
    """
    For a fixed center and orientation, compute the minimum side length needed
    for an equilateral triangle to contain all points.
    """
    points = np.asarray(points, dtype=float)
    pts = points - np.array([cx, cy], dtype=float)

    angles = theta + np.array([0, 2*np.pi/3, 4*np.pi/3])
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])

    proj = pts @ dirs.T
    min_proj = np.min(proj, axis=0)

    # condition: proj >= -R/2 + margin
    # so R >= -2 * (min_proj - margin)
    R_required = np.max(np.maximum(0.0, -2.0 * (min_proj - margin_px)))

    side_required = np.sqrt(3) * R_required

    return side_required

def polygon_outline_band_mask(shape, verts, width_px=2.0):
    """
    Make a thin band around a polygon outline.
    Used to test whether the fitted triangle edge lies on dark pixels.
    """
    mask = polygon_to_mask(shape, verts)

    eroded = binary_erosion(mask, iterations=1)
    outline = mask & (~eroded)

    if outline.sum() == 0:
        outline = mask.copy()

    outline_band = distance_transform_edt(~outline) <= width_px

    return outline_band

def initial_triangle_containing_poly(
    poly_xy,
    cx,
    cy,
    side_px_nominal,
    side_px_max,
    margin_px=1.0,
    n_theta=90,
):
    """
    Find orientation and size for an initial triangle centered at cx,cy
    that tries to contain the YOLO polygon with minimal side length.
    """
    theta_grid = np.linspace(0, 2*np.pi/3, n_theta, endpoint=False)

    required_sides = np.array([
        required_side_to_contain_points(
            poly_xy,
            cx=cx,
            cy=cy,
            theta=theta,
            margin_px=margin_px,
        )
        for theta in theta_grid
    ])

    best_i = np.argmin(required_sides)
    theta0 = theta_grid[best_i]
    required_side_px = required_sides[best_i]

    side0 = max(side_px_nominal, required_side_px)
    side0 = min(side0, side_px_max)

    inside = points_inside_equilateral_triangle(
        poly_xy,
        cx=cx,
        cy=cy,
        side_px=side0,
        theta=theta0,
        margin_px=0.0,
    )

    return {
        "theta0": theta0,
        "side0": side0,
        "required_side_px": required_side_px,
        "start_inside_fraction": inside.mean(),
    }
# %%
# def fit_triangle_prism_local_dark_edges_near_poly(
#     arr8,
#     poly_xy,
#     um_per_px=0.32,
#     side_um=10.0,
#     thickness_um=2.0,

#     # initial triangle / size
#     side_slack_px=6.0,
#     contain_margin_px=1.0,

#     # local optimization limits
#     max_center_shift_um=1.5,
#     max_rotation_deg=15.0,
#     side_local_slack_px=2.0,
#     allow_shrink_from_start=False,

#     # dark-pixel target around initial triangle band
#     blur_sigma=1.0,
#     dark_search_dist_px=5.0,
#     dark_pixel_fraction=0.25,

#     # objective
#     max_dark_dist_px=2.0,
#     band_dark_fraction=0.80,
#     w_dark_dist=0.20,
#     w_far_dark=1.0,
#     w_center=0.10,
#     w_rotation=0.05,
#     w_contain=0.5,
#     max_poly_outside_fraction=0.25,

#     maxiter=80,
# ):
#     """
#     Faster, more conservative fit.

#     Steps:
#     1. Put triangle at YOLO polygon centroid.
#     2. Choose initial orientation/size so YOLO polygon is inside.
#     3. Build dark-pixel target near that initial triangle band.
#     4. Optimize only locally: small shift, small rotation, small size change.
#     """

#     arr8 = np.asarray(arr8)
#     poly_xy = np.asarray(poly_xy, dtype=float)

#     if arr8.ndim != 2:
#         raise ValueError("arr8 must be 2D grayscale")

#     if poly_xy.ndim != 2 or poly_xy.shape[1] != 2:
#         raise ValueError("poly_xy must have shape (N, 2)")

#     if len(poly_xy) < 3:
#         raise ValueError("poly_xy must contain at least 3 points")

#     # -------------------------
#     # Physical to pixel units
#     # -------------------------
#     side_px_nominal = side_um / um_per_px
#     side_px_max = side_px_nominal + side_slack_px

#     thickness_px = thickness_um / um_per_px
#     max_center_shift_px = max_center_shift_um / um_per_px
#     max_rotation_rad = np.deg2rad(max_rotation_deg)

#     # -------------------------
#     # Initial center from YOLO polygon
#     # -------------------------
#     x0_global, y0_global = polygon_centroid(poly_xy)

#     init = initial_triangle_containing_poly(
#         poly_xy,
#         cx=x0_global,
#         cy=y0_global,
#         side_px_nominal=side_px_nominal,
#         side_px_max=side_px_max,
#         margin_px=contain_margin_px,
#         n_theta=90,
#     )

#     theta0 = init["theta0"]
#     side0 = init["side0"]

#     # -------------------------
#     # Crop around particle
#     # -------------------------
#     x_min0 = int(np.floor(poly_xy[:, 0].min()))
#     x_max0 = int(np.ceil(poly_xy[:, 0].max()))
#     y_min0 = int(np.floor(poly_xy[:, 1].min()))
#     y_max0 = int(np.ceil(poly_xy[:, 1].max()))

#     H, W = arr8.shape

#     pad = int(np.ceil(
#         side_px_max / np.sqrt(3)
#         + thickness_px
#         + max_center_shift_px
#         + dark_search_dist_px
#         + 10
#     ))

#     x_min = max(0, x_min0 - pad)
#     x_max = min(W, x_max0 + pad + 1)
#     y_min = max(0, y_min0 - pad)
#     y_max = min(H, y_max0 + pad + 1)

#     img_crop = arr8[y_min:y_max, x_min:x_max].astype(float)
#     crop_shape = img_crop.shape

#     poly_local = poly_xy.copy()
#     poly_local[:, 0] -= x_min
#     poly_local[:, 1] -= y_min

#     x0 = x0_global - x_min
#     y0 = y0_global - y_min

#     # -------------------------
#     # YOLO mask only for visualization / containment
#     # -------------------------
#     poly_mask = polygon_to_mask(crop_shape, poly_local)

#     # -------------------------
#     # Initial triangle band
#     # -------------------------
#     init_out = triangle_prism_band_masks(
#         crop_shape,
#         cx=x0,
#         cy=y0,
#         side_px=side0,
#         theta=theta0,
#         thickness_px=thickness_px,
#     )

#     if init_out is None:
#         raise RuntimeError("Initial triangle band construction failed")

#     (
#         init_outer_band,
#         init_inner_band,
#         init_full_band,
#         init_outer_verts,
#         init_mid_verts,
#         init_inner_verts,
#     ) = init_out

#     # Use only dark pixels near the initial triangle band.
#     # This prevents the fit from jumping to neighboring particles.
#     dist_to_init_band = distance_transform_edt(~init_full_band)
#     search_region = dist_to_init_band <= dark_search_dist_px

#     if search_region.sum() < 10:
#         search_region = np.ones(crop_shape, dtype=bool)

#     # -------------------------
#     # Darkness image
#     # -------------------------
#     img_smooth = gaussian_filter(img_crop, sigma=blur_sigma)

#     lo, hi = np.percentile(img_smooth, [1, 99])
#     img_norm = np.clip((img_smooth - lo) / (hi - lo + 1e-9), 0, 1)
#     darkness = 1.0 - img_norm

#     darkness_inside_search = darkness[search_region]

#     threshold = np.quantile(
#         darkness_inside_search,
#         1.0 - dark_pixel_fraction
#     )

#     dark_mask = search_region & (darkness >= threshold)

#     if dark_mask.sum() < 10:
#         threshold = np.quantile(darkness, 1.0 - dark_pixel_fraction)
#         dark_mask = darkness >= threshold

#     dist_to_dark = distance_transform_edt(~dark_mask)

#     # -------------------------
#     # Side bounds around initial side
#     # -------------------------
#     if allow_shrink_from_start:
#         side_min = max(side_px_nominal, side0 - side_local_slack_px)
#     else:
#         side_min = side0

#     side_max = min(side_px_max, side0 + side_local_slack_px)

#     if side_max < side_min:
#         side_max = side_min

#     # -------------------------
#     # Objective
#     # -------------------------
#     def objective(params):
#         dx, dy, dtheta, side_px = params

#         cx = x0 + dx
#         cy = y0 + dy
#         theta = theta0 + dtheta

#         out = triangle_prism_band_masks(
#             crop_shape,
#             cx=cx,
#             cy=cy,
#             side_px=side_px,
#             theta=theta,
#             thickness_px=thickness_px,
#         )

#         if out is None:
#             return 1e6

#         outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

#         if full_band.sum() < 10:
#             return 1e6

#         # Darkness score over much of the band.
#         # Higher band_dark_fraction = less chance of fitting only one corner.
#         band_vals = darkness[full_band]

#         if band_dark_fraction >= 1.0:
#             dark_score = np.mean(band_vals)
#         else:
#             k = max(5, int(band_dark_fraction * len(band_vals)))
#             top_vals = np.partition(band_vals, -k)[-k:]
#             dark_score = np.mean(top_vals)

#         # Distance to local dark pixels
#         d = dist_to_dark[full_band]

#         dark_dist_penalty = np.mean(
#             np.clip(d / (max_dark_dist_px + 1e-9), 0, 3) ** 2
#         )

#         far_dark_fraction = np.mean(d > max_dark_dist_px)

#         # Keep near YOLO center
#         center_penalty = (dx**2 + dy**2) / (max_center_shift_px**2 + 1e-9)

#         # Keep near initial orientation
#         rotation_penalty = (dtheta / (max_rotation_rad + 1e-9))**2

#         # Do not let the fitted triangle drift too far away from YOLO polygon
#         inside = points_inside_equilateral_triangle(
#             poly_local,
#             cx=cx,
#             cy=cy,
#             side_px=side_px,
#             theta=theta,
#             margin_px=-1.0,
#         )

#         outside_fraction = 1.0 - inside.mean()

#         if outside_fraction > max_poly_outside_fraction:
#             return 1e6 + 1000 * outside_fraction

#         loss = (
#             -dark_score
#             + w_dark_dist * dark_dist_penalty
#             + w_far_dark * far_dark_fraction
#             + w_center * center_penalty
#             + w_rotation * rotation_penalty
#             + w_contain * outside_fraction
#         )

#         return loss

#     bounds = [
#         (-max_center_shift_px, max_center_shift_px),
#         (-max_center_shift_px, max_center_shift_px),
#         (-max_rotation_rad, max_rotation_rad),
#         (side_min, side_max),
#     ]

#     res = minimize(
#         objective,
#         x0=np.array([0.0, 0.0, 0.0, side0]),
#         method="Powell",
#         bounds=bounds,
#         options={
#             "maxiter": maxiter,
#             "xtol": 1e-2,
#             "ftol": 1e-3,
#         },
#     )

#     dx_best, dy_best, dtheta_best, side_px_best = res.x

#     cx_best = x0 + dx_best
#     cy_best = y0 + dy_best
#     theta_best = theta0 + dtheta_best

#     final_out = triangle_prism_band_masks(
#         crop_shape,
#         cx=cx_best,
#         cy=cy_best,
#         side_px=side_px_best,
#         theta=theta_best,
#         thickness_px=thickness_px,
#     )

#     if final_out is None:
#         raise RuntimeError("Final triangle band construction failed")

#     (
#         outer_band,
#         inner_band,
#         full_band,
#         outer_verts,
#         mid_verts,
#         inner_verts,
#     ) = final_out

#     # Diagnostics
#     band_vals = darkness[full_band]

#     if band_dark_fraction >= 1.0:
#         dark_score = np.mean(band_vals)
#     else:
#         k = max(5, int(band_dark_fraction * len(band_vals)))
#         top_vals = np.partition(band_vals, -k)[-k:]
#         dark_score = np.mean(top_vals)

#     d_final = dist_to_dark[full_band]
#     mean_dark_distance = d_final.mean()
#     far_dark_fraction = np.mean(d_final > max_dark_dist_px)

#     inside_final = points_inside_equilateral_triangle(
#         poly_local,
#         cx=cx_best,
#         cy=cy_best,
#         side_px=side_px_best,
#         theta=theta_best,
#         margin_px=-1.0,
#     )

#     final_poly_inside_fraction = inside_final.mean()

#     outer_dark = darkness[outer_band].mean()
#     inner_dark = darkness[inner_band].mean()

#     # Convert vertices back to global coordinates
#     outer_global = outer_verts.copy()
#     mid_global = mid_verts.copy()
#     inner_global = inner_verts.copy()

#     init_outer_global = init_outer_verts.copy()
#     init_mid_global = init_mid_verts.copy()
#     init_inner_global = init_inner_verts.copy()

#     for verts in [
#         outer_global,
#         mid_global,
#         inner_global,
#         init_outer_global,
#         init_mid_global,
#         init_inner_global,
#     ]:
#         verts[:, 0] += x_min
#         verts[:, 1] += y_min

#     x_refined = cx_best + x_min
#     y_refined = cy_best + y_min

#     return {
#         "x_refined": x_refined,
#         "y_refined": y_refined,
#         "theta": theta_best,
#         "theta0": theta0,
#         "dtheta": dtheta_best,

#         "side_px": side_px_best,
#         "side_um": side_px_best * um_per_px,
#         "side0_px": side0,
#         "side0_um": side0 * um_per_px,
#         "side_px_nominal": side_px_nominal,
#         "side_px_max": side_px_max,
#         "required_side_px": init["required_side_px"],

#         "thickness_px": thickness_px,
#         "loss": res.fun,
#         "success": res.success,

#         "dark_score": dark_score,
#         "outer_dark": outer_dark,
#         "inner_dark": inner_dark,
#         "mean_dark_distance": mean_dark_distance,
#         "far_dark_fraction": far_dark_fraction,
#         "dark_threshold": threshold,

#         "start_inside_fraction": init["start_inside_fraction"],
#         "final_poly_inside_fraction": final_poly_inside_fraction,

#         "dx_px": dx_best,
#         "dy_px": dy_best,
#         "center_shift_px": np.sqrt(dx_best**2 + dy_best**2),
#         "rotation_change_deg": np.rad2deg(dtheta_best),

#         "outer_vertices": outer_global,
#         "mid_vertices": mid_global,
#         "inner_vertices": inner_global,

#         "initial_outer_vertices": init_outer_global,
#         "initial_mid_vertices": init_mid_global,
#         "initial_inner_vertices": init_inner_global,

#         "crop_box": (y_min, y_max, x_min, x_max),
#         "poly_local": poly_local,
#         "poly_mask_local": poly_mask,
#         "search_region_local": search_region,
#         "dark_mask_local": dark_mask,
#         "darkness_crop": darkness,
#         "dist_to_dark_crop": dist_to_dark,

#         "outer_band_local": outer_band,
#         "inner_band_local": inner_band,
#         "full_band_local": full_band,
#         "initial_full_band_local": init_full_band,
#     }
#%%
def fit_triangle_prism_local_dark_edges_near_poly(
    arr8,
    poly_xy,
    um_per_px=0.32,
    side_um=10.0,
    thickness_um=2.0,

    # initial triangle / size
    side_slack_px=6.0,
    contain_margin_px=1.0,

    # local optimization limits
    max_center_shift_um=1.5,
    max_rotation_deg=15.0,
    side_local_slack_px=2.0,
    allow_shrink_from_start=False,

    # dark-pixel target around initial triangle band
    blur_sigma=1.0,
    dark_search_dist_px=5.0,
    dark_pixel_fraction=0.25,

    # objective
    max_dark_dist_px=2.0,
    band_dark_fraction=0.80,
    w_dark_dist=0.20,
    w_far_dark=1.0,
    w_center=0.10,
    w_rotation=0.05,
    w_contain=0.5,
    max_poly_outside_fraction=0.0,

    # NEW: punish bright fitted edges
    edge_check_width_px=2.0,
    min_edge_darkness=0.45,
    max_bright_edge_fraction=0.20,
    w_bright_edge=3.0,
    w_bright_edge_fraction=2.0,

    maxiter=80,
):
    """
    Faster, conservative local fit.

    Steps:
    1. Put triangle at YOLO polygon centroid.
    2. Choose initial orientation/size so YOLO polygon is inside.
    3. Build dark-pixel target near that initial triangle band.
    4. Optimize only locally: small shift, small rotation, small size change.
    5. Extra penalty if the fitted triangle edge lies on bright pixels.
    """

    arr8 = np.asarray(arr8)
    poly_xy = np.asarray(poly_xy, dtype=float)

    if arr8.ndim != 2:
        raise ValueError("arr8 must be 2D grayscale")

    if poly_xy.ndim != 2 or poly_xy.shape[1] != 2:
        raise ValueError("poly_xy must have shape (N, 2)")

    if len(poly_xy) < 3:
        raise ValueError("poly_xy must contain at least 3 points")

    # -------------------------
    # Physical to pixel units
    # -------------------------
    side_px_nominal = side_um / um_per_px
    side_px_max = side_px_nominal + side_slack_px

    thickness_px = thickness_um / um_per_px
    max_center_shift_px = max_center_shift_um / um_per_px
    max_rotation_rad = np.deg2rad(max_rotation_deg)

    # -------------------------
    # Initial center from YOLO polygon
    # -------------------------
    x0_global, y0_global = polygon_centroid(poly_xy)

    init = initial_triangle_containing_poly(
        poly_xy,
        cx=x0_global,
        cy=y0_global,
        side_px_nominal=side_px_nominal,
        side_px_max=side_px_max,
        margin_px=contain_margin_px,
        n_theta=90,
    )

    theta0 = init["theta0"]
    side0 = init["side0"]

    # -------------------------
    # Crop around particle
    # -------------------------
    x_min0 = int(np.floor(poly_xy[:, 0].min()))
    x_max0 = int(np.ceil(poly_xy[:, 0].max()))
    y_min0 = int(np.floor(poly_xy[:, 1].min()))
    y_max0 = int(np.ceil(poly_xy[:, 1].max()))

    H, W = arr8.shape

    pad = int(np.ceil(
        side_px_max / np.sqrt(3)
        + thickness_px
        + max_center_shift_px
        + dark_search_dist_px
        + edge_check_width_px
        + 10
    ))

    x_min = max(0, x_min0 - pad)
    x_max = min(W, x_max0 + pad + 1)
    y_min = max(0, y_min0 - pad)
    y_max = min(H, y_max0 + pad + 1)

    img_crop = arr8[y_min:y_max, x_min:x_max].astype(float)
    crop_shape = img_crop.shape

    poly_local = poly_xy.copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min

    x0 = x0_global - x_min
    y0 = y0_global - y_min

    # -------------------------
    # YOLO mask only for visualization / containment
    # -------------------------
    poly_mask = polygon_to_mask(crop_shape, poly_local)

    # -------------------------
    # Initial triangle band
    # -------------------------
    init_out = triangle_prism_band_masks(
        crop_shape,
        cx=x0,
        cy=y0,
        side_px=side0,
        theta=theta0,
        thickness_px=thickness_px,
    )

    if init_out is None:
        raise RuntimeError("Initial triangle band construction failed")

    (
        init_outer_band,
        init_inner_band,
        init_full_band,
        init_outer_verts,
        init_mid_verts,
        init_inner_verts,
    ) = init_out

    # Use only dark pixels near the initial triangle band.
    # This prevents the fit from jumping to neighboring particles.
    dist_to_init_band = distance_transform_edt(~init_full_band)
    search_region = dist_to_init_band <= dark_search_dist_px

    if search_region.sum() < 10:
        search_region = np.ones(crop_shape, dtype=bool)

    # -------------------------
    # Darkness image
    # -------------------------
    img_smooth = gaussian_filter(img_crop, sigma=blur_sigma)

    lo, hi = np.percentile(img_smooth, [1, 99])
    img_norm = np.clip((img_smooth - lo) / (hi - lo + 1e-9), 0, 1)

    # darkness: 1 = black/dark, 0 = bright
    darkness = 1.0 - img_norm

    darkness_inside_search = darkness[search_region]

    threshold = np.quantile(
        darkness_inside_search,
        1.0 - dark_pixel_fraction
    )

    dark_mask = search_region & (darkness >= threshold)

    if dark_mask.sum() < 10:
        threshold = np.quantile(darkness, 1.0 - dark_pixel_fraction)
        dark_mask = darkness >= threshold

    dist_to_dark = distance_transform_edt(~dark_mask)

    # -------------------------
    # Side bounds around initial side
    # -------------------------
    if allow_shrink_from_start:
        side_min = max(side_px_nominal, side0 - side_local_slack_px)
    else:
        side_min = side0

    side_max = min(side_px_max, side0 + side_local_slack_px)

    if side_max < side_min:
        side_max = side_min

    # -------------------------
    # Objective
    # -------------------------
    def objective(params):
        dx, dy, dtheta, side_px = params

        cx = x0 + dx
        cy = y0 + dy
        theta = theta0 + dtheta

        out = triangle_prism_band_masks(
            crop_shape,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta,
            thickness_px=thickness_px,
        )

        if out is None:
            return 1e6

        outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

        if full_band.sum() < 10:
            return 1e6

        # Darkness score over much of the band
        band_vals = darkness[full_band]

        if band_dark_fraction >= 1.0:
            dark_score = np.mean(band_vals)
        else:
            k = max(5, int(band_dark_fraction * len(band_vals)))
            top_vals = np.partition(band_vals, -k)[-k:]
            dark_score = np.mean(top_vals)

        # Distance to local dark pixels
        d = dist_to_dark[full_band]

        dark_dist_penalty = np.mean(
            np.clip(d / (max_dark_dist_px + 1e-9), 0, 3) ** 2
        )

        far_dark_fraction = np.mean(d > max_dark_dist_px)

        # NEW: punish bright fitted edge
        # Use middle triangle outline as representative edge.
        edge_check_mask = polygon_outline_band_mask(
            crop_shape,
            mid_verts,
            width_px=edge_check_width_px,
        )

        edge_vals = darkness[edge_check_mask]

        if edge_vals.size < 5:
            return 1e6

        bright_edge_fraction = np.mean(edge_vals < min_edge_darkness)

        bright_edge_penalty = np.mean(
            (
                np.clip(min_edge_darkness - edge_vals, 0, None)
                / (min_edge_darkness + 1e-9)
            ) ** 2
        )

        if bright_edge_fraction > max_bright_edge_fraction:
            return 1e6 + 1000 * bright_edge_fraction

        # Keep near YOLO center
        center_penalty = (dx**2 + dy**2) / (max_center_shift_px**2 + 1e-9)

        # Keep near initial orientation
        rotation_penalty = (dtheta / (max_rotation_rad + 1e-9))**2

        # Do not let the fitted triangle drift too far away from YOLO polygon
        inside = points_inside_equilateral_triangle(
            poly_local,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta,
            margin_px=-1.0,
        )

        outside_fraction = 1.0 - inside.mean()

        if outside_fraction > max_poly_outside_fraction:
            return 1e6 + 1000 * outside_fraction

        loss = (
            -dark_score
            + w_dark_dist * dark_dist_penalty
            + w_far_dark * far_dark_fraction
            + w_center * center_penalty
            + w_rotation * rotation_penalty
            + w_contain * outside_fraction
            + w_bright_edge * bright_edge_penalty
            + w_bright_edge_fraction * bright_edge_fraction
        )

        return loss

    bounds = [
        (-max_center_shift_px, max_center_shift_px),
        (-max_center_shift_px, max_center_shift_px),
        (-max_rotation_rad, max_rotation_rad),
        (side_min, side_max),
    ]

    res = minimize(
        objective,
        x0=np.array([0.0, 0.0, 0.0, side0]),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "xtol": 1e-2,
            "ftol": 1e-3,
        },
    )

    dx_best, dy_best, dtheta_best, side_px_best = res.x

    cx_best = x0 + dx_best
    cy_best = y0 + dy_best
    theta_best = theta0 + dtheta_best

    final_out = triangle_prism_band_masks(
        crop_shape,
        cx=cx_best,
        cy=cy_best,
        side_px=side_px_best,
        theta=theta_best,
        thickness_px=thickness_px,
    )

    if final_out is None:
        raise RuntimeError("Final triangle band construction failed")

    (
        outer_band,
        inner_band,
        full_band,
        outer_verts,
        mid_verts,
        inner_verts,
    ) = final_out

    # -------------------------
    # Diagnostics
    # -------------------------
    band_vals = darkness[full_band]

    if band_dark_fraction >= 1.0:
        dark_score = np.mean(band_vals)
    else:
        k = max(5, int(band_dark_fraction * len(band_vals)))
        top_vals = np.partition(band_vals, -k)[-k:]
        dark_score = np.mean(top_vals)

    d_final = dist_to_dark[full_band]
    mean_dark_distance = d_final.mean()
    far_dark_fraction = np.mean(d_final > max_dark_dist_px)

    inside_final = points_inside_equilateral_triangle(
        poly_local,
        cx=cx_best,
        cy=cy_best,
        side_px=side_px_best,
        theta=theta_best,
        margin_px=-1.0,
    )

    final_poly_inside_fraction = inside_final.mean()

    outer_dark = darkness[outer_band].mean()
    inner_dark = darkness[inner_band].mean()

    # Final bright-edge diagnostics
    edge_check_mask = polygon_outline_band_mask(
        crop_shape,
        mid_verts,
        width_px=edge_check_width_px,
    )

    edge_vals = darkness[edge_check_mask]

    if edge_vals.size > 0:
        bright_edge_fraction = np.mean(edge_vals < min_edge_darkness)

        edge_bright_penalty = np.mean(
            (
                np.clip(min_edge_darkness - edge_vals, 0, None)
                / (min_edge_darkness + 1e-9)
            ) ** 2
        )

        mean_edge_darkness = np.mean(edge_vals)
    else:
        bright_edge_fraction = np.nan
        edge_bright_penalty = np.nan
        mean_edge_darkness = np.nan

    # -------------------------
    # Convert vertices back to global coordinates
    # -------------------------
    outer_global = outer_verts.copy()
    mid_global = mid_verts.copy()
    inner_global = inner_verts.copy()

    init_outer_global = init_outer_verts.copy()
    init_mid_global = init_mid_verts.copy()
    init_inner_global = init_inner_verts.copy()

    for verts in [
        outer_global,
        mid_global,
        inner_global,
        init_outer_global,
        init_mid_global,
        init_inner_global,
    ]:
        verts[:, 0] += x_min
        verts[:, 1] += y_min

    x_refined = cx_best + x_min
    y_refined = cy_best + y_min

    return {
        "x_refined": x_refined,
        "y_refined": y_refined,
        "theta": theta_best,
        "theta0": theta0,
        "dtheta": dtheta_best,

        "side_px": side_px_best,
        "side_um": side_px_best * um_per_px,
        "side0_px": side0,
        "side0_um": side0 * um_per_px,
        "side_px_nominal": side_px_nominal,
        "side_px_max": side_px_max,
        "required_side_px": init["required_side_px"],

        "thickness_px": thickness_px,
        "loss": res.fun,
        "success": res.success,

        "dark_score": dark_score,
        "outer_dark": outer_dark,
        "inner_dark": inner_dark,
        "mean_dark_distance": mean_dark_distance,
        "far_dark_fraction": far_dark_fraction,
        "dark_threshold": threshold,

        "bright_edge_fraction": bright_edge_fraction,
        "edge_bright_penalty": edge_bright_penalty,
        "mean_edge_darkness": mean_edge_darkness,
        "min_edge_darkness": min_edge_darkness,

        "start_inside_fraction": init["start_inside_fraction"],
        "final_poly_inside_fraction": final_poly_inside_fraction,

        "dx_px": dx_best,
        "dy_px": dy_best,
        "center_shift_px": np.sqrt(dx_best**2 + dy_best**2),
        "rotation_change_deg": np.rad2deg(dtheta_best),

        "outer_vertices": outer_global,
        "mid_vertices": mid_global,
        "inner_vertices": inner_global,

        "initial_outer_vertices": init_outer_global,
        "initial_mid_vertices": init_mid_global,
        "initial_inner_vertices": init_inner_global,

        "crop_box": (y_min, y_max, x_min, x_max),
        "poly_local": poly_local,
        "poly_mask_local": poly_mask,
        "search_region_local": search_region,
        "dark_mask_local": dark_mask,
        "darkness_crop": darkness,
        "dist_to_dark_crop": dist_to_dark,

        "outer_band_local": outer_band,
        "inner_band_local": inner_band,
        "full_band_local": full_band,
        "initial_full_band_local": init_full_band,
        "edge_check_mask_local": edge_check_mask,
    }
# %%
def plot_triangle_prism_local_dark_fit(
    arr8,
    poly_xy,
    res,
    alpha_poly=0.15,
    alpha_dark=0.35,
    show_dark_pixels=True,
    show_initial=True,
):
    y_min, y_max, x_min, x_max = res["crop_box"]

    crop = arr8[y_min:y_max, x_min:x_max]

    poly_local = np.asarray(poly_xy, dtype=float).copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min
    poly_closed = np.vstack([poly_local, poly_local[0]])

    def local_closed(vertices_global):
        v = vertices_global.copy()
        v[:, 0] -= x_min
        v[:, 1] -= y_min
        return np.vstack([v, v[0]])

    outer_c = local_closed(res["outer_vertices"])
    mid_c = local_closed(res["mid_vertices"])
    inner_c = local_closed(res["inner_vertices"])

    init_outer_c = local_closed(res["initial_outer_vertices"])
    init_mid_c = local_closed(res["initial_mid_vertices"])
    init_inner_c = local_closed(res["initial_inner_vertices"])

    x_ref = res["x_refined"] - x_min
    y_ref = res["y_refined"] - y_min

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(crop, cmap="gray")

    # YOLO polygon fill
    poly_mask = res["poly_mask_local"]
    poly_overlay = np.zeros((*poly_mask.shape, 4), dtype=float)
    poly_overlay[..., 0] = 1.0
    poly_overlay[..., 3] = alpha_poly * poly_mask.astype(float)
    ax.imshow(poly_overlay)

    # Dark pixels used for fitting
    if show_dark_pixels:
        dark_mask = res["dark_mask_local"]
        dark_overlay = np.zeros((*dark_mask.shape, 4), dtype=float)
        dark_overlay[..., 1] = 1.0
        dark_overlay[..., 2] = 1.0
        dark_overlay[..., 3] = alpha_dark * dark_mask.astype(float)
        ax.imshow(dark_overlay)

    # YOLO polygon outline
    ax.plot(
        poly_closed[:, 0],
        poly_closed[:, 1],
        linewidth=2,
        label="YOLO poly outline"
    )

    # Initial triangle
    if show_initial:
        ax.plot(
            init_outer_c[:, 0],
            init_outer_c[:, 1],
            linewidth=1.5,
            linestyle=":",
            color="white",
            label="initial outer triangle"
        )
        ax.plot(
            init_mid_c[:, 0],
            init_mid_c[:, 1],
            linewidth=1.0,
            linestyle=":",
            color="white"
        )
        ax.plot(
            init_inner_c[:, 0],
            init_inner_c[:, 1],
            linewidth=1.0,
            linestyle=":",
            color="white"
        )

    # Fitted triangle
    ax.plot(
        outer_c[:, 0],
        outer_c[:, 1],
        linewidth=2,
        label="fitted outer triangle"
    )
    ax.plot(
        mid_c[:, 0],
        mid_c[:, 1],
        linewidth=2,
        linestyle="--",
        label="fitted middle"
    )
    ax.plot(
        inner_c[:, 0],
        inner_c[:, 1],
        linewidth=2,
        label="fitted inner triangle"
    )

    ax.scatter(
        x_ref,
        y_ref,
        s=80,
        marker="x",
        label="fitted center"
    )

    ax.set_title(
        f"Local triangle fit to dark edges\n"
        f"dark_score={res['dark_score']:.3f}, "
        f"mean dark dist={res['mean_dark_distance']:.2f}px, "
        f"far_dark={res['far_dark_fraction']:.2f}\n"
        f"side={res['side_px']:.2f}px = {res['side_um']:.2f} µm, "
        f"shift={res['center_shift_px']:.2f}px, "
        f"dtheta={res['rotation_change_deg']:.1f}°"
    )
\

    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.show()
# %%

for  particle_id in range(0,200):
   
    row = det_df[det_df["particle_id"] == particle_id].iloc[0]
    poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)

    res_local = fit_triangle_prism_local_dark_edges_near_poly(
        arr8,
        poly_xy,
        um_per_px=mpp_real,
        side_um=10.0,
        thickness_um=2.0,

        side_slack_px=3.0,
        contain_margin_px=0.0,

        max_center_shift_um=1.5,
        max_rotation_deg=15.0,
        side_local_slack_px=2.0,

        dark_search_dist_px=3.0,
        dark_pixel_fraction=0.25,

        max_dark_dist_px=2.0,
        band_dark_fraction=0.80,

        w_dark_dist=0.20,
        w_far_dark=1.0,
        w_center=0.10,
        w_rotation=0.05,
        w_contain=0.5,

        maxiter=100,
    )

    print("side px:", res_local["side_px"])
    print("side um:", res_local["side_um"])
    print("center shift px:", res_local["center_shift_px"])
    print("rotation change deg:", res_local["rotation_change_deg"])
    print("start inside fraction:", res_local["start_inside_fraction"])
    print("final inside fraction:", res_local["final_poly_inside_fraction"])

    plot_triangle_prism_local_dark_fit(arr8, poly_xy, res_local)
# %%

res_local = fit_triangle_prism_local_dark_edges_near_poly(
    arr8,
    poly_xy,
    um_per_px=mpp_real,
    side_um=10.0,
    thickness_um=2.0,

    side_slack_px=3.0,
    contain_margin_px=1.0,

    max_center_shift_um=1.2,
    max_rotation_deg=12.0,
    side_local_slack_px=1.5,

    dark_search_dist_px=4.0,
    dark_pixel_fraction=0.25,

    max_dark_dist_px=2.0,
    band_dark_fraction=0.90,

    w_dark_dist=0.20,
    w_far_dark=1.0,
    w_center=0.15,
    w_rotation=0.08,
    w_contain=0.8,

    edge_check_width_px=2.0,
    min_edge_darkness=0.45,
    max_bright_edge_fraction=0.15,
    w_bright_edge=4.0,
    w_bright_edge_fraction=3.0,

    maxiter=80,
)

print("bright edge fraction:", res_local["bright_edge_fraction"])
print("mean edge darkness:", res_local["mean_edge_darkness"])
print("edge bright penalty:", res_local["edge_bright_penalty"])
# %%
particle_id = 11

row = det_df[det_df["particle_id"] == particle_id].iloc[0]
poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)

res_local = fit_triangle_prism_local_dark_edges_near_poly(
    arr8,
    poly_xy,
    um_per_px=mpp_real,
    side_um=10.0,
    thickness_um=2.0,

    side_slack_px=3.0,
    contain_margin_px=1.0,

    max_center_shift_um=1.2,
    max_rotation_deg=12.0,
    side_local_slack_px=1.5,

    dark_search_dist_px=4.0,
    dark_pixel_fraction=0.25,

    max_dark_dist_px=2.0,
    band_dark_fraction=0.90,

    w_dark_dist=0.20,
    w_far_dark=1.0,
    w_center=0.15,
    w_rotation=0.08,
    w_contain=0.8,

    edge_check_width_px=2.0,
    min_edge_darkness=0.45,
    max_bright_edge_fraction=0.15,
    w_bright_edge=7.0,
    w_bright_edge_fraction=10.0,

    maxiter=80,
)

print("bright edge fraction:", res_local["bright_edge_fraction"])
print("mean edge darkness:", res_local["mean_edge_darkness"])
print("edge bright penalty:", res_local["edge_bright_penalty"])

plot_triangle_prism_local_dark_fit(arr8, poly_xy, res_local)
# %%





import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.optimize import minimize


def fit_triangle_prism_local_band_darkness_near_poly(
    arr8,
    poly_xy,
    um_per_px=0.32,
    side_um=10.0,
    thickness_um=2.0,

    # initial triangle around YOLO segmentation
    side_slack_px=6.0,
    contain_margin_px=1.0,

    # local optimization limits
    max_center_shift_um=1.5,
    max_rotation_deg=20.0,
    side_local_slack_px=2.0,
    allow_shrink_from_start=True,

    # darkness scoring
    blur_sigma=1.0,
    dark_top_fraction=0.25,

    # YOLO-outline / containment constraints
    max_edge_dist_um=3.0,
    max_far_fraction=0.50,
    max_poly_outside_fraction=0.05,

    # objective weights
    w_dist=0.5,
    w_far=1.0,
    w_center=0.10,
    w_rotation=0.05,
    w_contain=0.5,

    maxiter=80,
):
    """
    Local triangle-band fit near a YOLO segmentation.

    This is a hybrid between the two previous functions:

    - Starts from a triangle that approximately contains the YOLO polygon.
    - Optimizes locally around that starting triangle.
    - Does NOT build a dark-pixel mask.
    - Scores darkness directly from the fitted triangle band.
    - Keeps the fitted band near the YOLO outline.
    - Optionally requires most YOLO polygon points to remain inside the triangle.

    Requires helper functions:
        polygon_centroid
        polygon_to_mask
        triangle_prism_band_masks
        initial_triangle_containing_poly
        points_inside_equilateral_triangle
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
    # Physical to pixel units
    # -------------------------
    side_px_nominal = side_um / um_per_px
    side_px_max = side_px_nominal + side_slack_px

    thickness_px = thickness_um / um_per_px
    max_center_shift_px = max_center_shift_um / um_per_px
    max_rotation_rad = np.deg2rad(max_rotation_deg)
    max_edge_dist_px = max_edge_dist_um / um_per_px

    # -------------------------
    # Initial center from YOLO polygon
    # -------------------------
    x0_global, y0_global = polygon_centroid(poly_xy)

    # Find initial orientation/size around the segmentation
    init = initial_triangle_containing_poly(
        poly_xy,
        cx=x0_global,
        cy=y0_global,
        side_px_nominal=side_px_nominal,
        side_px_max=side_px_max,
        margin_px=contain_margin_px,
        n_theta=90,
    )

    theta0 = init["theta0"]
    side0 = init["side0"]

    # -------------------------
    # Crop around particle
    # -------------------------
    x_min0 = int(np.floor(poly_xy[:, 0].min()))
    x_max0 = int(np.ceil(poly_xy[:, 0].max()))
    y_min0 = int(np.floor(poly_xy[:, 1].min()))
    y_max0 = int(np.ceil(poly_xy[:, 1].max()))

    H, W = arr8.shape

    pad = int(np.ceil(
        side_px_max / np.sqrt(3)
        + thickness_px
        + max_center_shift_px
        + max_edge_dist_px
        + 10
    ))

    x_min = max(0, x_min0 - pad)
    x_max = min(W, x_max0 + pad + 1)
    y_min = max(0, y_min0 - pad)
    y_max = min(H, y_max0 + pad + 1)

    img_crop = arr8[y_min:y_max, x_min:x_max].astype(float)
    crop_shape = img_crop.shape

    poly_local = poly_xy.copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min

    x0 = x0_global - x_min
    y0 = y0_global - y_min

    # -------------------------
    # YOLO polygon outline distance map
    # -------------------------
    poly_mask = polygon_to_mask(crop_shape, poly_local)

    poly_eroded = binary_erosion(poly_mask)
    poly_outline = poly_mask ^ poly_eroded

    if poly_outline.sum() == 0:
        poly_outline = poly_mask.copy()

    dist_to_outline = distance_transform_edt(~poly_outline)

    # -------------------------
    # Darkness image
    # -------------------------
    img_smooth = gaussian_filter(img_crop, sigma=blur_sigma)

    lo, hi = np.percentile(img_smooth, [1, 99])
    img_norm = np.clip((img_smooth - lo) / (hi - lo + 1e-9), 0, 1)

    # darkness = 1 means dark/black, 0 means bright
    darkness = 1.0 - img_norm

    # -------------------------
    # Initial triangle band for diagnostics
    # -------------------------
    init_out = triangle_prism_band_masks(
        crop_shape,
        cx=x0,
        cy=y0,
        side_px=side0,
        theta=theta0,
        thickness_px=thickness_px,
    )

    if init_out is None:
        raise RuntimeError("Initial triangle band construction failed")

    (
        init_outer_band,
        init_inner_band,
        init_full_band,
        init_outer_verts,
        init_mid_verts,
        init_inner_verts,
    ) = init_out

    # -------------------------
    # Side bounds around initial segmentation-based size
    # -------------------------
    if allow_shrink_from_start:
        side_min = max(1.0, side0 - side_local_slack_px)
    else:
        side_min = side0

    side_max = min(side_px_max, side0 + side_local_slack_px)

    if side_max < side_min:
        side_max = side_min

    # -------------------------
    # Objective
    # -------------------------
    def objective(params):
        dx, dy, dtheta, side_px = params

        cx = x0 + dx
        cy = y0 + dy
        theta = theta0 + dtheta

        out = triangle_prism_band_masks(
            crop_shape,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta,
            thickness_px=thickness_px,
        )

        if out is None:
            return 1e6

        outer_band, inner_band, full_band, outer_verts, mid_verts, inner_verts = out

        if full_band.sum() < 10:
            return 1e6

        # -------------------------
        # Direct darkness score
        # -------------------------
        band_vals = darkness[full_band]

        k = max(5, int(dark_top_fraction * len(band_vals)))
        top_vals = np.partition(band_vals, -k)[-k:]

        dark_score = np.mean(top_vals)

        # -------------------------
        # Distance to YOLO outline
        # -------------------------
        d = dist_to_outline[full_band]

        far_fraction = np.mean(d > max_edge_dist_px)

        if far_fraction > max_far_fraction:
            return 1e6 + 1000 * far_fraction

        dist_penalty = np.mean(
            np.clip(d / (max_edge_dist_px + 1e-9) - 1.0, 0, None) ** 2
        )

        # -------------------------
        # Keep near YOLO centroid
        # -------------------------
        center_penalty = (dx**2 + dy**2) / (max_center_shift_px**2 + 1e-9)

        # -------------------------
        # Keep near initial orientation
        # -------------------------
        rotation_penalty = (dtheta / (max_rotation_rad + 1e-9))**2

        # -------------------------
        # Keep YOLO polygon mostly inside fitted triangle
        # -------------------------
        inside = points_inside_equilateral_triangle(
            poly_local,
            cx=cx,
            cy=cy,
            side_px=side_px,
            theta=theta,
            margin_px=-1.0,
        )

        outside_fraction = 1.0 - inside.mean()

        if outside_fraction > max_poly_outside_fraction:
            return 1e6 + 1000 * outside_fraction

        # -------------------------
        # Final loss
        # -------------------------
        loss = (
            -dark_score
            + w_dist * dist_penalty
            + w_far * far_fraction
            + w_center * center_penalty
            + w_rotation * rotation_penalty
            + w_contain * outside_fraction
        )

        return loss

    bounds = [
        (-max_center_shift_px, max_center_shift_px),
        (-max_center_shift_px, max_center_shift_px),
        (-max_rotation_rad, max_rotation_rad),
        (side_min, side_max),
    ]

    res = minimize(
        objective,
        x0=np.array([0.0, 0.0, 0.0, side0]),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "xtol": 1e-2,
            "ftol": 1e-3,
        },
    )

    dx_best, dy_best, dtheta_best, side_px_best = res.x

    cx_best = x0 + dx_best
    cy_best = y0 + dy_best
    theta_best = theta0 + dtheta_best

    final_out = triangle_prism_band_masks(
        crop_shape,
        cx=cx_best,
        cy=cy_best,
        side_px=side_px_best,
        theta=theta_best,
        thickness_px=thickness_px,
    )

    if final_out is None:
        raise RuntimeError("Final triangle band construction failed")

    (
        outer_band,
        inner_band,
        full_band,
        outer_verts,
        mid_verts,
        inner_verts,
    ) = final_out

    # -------------------------
    # Diagnostics
    # -------------------------
    band_vals = darkness[full_band]
    k = max(5, int(dark_top_fraction * len(band_vals)))
    top_vals = np.partition(band_vals, -k)[-k:]
    dark_score = np.mean(top_vals)

    outer_dark = darkness[outer_band].mean()
    inner_dark = darkness[inner_band].mean()

    d_final = dist_to_outline[full_band]
    mean_outline_distance = d_final.mean()
    far_fraction = np.mean(d_final > max_edge_dist_px)

    inside_final = points_inside_equilateral_triangle(
        poly_local,
        cx=cx_best,
        cy=cy_best,
        side_px=side_px_best,
        theta=theta_best,
        margin_px=-1.0,
    )

    final_poly_inside_fraction = inside_final.mean()

    # -------------------------
    # Convert vertices back to global coordinates
    # -------------------------
    outer_global = outer_verts.copy()
    mid_global = mid_verts.copy()
    inner_global = inner_verts.copy()

    init_outer_global = init_outer_verts.copy()
    init_mid_global = init_mid_verts.copy()
    init_inner_global = init_inner_verts.copy()

    for verts in [
        outer_global,
        mid_global,
        inner_global,
        init_outer_global,
        init_mid_global,
        init_inner_global,
    ]:
        verts[:, 0] += x_min
        verts[:, 1] += y_min

    x_refined = cx_best + x_min
    y_refined = cy_best + y_min

    return {
        "x_refined": x_refined,
        "y_refined": y_refined,

        "theta": theta_best,
        "theta0": theta0,
        "dtheta": dtheta_best,
        "rotation_change_deg": np.rad2deg(dtheta_best),

        "side_px": side_px_best,
        "side_um": side_px_best * um_per_px,
        "side0_px": side0,
        "side0_um": side0 * um_per_px,
        "side_px_nominal": side_px_nominal,
        "side_px_max": side_px_max,
        "required_side_px": init["required_side_px"],

        "thickness_px": thickness_px,
        "loss": res.fun,
        "success": res.success,

        "dark_score": dark_score,
        "outer_dark": outer_dark,
        "inner_dark": inner_dark,

        "mean_outline_distance": mean_outline_distance,
        "far_fraction": far_fraction,

        "start_inside_fraction": init["start_inside_fraction"],
        "final_poly_inside_fraction": final_poly_inside_fraction,

        "dx_px": dx_best,
        "dy_px": dy_best,
        "center_shift_px": np.sqrt(dx_best**2 + dy_best**2),

        "outer_vertices": outer_global,
        "mid_vertices": mid_global,
        "inner_vertices": inner_global,

        "initial_outer_vertices": init_outer_global,
        "initial_mid_vertices": init_mid_global,
        "initial_inner_vertices": init_inner_global,

        "crop_box": (y_min, y_max, x_min, x_max),

        "poly_local": poly_local,
        "poly_mask_local": poly_mask,
        "poly_outline_local": poly_outline,

        "darkness_crop": darkness,
        "dist_to_outline_crop": dist_to_outline,

        "outer_band_local": outer_band,
        "inner_band_local": inner_band,
        "full_band_local": full_band,
        "initial_full_band_local": init_full_band,
    }
# %%
def plot_triangle_prism_local_band_darkness_fit(
    arr8,
    poly_xy,
    res,
    alpha_poly=0.15,
    alpha_initial_band=0.20,
    alpha_final_band=0.25,
    show_initial=True,
    show_band_overlay=True,
    show_darkness_panel=True,
):
    """
    Plot result from fit_triangle_prism_local_band_darkness_near_poly().

    Shows:
    - image crop
    - YOLO polygon
    - initial triangle from segmentation
    - final fitted triangle
    - fitted center
    - optional band overlays
    - optional darkness crop panel

    This version does NOT require dark_mask_local.
    """

    y_min, y_max, x_min, x_max = res["crop_box"]

    crop = arr8[y_min:y_max, x_min:x_max]

    # -------------------------
    # Local YOLO polygon
    # -------------------------
    poly_local = np.asarray(poly_xy, dtype=float).copy()
    poly_local[:, 0] -= x_min
    poly_local[:, 1] -= y_min
    poly_closed = np.vstack([poly_local, poly_local[0]])

    # -------------------------
    # Helper: global vertices -> local closed polygon
    # -------------------------
    def local_closed(vertices_global):
        v = np.asarray(vertices_global, dtype=float).copy()
        v[:, 0] -= x_min
        v[:, 1] -= y_min
        return np.vstack([v, v[0]])

    outer_c = local_closed(res["outer_vertices"])
    mid_c = local_closed(res["mid_vertices"])
    inner_c = local_closed(res["inner_vertices"])

    init_outer_c = local_closed(res["initial_outer_vertices"])
    init_mid_c = local_closed(res["initial_mid_vertices"])
    init_inner_c = local_closed(res["initial_inner_vertices"])

    x_ref = res["x_refined"] - x_min
    y_ref = res["y_refined"] - y_min

    # -------------------------
    # Create figure
    # -------------------------
    if show_darkness_panel:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        ax = axes[0]
        ax_dark = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax_dark = None

    # -------------------------
    # Main image
    # -------------------------
    ax.imshow(crop, cmap="gray")

    # YOLO polygon fill
    poly_mask = res["poly_mask_local"]
    poly_overlay = np.zeros((*poly_mask.shape, 4), dtype=float)
    poly_overlay[..., 0] = 1.0
    poly_overlay[..., 3] = alpha_poly * poly_mask.astype(float)
    ax.imshow(poly_overlay)

    # Optional fitted/initial band overlays
    if show_band_overlay:
        if show_initial:
            init_band = res["initial_full_band_local"]
            init_overlay = np.zeros((*init_band.shape, 4), dtype=float)
            init_overlay[..., 2] = 1.0
            init_overlay[..., 3] = alpha_initial_band * init_band.astype(float)
            ax.imshow(init_overlay)

        final_band = res["full_band_local"]
        final_overlay = np.zeros((*final_band.shape, 4), dtype=float)
        final_overlay[..., 1] = 1.0
        final_overlay[..., 3] = alpha_final_band * final_band.astype(float)
        ax.imshow(final_overlay)

    # YOLO polygon outline
    ax.plot(
        poly_closed[:, 0],
        poly_closed[:, 1],
        linewidth=2,
        color="red",
        label="YOLO polygon"
    )

    # Initial triangle
    if show_initial:
        ax.plot(
            init_outer_c[:, 0],
            init_outer_c[:, 1],
            linewidth=1.5,
            linestyle=":",
            color="white",
            label="initial outer"
        )
        ax.plot(
            init_mid_c[:, 0],
            init_mid_c[:, 1],
            linewidth=1.2,
            linestyle=":",
            color="white",
            label="initial middle"
        )
        ax.plot(
            init_inner_c[:, 0],
            init_inner_c[:, 1],
            linewidth=1.2,
            linestyle=":",
            color="white",
            label="initial inner"
        )

    # Final fitted triangle
    ax.plot(
        outer_c[:, 0],
        outer_c[:, 1],
        linewidth=2,
        color="cyan",
        label="fitted outer"
    )
    ax.plot(
        mid_c[:, 0],
        mid_c[:, 1],
        linewidth=2,
        linestyle="--",
        color="yellow",
        label="fitted middle"
    )
    ax.plot(
        inner_c[:, 0],
        inner_c[:, 1],
        linewidth=2,
        color="lime",
        label="fitted inner"
    )

    # Fitted center
    ax.scatter(
        x_ref,
        y_ref,
        s=80,
        marker="x",
        color="magenta",
        linewidths=2,
        label="fitted center"
    )

    ax.set_title(
        f"Local band-darkness triangle fit\n"
        f"dark_score={res['dark_score']:.3f}, "
        f"outline_dist={res['mean_outline_distance']:.2f}px, "
        f"far={res['far_fraction']:.2f}\n"
        f"side={res['side_px']:.2f}px = {res['side_um']:.2f} µm, "
        f"shift={res['center_shift_px']:.2f}px, "
        f"dtheta={res['rotation_change_deg']:.1f}°"
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)

    # -------------------------
    # Darkness panel
    # -------------------------
    if show_darkness_panel:
        darkness = res["darkness_crop"]

        ax_dark.imshow(darkness, cmap="gray")

        # Final band overlay on darkness image
        final_band = res["full_band_local"]
        final_overlay = np.zeros((*final_band.shape, 4), dtype=float)
        final_overlay[..., 1] = 1.0
        final_overlay[..., 3] = 0.35 * final_band.astype(float)
        ax_dark.imshow(final_overlay)

        # YOLO outline
        ax_dark.plot(
            poly_closed[:, 0],
            poly_closed[:, 1],
            linewidth=2,
            color="red",
            label="YOLO polygon"
        )

        # Final triangle
        ax_dark.plot(
            outer_c[:, 0],
            outer_c[:, 1],
            linewidth=2,
            color="cyan",
            label="fitted outer"
        )
        ax_dark.plot(
            mid_c[:, 0],
            mid_c[:, 1],
            linewidth=2,
            linestyle="--",
            color="yellow",
            label="fitted middle"
        )
        ax_dark.plot(
            inner_c[:, 0],
            inner_c[:, 1],
            linewidth=2,
            color="lime",
            label="fitted inner"
        )

        ax_dark.scatter(
            x_ref,
            y_ref,
            s=80,
            marker="x",
            color="magenta",
            linewidths=2,
        )

        ax_dark.set_title(
            "Darkness image\n"
            "white = darker pixels, green = fitted band"
        )

        ax_dark.set_aspect("equal")
        ax_dark.axis("off")
        ax_dark.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
# %%
for particle_id in range(0,20):

    row = det_df[det_df["particle_id"] == particle_id].iloc[0]
    poly_xy = np.array(json.loads(row["poly_json"]), dtype=float)

    res_band = fit_triangle_prism_local_band_darkness_near_poly(
        arr8,
        poly_xy,
        um_per_px=mpp_real,
        side_um=10.0,
        thickness_um=2.0,

        side_slack_px=3.0,
        contain_margin_px=1.0,

        max_center_shift_um=1.5,
        max_rotation_deg=20.0,
        side_local_slack_px=2.0,
        allow_shrink_from_start=True,

        dark_top_fraction=0.25,

        max_edge_dist_um=3.0,
        max_far_fraction=0.50,
        max_poly_outside_fraction=0.05,

        w_dist=0.4,
        w_far=0.7,
        w_center=0.10,
        w_rotation=0.05,
        w_contain=0.5,

        maxiter=100,
    )

    print("side px:", res_band["side_px"])
    print("side um:", res_band["side_um"])
    print("center shift px:", res_band["center_shift_px"])
    print("rotation change deg:", res_band["rotation_change_deg"])
    print("start inside fraction:", res_band["start_inside_fraction"])
    print("final inside fraction:", res_band["final_poly_inside_fraction"])
    print("dark score:", res_band["dark_score"])
    print("far fraction:", res_band["far_fraction"])

    plot_triangle_prism_local_band_darkness_fit(
        arr8,
        poly_xy,
        res_band,
        show_initial=True,
        show_band_overlay=True,
        show_darkness_panel=True,
    )
# %%
