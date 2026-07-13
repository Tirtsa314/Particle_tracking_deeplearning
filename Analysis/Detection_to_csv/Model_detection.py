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
from matplotlib.patches import Polygon, Circle


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

def reading_nd2(file_path, frame_index=-1):
    
    FRAME_INDEX = frame_index   # -1 = last frame, 0 = first frame, 10 = frame 10, etc.

    # Optional: inspect ND2 metadata
    with nd2.ND2File(file_path) as f:
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
    framesarr = nd2.imread(file_path, dask=True)

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

    return frame, arr8

def plot_full_detections(arr8, det_df, show_ids=False, show_centers=True, figsize=(20, 20)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr8, cmap="gray")

    for _, row in det_df.iterrows():
        poly = np.array(json.loads(row["poly_json"]))

        # draw polygon outline
        patch = Polygon(poly, closed=True, fill=False, edgecolor="lime", linewidth=1.5)
        ax.add_patch(patch)

        # optional: draw center
        if show_centers:
            ax.add_patch(Circle((row["x"], row["y"]), radius=3, color="red"))

        # optional: draw particle id
        if show_ids:
            ax.text(
                row["x"], row["y"],
                str(int(row["particle_id"])),
                color="yellow", fontsize=8,
                ha="center", va="center"
            )

    ax.set_title(f"Full image with {len(det_df)} detections")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

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

def Model_detection(file_path,
                    model, 
                    frame_index=-1, 
                    mpp_real = 0.06, 
                    side_real_um = 1.0,
                    confidence_threshold=0.05,
):

    mpp_train = 0.32
    side_train_um = 10.0

    mpp_real = 0.06
    side_real_um = 1.0

    train_side_px = side_train_um / mpp_train
    real_side_px = side_real_um / mpp_real

    side_ref_um = 10.0
    mpp_ref = 0.32
    side_ref_px = side_ref_um / mpp_ref      # 31.25 px

    side_real_px = side_real_um / mpp_real   # for your new case: 1.0 / 0.06 = 16.67 px

    physical_scale = side_real_um / side_ref_um
    pixel_scale = side_real_px / side_ref_px

    frame, arr8 = reading_nd2(file_path, frame_index)

    model = model

    SCALE = train_side_px / real_side_px
    CONF = confidence_threshold
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
    return det_df, frame, arr8

    