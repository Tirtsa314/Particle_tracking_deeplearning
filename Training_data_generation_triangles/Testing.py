#%%

import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import matplotlib.image as mpimg

import numpy as np

def to_uint8(img, mode="minmax", p_low=1, p_high=99):
    """
    Convert an image to uint8.

    Parameters
    ----------
    img : array-like
        Input image, e.g. uint16 grayscale or float image.
    mode : {"minmax", "percentile", "divide256"}
        Conversion method:
        - "minmax": scale img min..max -> 0..255
        - "percentile": scale percentiles p_low..p_high -> 0..255
        - "divide256": simple uint16 -> uint8 by img / 256
    p_low, p_high : float
        Percentiles used when mode="percentile".

    Returns
    -------
    img8 : np.ndarray
        uint8 image
    """
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
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_30-03.pt") 
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_30-03_2.pt") #model without overlap setting

# model = YOLO(r"C:\Users\DenHaan\Downloads\best_no_overlap_31-03.pt") 
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_01_04_nooverlap_ratio2_best_final.pt")
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_nooverlap_ratio2_31-03_1855.pt")
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_nooverlap_ratio2_30-03.pt")
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_09-04_1epoch.pt")
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_09-04_3epoch.pt")
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_09-04_5epoch.pt")
# model = YOLO(r"C:\Users\DenHaan\Downloads\best_09-04_20epoch.pt")
# frame = r"C:\Users\DenHaan\Downloads\frame0"

# frame_path = r"C:\Users\DenHaan\Downloads\frame0.png"
# frame_path = r"C:\Cluster_mindist20\images\train\001088.png"
# frame_path = r"C:\Cluster_mindist20\images\train\000201.png"
# frame_path = r"C:\Train_set_small_28-03\images\train\000001.png"
# frame_path = r"C:\Users\DenHaan\Downloads\000001.png"

frame_path = r"C:\Users\DenHaan\Downloads\lastframe.png"
# frame_path = r"C:\Cluster_mindist20\images\train\001687.png"
frame = mpimg.imread(frame_path)
frame = to_uint8(frame, mode="percentile", p_low=1, p_high=99)

# plt.figure(figsize=(6,6))
# plt.imshow(frame, cmap="gray")
# plt.axis("off")
# plt.tight_layout()
# plt.show()

# # results = model.predict(source=r"C:\Users\DenHaan\Downloads\frame0.png",
# #                          save=True,
# #                          conf=0.001)
# results = model.predict(source=frame_path,
#                          save=True,
#                          conf=0.00001)

# # %%
# pred_img = results[0].plot(labels=False, boxes=False)

# plt.figure(figsize=(10,10))
# plt.imshow(pred_img[..., ::-1])
# plt.axis("off")
# plt.show()
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# img = Image.open(frame_path)
# img = to_uint8(img, mode="percentile", p_low=1, p_high=99)
# print("mode:", img.mode, "size:", img.size)

# arr = np.array(img)
# print("raw:", arr.dtype, arr.shape, arr.min(), arr.max())

# plt.figure(figsize=(6,6))
# plt.imshow(arr if arr.ndim == 3 else arr, cmap=None if arr.ndim == 3 else "gray")
# plt.axis("off")
# plt.show()
# %%
# import numpy as np
# from PIL import Image
# """For smaller:"""

# frame = np.array(Image.open(frame_path).convert("RGB"))

# H, W = frame.shape[:2]

# # choose one quarter
# # top-left:
# crop = frame[:H//2, :W//2]

# # other options:
# # top-right:    crop = frame[:H//2, W//2:]
# # bottom-left:  crop = frame[H//2:, :W//2]
# # bottom-right: crop = frame[H//2:, W//2:]

# crop = frame[:H//2, :W//2]


plt.figure(figsize=(6,6))
plt.imshow(frame, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

# results = model.predict(
#     source=crop,
#     save=True,
#     conf=0.1
# )

# pred_img = results[0].plot(labels=False, boxes=False)

# plt.figure(figsize=(10,10))
# plt.imshow(pred_img[..., ::-1])
# plt.axis("off")
# plt.show()


# %%
"""For 512:"""
import numpy as np
from PIL import Image

frame = np.array(Image.open(frame_path).convert("RGB"))

H, W = frame.shape[:2]

# choose one quarter
# top-left:
x=500
y=500
crop = frame[x:H*1//4+x, +y:W*1//4+y]

# other options:
# top-right:    crop = frame[:H//2, W//2:]
# bottom-left:  crop = frame[H//2:, :W//2]
# bottom-right: crop = frame[H//2:, W//2:]



plt.figure(figsize=(6,6))
plt.imshow(crop)
plt.axis("off")
plt.tight_layout()
plt.show()

results = model.predict(
    source=crop,
    save=True,
    conf=0.5
)

pred_img = results[0].plot(labels=False, boxes=False)

plt.figure(figsize=(10,10))
plt.imshow(pred_img[..., ::-1])
plt.axis("off")
plt.show()
# %%
print("frame:", frame.shape, frame.dtype, frame.min(), frame.max())
print("crop:", crop.shape, crop.dtype, crop.min(), crop.max())

plt.figure(figsize=(6,6))
plt.imshow(crop)
plt.axis("off")
plt.show()

results = model.predict(
    source=crop,
    save=True,
    conf=0.001,
)
# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ---------------------------
# settings
# ---------------------------
model = YOLO(r"C:\Users\DenHaan\Downloads\best_no_overlap_31-03.pt") 
frame_path = r"C:\Users\DenHaan\Downloads\frame0.png"

CONF = 0.5
N_ROWS = 4     # 2 x 4 = 8 tiles total
N_COLS = 4
OVERLAP = 0    # helps reduce border misses / seams

# ---------------------------
# load image
# ---------------------------
frame = np.array(Image.open(frame_path).convert("RGB"))
H, W = frame.shape[:2]

# helper: evenly spaced tile boundaries
ys = np.linspace(0, H, N_ROWS + 1, dtype=int)
xs = np.linspace(0, W, N_COLS + 1, dtype=int)

# output image with overlay
overlay_full = frame.copy()

# ---------------------------
# tiled prediction + stitching
# ---------------------------
for r in range(N_ROWS):
    for c in range(N_COLS):
        # exact tile we want to fill back in
        y0, y1 = ys[r], ys[r + 1]
        x0, x1 = xs[c], xs[c + 1]

        # padded tile for inference
        y0p = max(0, y0 - OVERLAP)
        y1p = min(H, y1 + OVERLAP)
        x0p = max(0, x0 - OVERLAP)
        x1p = min(W, x1 + OVERLAP)

        tile = frame[y0p:y1p, x0p:x1p]

        results = model.predict(
            source=tile,
            conf=CONF,
            save=False,      # important: do not save every tile separately
            verbose=False
        )

        # get plotted tile in RGB
        plotted_tile = np.array(
            results[0].plot(
                labels=False,
                boxes=False,
                pil=True
            )
        )

        # only paste back the non-padded center part
        top = y0 - y0p
        bottom = top + (y1 - y0)
        left = x0 - x0p
        right = left + (x1 - x0)

        overlay_full[y0:y1, x0:x1] = plotted_tile[top:bottom, left:right]

        print(
            f"tile ({r}, {c}) done | "
            f"tile=({y0}:{y1}, {x0}:{x1}) | "
            f"padded=({y0p}:{y1p}, {x0p}:{x1p})"
        )

# ---------------------------
# show result
# ---------------------------
plt.figure(figsize=(10, 10))
plt.imshow(overlay_full)
plt.axis("off")
plt.tight_layout()
plt.show()

# save
out_path = r"C:\Users\DenHaan\Downloads\frame0_overlay_8tiles.png"
Image.fromarray(overlay_full).save(out_path)
print("Saved to:", out_path)
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ---------------------------
# settings
# ---------------------------
model = YOLO(r"C:\Users\DenHaan\Downloads\best_dinges_60epoch.pt")
frame_path = r"C:\Users\DenHaan\Downloads\lastframe.png"

CONF = 0.05
TILE = 512
STRIDE = 256   # 256 = half-overlap, so you get the "middle" predictions too

# if the double overlay becomes too strong, reduce this a bit
DELTA_GAIN = 1.0

# ---------------------------
# load image
# ---------------------------

img16 = np.array(Image.open(frame_path))   # no convert("RGB")
arr8 = to_uint8(img16, mode="percentile", p_low=1, p_high=99)
frame = np.stack([arr8, arr8, arr8], axis=-1)   # now shape is (H, W, 3)

print("img16:", img16.shape, img16.dtype, img16.min(), img16.max())
print("arr8 :", arr8.shape, arr8.dtype, arr8.min(), arr8.max())
print("frame:", frame.shape, frame.dtype)
H, W = frame.shape[:2]

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

ys = make_starts(H, TILE, STRIDE)
xs = make_starts(W, TILE, STRIDE)

print("y starts:", ys)
print("x starts:", xs)
print("total tiles:", len(ys) * len(xs))

# ---------------------------
# tiled prediction + additive overlay
# ---------------------------
for yi, y0 in enumerate(ys):
    for xi, x0 in enumerate(xs):
        y1 = y0 + TILE
        x1 = x0 + TILE

        tile = frame[y0:y1, x0:x1]

        results = model.predict(
            source=tile,
            conf=CONF,
            imgsz=TILE,     # keep inference size at 512
            save=False,
            verbose=False
        )

        # RGB PIL image because pil=True
        plotted_tile = np.array(
            results[0].plot(
                labels=False,
                boxes=False,
                pil=True
            ),
            dtype=np.float32
        )

        # only add the overlay part, not the base tile again
        delta = (plotted_tile - tile.astype(np.float32)) * DELTA_GAIN

        # accumulate double predictions on top of each other
        delta_accum[y0:y1, x0:x1] += delta

        print(f"tile ({yi}, {xi}) done | y={y0}:{y1}, x={x0}:{x1}")

# ---------------------------
# final combined overlay
# ---------------------------
overlay_full = np.clip(frame_f + delta_accum, 0, 255).astype(np.uint8)

# ---------------------------
# show result
# ---------------------------
plt.figure(figsize=(10, 10))
plt.imshow(overlay_full)
plt.axis("off")
plt.tight_layout()
plt.show()

# save
out_path = r"C:\Users\DenHaan\Downloads\frame0_overlay_512_stride256_added.png"
Image.fromarray(overlay_full).save(out_path)
print("Saved to:", out_path)
# %%
