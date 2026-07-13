
# #%%
# import sys
# print(sys.executable)
# #%%
# import numpy as np
# import cv2
# import shapely
# #%%
# import deeptrack as dt
# print("deeptrack imported")
# %% 
import os, time, queue
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
from Training_data_generation_triangles.Image_creation.Image_generator import generate_image
from Training_data_generation_triangles.Image_creation.Triangle_Sampler_Class import TrianglePrismSampler
from Training_data_generation_triangles.Image_creation.particle_position.cluster_position import cluster_positions
from deeptrack import units as u
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Training_data_generation_triangles.Image_creation.Image_generator import generate_image
from Training_data_generation_triangles.Image_creation.Triangle_Sampler_Class import TrianglePrismSampler
from Training_data_generation_triangles.Image_creation.particle_position.cluster_position import cluster_positions, clusterpositions_test_one

# --- your helpers (same as before) ---

from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection

def _largest_polygon(geom):
    if geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)

    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)

    return None


def crop_image_and_outlines(
    img_u8,
    outlines,
    crop_xywh,
    clip_polygons=True,
    min_area_px=20,
):
    """
    crop_xywh = (x0, y0, crop_w, crop_h)

    img_u8: final saved image, already downsampled if you use scale
    outlines: polygons in the same pixel coordinates as img_u8
    """

    x0, y0, crop_w, crop_h = crop_xywh

    H, W = img_u8.shape[:2]

    x1 = x0 + crop_w
    y1 = y0 + crop_h

    # Keep crop inside image
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(W, int(x1))
    y1 = min(H, int(y1))

    cropped_img = img_u8[y0:y1, x0:x1].copy()
    cropped_H, cropped_W = cropped_img.shape[:2]

    crop_box = box(x0, y0, x1, y1)

    cropped_outlines = []

    for outline in outlines:
        pts = np.asarray(outline, dtype=float)

        if len(pts) < 3:
            continue

        if clip_polygons:
            poly = Polygon(pts)

            if not poly.is_valid:
                poly = poly.buffer(0)

            inter = poly.intersection(crop_box)
            poly = _largest_polygon(inter)

            if poly is None or poly.area < min_area_px:
                continue

            new_pts = np.asarray(poly.exterior.coords[:-1], dtype=float)

        else:
            # Only keep particles fully inside crop
            inside = (
                (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
                (pts[:, 1] >= y0) & (pts[:, 1] < y1)
            )

            if not np.all(inside):
                continue

            new_pts = pts.copy()

        # Shift from original image coordinates to cropped image coordinates
        new_pts[:, 0] -= x0
        new_pts[:, 1] -= y0

        # Numerical safety
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, cropped_W - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, cropped_H - 1)

        if len(new_pts) >= 3:
            cropped_outlines.append(new_pts)

    return cropped_img, cropped_outlines

def to_uint8_gray(img, p_low=0, p_high=99):
    # img = np.squeeze(img).astype(np.float32)

    # raw_bg = np.median(img)
    # out_bg = 0.7
    # gain = 0.5  # try 0.14, 0.16, 0.18

    # # darker-than-background particles
    # # img = out_bg + gain * (img - raw_bg)
    # dev = img - raw_bg
    # neg = np.minimum(dev, 0.0)   # darker-than-background part
    # pos = np.maximum(dev, 0.0)   # brighter-than-background part

    # gain_neg = 0.5
    # gain_pos = 0.5
    # knee = 0.3   # smaller = stronger highlight compression

    # img = (
    #     out_bg
    #     + gain_neg * neg
    #     + gain_pos * knee * np.tanh(pos / knee)
    # )

    # img = np.clip(img, 0, 1)



    img = np.squeeze(img).astype(np.float32)

    lo, hi = np.percentile(img, [p_low, p_high])

    print("raw min/max:", img.min(), img.max())
    print("lo, hi:", lo, hi)
    print("raw percentiles:", np.percentile(img, [0.1, 1, 5, 50, 95, 99, 99.9]))

    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0, 1)

    return (255.0 * img).astype(np.uint8)

    # return (255.0 * img).astype(np.uint8)





    # img = np.squeeze(img).astype(np.float32)

    # # # optional slight blur
    # # img = cv2.GaussianBlur(img, (0, 0), 0.7)

    # # fixed camera-like mapping
    # bg = 0.0
    # contrast = 0.10
    # img = bg + contrast * img

    # # small noise
    # img += np.random.normal(0, 0.01, img.shape)

    # # clip and convert
    # img = np.clip(img, 0, 1)
    # img = (255.0 * img).astype(np.uint8)

    # if img.ndim == 3 and img.shape[-1] == 1:
    #     img = img[..., 0]


        # keep the original image as much as possible
    # img = np.clip(img, 0, 1)
    # img = (255.0 * img).astype(np.uint8)

    # if img.ndim == 3 and img.shape[-1] == 1:
    #     img = img[..., 0]

    # return img

def outlines_to_yolo(outlines, H, W, class_id=0):
    lines = []
    for outline in outlines:
        if outline is None or len(outline) < 3:
            continue
        pts = np.asarray(outline, dtype=np.float32).copy()
        pts[:, 0] /= W  # x
        pts[:, 1] /= H  # y
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"{class_id} {coords}")
    return lines


def write_data_yaml(out_dir: Path, class_name="triangle"):
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: {class_name}\n"
    )

def next_index(images_dir: Path) -> int:
    existing = sorted(images_dir.glob("*.png"))
    if not existing:
        return 0
    try:
        return int(existing[-1].stem) + 1
    except ValueError:
        return len(existing)

# --- worker + writer ---
def worker_loop(out_q: Queue, class_id=0, min_area=30, n=None, scale=None):
    # rot_sampler_fn, pos_sampler_fn, N = cluster_positions()   # one cluster only

    pixel_size_um = 0.32
    side_length_um = 10.0
    thickness_um = 2.5

    side_length_px = scale * side_length_um / pixel_size_um
    thickness_px = scale * thickness_um / pixel_size_um
    # sampler = TrianglePrismSampler(
    #     H=2048, W=2048, margin=50, side_length_px=side_length_px, thickness_px=thickness_px,
    #     rot_sampler_fn=rot_sampler_fn,
    #     pos_sampler_fn=pos_sampler_fn,
    # )

    # # N = np.random.randint(10, 100)
    # image_of_particles = generate_image(
    #     sampler=sampler,
    #     pos_sampler=sampler.pos_sampler,
    #     rot_sampler=sampler.rot_sampler,
    # )

    print("|Worker loop started", flush=True)

    try:
        while True:
            rot_sampler_fn, pos_sampler_fn, N_cluster = cluster_positions(
                scale=scale,
                side_um=side_length_um,
                um_per_pixel=pixel_size_um,
            )

            sampler = TrianglePrismSampler(
            H=512*scale, W=512*scale, margin=0, side_length_px=side_length_px, thickness_px=thickness_px,
            rot_sampler_fn=rot_sampler_fn,
            pos_sampler_fn=pos_sampler_fn,
            scale=scale
    )
            image_of_particles = generate_image(
                sampler=sampler,
                pos_sampler=sampler.pos_sampler,
                rot_sampler=sampler.rot_sampler,
                N=N_cluster,
                image_size=512*scale,
                scale=scale
            )

            sampler.reset()
            print("starting resolve...", flush=True)
            t = time.time()
            img = image_of_particles.update().resolve()
            print(np.asarray(img).shape)


            print("resolve took", time.time() - t, "sec", flush=True)
            # outlines = [outline / scale for outline in sampler.outlines_px]

            # H, W = img.shape[:2]
            # lines = outlines_to_yolo(outlines, H=H, W=W, class_id=class_id)

            # img_u8 = to_uint8_gray(img)

            # out_q.put((img_u8, lines))
            # print("Produced 1 sample")
            outlines = [np.asarray(outline, dtype=float) / scale for outline in sampler.outlines_px]

            img_u8 = to_uint8_gray(img)

            # Example: central 384 x 384 crop from final 512 x 512 image
            H, W = img_u8.shape[:2]
            crop_w = 384
            crop_h = 384
            x0 = (W - crop_w) // 2
            y0 = (H - crop_h) // 2

            img_u8, outlines = crop_image_and_outlines(
                img_u8,
                outlines,
                crop_xywh=(x0, y0, crop_w, crop_h),
                clip_polygons=True,
            )

            H, W = img_u8.shape[:2]
            lines = outlines_to_yolo(outlines, H=H, W=W, class_id=class_id)

            out_q.put((img_u8, lines))

    except Exception as e:
        import traceback
        print("Worker crashed!")
        traceback.print_exc()

def writer_loop(out_q: Queue, out_dir: Path, split: str, start_idx: int, print_every=100,
                png_compression=0):
    img_dir = out_dir / f"images/{split}"
    lab_dir = out_dir / f"labels/{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    i = start_idx
    t0 = time.time()
    print("Write loop is doing something")

    while True:
        img_u8, lines = out_q.get()  # blocks until data arrives
        stem = f"{i:06d}"
        img_path = img_dir / f"{stem}.png"
        lab_path = lab_dir / f"{stem}.txt"

        # faster PNG
        cv2.imwrite(str(img_path), img_u8, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        lab_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        i += 1
        if i % print_every == 0:
            dt = time.time() - t0
            rate = (i - start_idx) / dt if dt > 0 else 0
            print(f"{split}: wrote {i-start_idx} samples | ~{rate:.2f} samples/s")

def stream_export_yolo_parallel(
    out_dir,
    split="train",
    class_name="triangle",
    class_id=0,
    num_workers=None,
    queue_size=64,
    scale=None
):
    """
    Ctrl+C to stop (in terminal). Works on a normal multi-core PC.
    Best speed if out_dir is on local SSD.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_dir, class_name=class_name)

    img_dir = out_dir / f"images/{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_index(img_dir)

    if num_workers is None:
        num_workers = max(1, min(4, cpu_count() - 1))  # sane default

    out_q = Queue(maxsize=queue_size)

    # Important: each worker must have its own DeepTrack generator state
    # We'll create per-process callables by recreating the pipeline inside each worker via closures.

    procs = []
    for _ in range(num_workers):
        p = Process(target=worker_loop, args=(out_q, class_id, 30, None, scale), daemon=True)
        p.start()
        procs.append(p)

    writer = Process(target=writer_loop, args=(out_q, out_dir, split, start_idx), daemon=True)
    writer.start()

    print(f"Parallel export running with {num_workers} workers.")
    print(f"Writing to: {out_dir}  (split={split}, start={start_idx})")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        for p in procs:
            p.terminate()
        writer.terminate()
        for p in procs:
            p.join()
        writer.join()
        print("Stopped.")


def export_one_yolo_test(
    out_dir,
    split="train",
    stem="000000",
    class_id=0,
    class_name="triangle",
    scale=2,
    show_overlay=True,
):
    """
    Generate exactly one YOLO training image + label file,
    and optionally show an overlay of the saved YOLO polygons.

    This is a small test version of stream_export_yolo_parallel.
    """
 
    out_dir = Path(out_dir)
    img_dir = out_dir / "images" / split
    lab_dir = out_dir / "labels" / split
    overlay_dir = out_dir / "overlays" / split

    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    write_data_yaml(out_dir, class_name=class_name)

    pixel_size_um = 0.32
    side_length_um = 10.0
    thickness_um = 2.5

    side_length_px = scale * side_length_um / pixel_size_um
    thickness_px = scale * thickness_um / pixel_size_um

    print("side_length_px =", side_length_px)
    print("thickness_px =", thickness_px)

    # Use this if your current cluster_positions takes arguments.
    rot_sampler_fn, pos_sampler_fn, N_cluster = clusterpositions_test_one(
        scale=scale,
        side_um=side_length_um,
        um_per_pixel=pixel_size_um,
    )

    print("N_cluster =", N_cluster)

    sampler = TrianglePrismSampler(
        H=512 * scale,
        W=512 * scale,
        margin=0,
        side_length_px=side_length_px,
        thickness_px=thickness_px,
        rot_sampler_fn=rot_sampler_fn,
        pos_sampler_fn=pos_sampler_fn,
    )

    image_of_particles = generate_image(
        sampler=sampler,
        pos_sampler=sampler.pos_sampler,
        rot_sampler=sampler.rot_sampler,
        N=N_cluster,
        um_per_pixel=pixel_size_um,
        image_size=512 * scale,
        side_length_um=side_length_um,
        thickness_um=thickness_um,
        scale=scale,
    )

    sampler.reset()

    print("Resolving one image...")
    t0 = time.time()
    img = image_of_particles.update().resolve()
    arr = np.asarray(img)

    print(type(img))
    print(arr.dtype)
    print(arr.shape)
    print(arr.min(), arr.max())
    print(np.percentile(arr, [0.1, 1, 50, 99, 99.9]))
    print("resolve took", time.time() - t0, "s")

    # img_u8 = to_uint8_gray(img)
    # H, W = img_u8.shape[:2]

    # print("final image shape =", img_u8.shape)

    # # Your outlines are high-res because positions/side length were generated with scale.
    # # Your final image is downsampled by AveragePooling(scale, scale, 1),
    # # so divide outlines by scale.
    # outlines = [np.asarray(outline, dtype=float) / scale for outline in sampler.outlines_px]

    # lines = outlines_to_yolo(outlines, H=H, W=W, class_id=class_id)

    img_u8 = to_uint8_gray(img)

    outlines = [np.asarray(outline, dtype=float) / scale for outline in sampler.outlines_px]

    H, W = img_u8.shape[:2]
    crop_w = 100
    crop_h = 100
    x0 = (W - crop_w) // 2
    y0 = (H - crop_h) // 2

    img_u8, outlines = crop_image_and_outlines(
        img_u8,
        outlines,
        crop_xywh=(x0, y0, crop_w, crop_h),
        clip_polygons=True,
    )

    H, W = img_u8.shape[:2]
    lines = outlines_to_yolo(outlines, H=H, W=W, class_id=class_id)

    img_path = img_dir / f"{stem}.png"
    lab_path = lab_dir / f"{stem}.txt"
    overlay_path = overlay_dir / f"{stem}_overlay.png"

    cv2.imwrite(str(img_path), img_u8)
    lab_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print("wrote image:", img_path)
    print("wrote label:", lab_path)
    print("number of labels:", len(lines))

    if show_overlay:
        plt.figure(figsize=(7, 7))
        plt.imshow(img_u8, cmap="gray")

        for poly in outlines:
            poly = np.asarray(poly, dtype=float)
            if len(poly) < 3:
                continue

            poly_closed = np.vstack([poly, poly[0]])
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], linewidth=0.6)
            plt.scatter(poly[:, 0], poly[:, 1], s=1)

        plt.title(f"{stem} | outlines={len(outlines)}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
        plt.show()

        print("wrote overlay:", overlay_path)

    return img_u8, outlines, lines

# def show_overlay(img_u8, outlines_px, title="overlay"):
#     plt.figure(figsize=(6,6))
#     plt.imshow(img_u8, cmap="gray")

#     for poly in outlines_px:
#         if poly is None or len(poly) < 3:
#             continue
#         poly = np.asarray(poly)

#         # close the polygon
#         x = np.r_[poly[:,0], poly[0,0]]
#         y = np.r_[poly[:,1], poly[0,1]]

#         plt.plot(x, y, linewidth=1)

#     plt.title(title)
#     plt.axis("off")
#     plt.show()
    
# def generate_one_sample(class_id=0, return_outlines=False):
#     rot_sampler_fn, pos_sampler_fn, N = cluster_positions()

#     sampler = TrianglePrismSampler(
#         H=2048, W=2048, margin=50, side_length_px=30, thickness_px=8,
#         rot_sampler_fn=rot_sampler_fn,
#         pos_sampler_fn=pos_sampler_fn,
#     )

#     # get pipeline + parts
#     image_of_particles = generate_image(
#         sampler=sampler,
#         pos_sampler=sampler.pos_sampler,
#         rot_sampler=sampler.rot_sampler,
#         N=N,
#     )

#     sampler.reset()

#     # Resolve the pipeline -> should be a single image (not a list)
#     img = image_of_particles.update().resolve()
#     img_u8 = to_uint8_gray(img)

#     # outlines were collected during sampling
#     outlines_px = sampler.outlines_px

#     # overlay check
#     show_overlay(img_u8, outlines_px, title="overlay from memory")

#     H, W = img_u8.shape[:2]
#     lines = outlines_to_yolo(outlines_px, H=H, W=W, class_id=class_id)

#     if return_outlines:
#         return img_u8, lines, outlines_px
#     return img_u8, lines

# import matplotlib.pyplot as plt
# # %%

# """Testing without writing"""



# img_u8, lines = generate_one_sample()
# print("img shape:", img_u8.shape)
# print("num label lines:", len(lines))
# print(lines[0][:80] if lines else "no labels")

# plt.figure(figsize=(6,6))
# plt.imshow(img_u8, cmap="gray")
# plt.axis("off")
# plt.show()

# #%%

# from pathlib import Path
# import cv2

# out_dir = Path(r"C:\Testtest")
# split = "train"
# img_dir = out_dir / "images" / split
# lab_dir = out_dir / "labels" / split
# img_dir.mkdir(parents=True, exist_ok=True)
# lab_dir.mkdir(parents=True, exist_ok=True)

# for i in range(1):
#     img_u8, lines = generate_one_sample()
#     stem = f"{i:06d}"
#     ok = cv2.imwrite(str(img_dir / f"{stem}.png"), img_u8)
#     (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
#     print("wrote", stem, "png_ok=", ok)

# print("done ->", img_dir, lab_dir)
# # %%


# def yolo_poly_txt_to_outlines(txt_path, W=2048, H=2048):
#     outlines = []
#     for line in Path(txt_path).read_text().strip().splitlines():
#         parts = line.split()
#         if len(parts) < 3:
#             continue
#         coords = list(map(float, parts[1:]))  # skip class id
#         pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
#         pts[:, 0] *= W
#         pts[:, 1] *= H
#         outlines.append(pts)
#     return outlines
# #%%

# stem = "000000"  # pick one you saved
# img = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
# outlines_px = yolo_poly_txt_to_outlines(lab_dir / f"{stem}.txt", W=2048, H=2048)

# show_overlay(img, outlines_px, title=f"overlay from files {stem}")
# # %%

# %%
