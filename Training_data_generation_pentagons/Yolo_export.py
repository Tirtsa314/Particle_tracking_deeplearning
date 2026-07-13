
import os, time, queue
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
from Training_data_generation_pentagons.Image_creation.Image_generator import generate_image
from Training_data_generation_pentagons.Image_creation.Pentagon_Sampler_Class import PentagonalPrismSampler
from Training_data_generation_pentagons.Image_creation.particle_position.cluster_position import cluster_positions
from deeptrack import units as u
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
# --- your helpers (same as before) ---



def to_uint8_gray(img):
    img = np.squeeze(img).astype(np.float32)

    # # plot raw image before processing
    # plt.figure(figsize=(6, 6))
    # plt.imshow(img, cmap="gray")
    # plt.title("Raw image before background/gain transform")
    # plt.colorbar()
    # plt.axis("off")
    # plt.show()

    raw_bg = np.median(img)
    out_bg = 0.7
    gain = 0.5  # try 0.14, 0.16, 0.18

    # darker-than-background particles
    # img = out_bg + gain * (img - raw_bg)
    dev = img - raw_bg
    neg = np.minimum(dev, 0.0)   # darker-than-background part
    pos = np.maximum(dev, 0.0)   # brighter-than-background part

    gain_neg = 0.5
    gain_pos = 0.5
    knee = 0.3   # smaller = stronger highlight compression

    img = (
        out_bg
        + gain_neg * neg
        + gain_pos * knee * np.tanh(pos / knee)
    )

    # tiny blur
    # img = cv2.GaussianBlur(img, (0, 0), 0.1)



    # raw_bg = np.median(img)
    # dev = img - raw_bg

    # out_bg = 0.80

    # # compress extremes, but keep dark edges stronger than bright side
    # img = out_bg + np.where(
    #     dev < 0,
    #     0.12 * np.tanh(dev / 0.06),   # dark side: preserve edges
    #     0.05 * np.tanh(dev / 0.06)    # bright side: flatten strong contrast
    # )

    # img = gaussian_filter(img, sigma=0.9)
    # print("min:", img.min())
    # print("max:", img.max())
    # print("percentiles:", np.percentile(img, [0.1, 1, 5, 50, 95, 99, 99.9]))

    # print("fraction < 0 :", np.mean(img < 0))
    # print("fraction > 1 :", np.mean(img > 1))

    img = np.clip(img, 0, 1)

    return (255.0 * img).astype(np.uint8)





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


def write_data_yaml(out_dir: Path, class_name="pentagon"):
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
    side_length_um = 7.0
    thickness_um = 2.0
    side_length_px = scale * side_length_um / pixel_size_um
    thickness_px = scale * thickness_um / pixel_size_um

    print("|Worker loop started", flush=True)

    try:
        while True:
            rot_sampler_fn, pos_sampler_fn, N_cluster = cluster_positions(scale)
            # MAX_N = 150
            # N = np.random.randint(10, min(N_cluster, MAX_N))
            sampler = PentagonalPrismSampler(
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
            # img_raw = np.squeeze(img)

            # plt.figure(figsize=(6,6))
            # plt.imshow(img_raw, cmap="gray")
            # plt.colorbar()
            # plt.show()


            print("resolve took", time.time() - t, "sec", flush=True)
            outlines = [outline / scale for outline in sampler.outlines_px]

            H, W = img.shape[:2]
            lines = outlines_to_yolo(outlines, H=H, W=W, class_id=class_id)

            img_u8 = to_uint8_gray(img)

            out_q.put((img_u8, lines))
            print("Produced 1 sample")

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
    class_name="pentagon",
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
