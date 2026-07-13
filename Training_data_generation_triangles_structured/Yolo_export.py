
import os, time, queue
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
import matplotlib.pyplot as plt
from Training_data_generation_triangles_structured.Image_creation.Image_generator import generate_image
from Training_data_generation_triangles_structured.Image_creation.Triangle_Sampler_Class import TrianglePrismSampler
from Training_data_generation_triangles_structured.Image_creation.particle_position.cluster_position import cluster_positions, clusterpositions_test_one




def to_uint8_gray(img, p_low=0, p_high=100, contrast=0.5, out_bg=0.7, dark_gain=1.8):
    img = np.squeeze(img).astype(np.float32)

    lo, hi = np.percentile(img, [p_low, p_high])

    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    # normalize to 0–1
    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0, 1)

    # reduce contrast around background level
    img = out_bg + contrast * (img - out_bg)
    img = np.clip(img, 0, 1)
    
    dark_mask = img < np.percentile(img, 10)
    img[dark_mask] = np.percentile(img, [10]) + dark_gain * (img[dark_mask] - np.percentile(img,10))

    return (255.0 * img).astype(np.uint8)


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


def worker_loop(out_q: Queue, class_id=0, scale=None):

    pixel_size_um = 0.32
    side_length_um = 10.0
    thickness_um = 2.5

    side_length_px = scale * side_length_um / pixel_size_um
    thickness_px = scale * thickness_um / pixel_size_um

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
                pos_sampler=sampler.pos_sampler,
                rot_sampler=sampler.rot_sampler,
                N=N_cluster,
                image_size=512*scale,
                scale=scale
            )

            sampler.reset()
            print("starting resolve", flush=True)
            t = time.time()
            img = image_of_particles.update().resolve()
            print(np.asarray(img).shape)


            print("resolve took", time.time() - t, "sec", flush=True)

            outlines = [np.asarray(outline, dtype=float) / scale for outline in sampler.outlines_px]

            img_u8 = to_uint8_gray(img)

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
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_dir, class_name=class_name)

    img_dir = out_dir / f"images/{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_index(img_dir)

    if num_workers is None:
        num_workers = max(1, min(4, cpu_count() - 1))  

    out_q = Queue(maxsize=queue_size)

    procs = []
    for _ in range(num_workers):
        p = Process(target=worker_loop, args=(out_q, class_id, scale), daemon=True)
        p.start()
        procs.append(p)

    writer = Process(target=writer_loop, args=(out_q, out_dir, split, start_idx), daemon=True)
    writer.start()

    print(f"Parallel export running with {num_workers} workers.")
    print(f"Writing to: {out_dir}  (split={split}, start={start_idx})")

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

    print("resolve took", time.time() - t0, "s")

    img_u8 = to_uint8_gray(img)

    outlines = [np.asarray(outline, dtype=float) / scale for outline in sampler.outlines_px]

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

