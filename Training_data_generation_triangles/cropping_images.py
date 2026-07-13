#%% 
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon, box, MultiPolygon

# --------------------------------------------------
# Settings
# --------------------------------------------------
src_root = Path(r"C:\Train_set_small_28-03")   # your current dataset root
dst_root = Path(r"C:\Train_set_small_28-03_tiled512")

tile_size = 512
img_size = 1024   # original image size
splits = ["train", "val"]   # add "test" if you have it

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def parse_yolo_seg_line(line, W, H):
    """
    YOLO seg line format:
    class x1 y1 x2 y2 x3 y3 ...
    coordinates are normalized to [0,1]
    """
    parts = line.strip().split()
    if len(parts) < 7:
        return None  # need class + at least 3 points
    cls = int(float(parts[0]))
    coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= W
    coords[:, 1] *= H
    return cls, coords

def polygon_to_yolo_line(cls, poly_coords, tile_x, tile_y, tile_w, tile_h):
    """
    Convert absolute polygon coords back to normalized coords inside one tile.
    """
    pts = np.array(poly_coords, dtype=np.float32)

    # move into tile-local coords
    pts[:, 0] -= tile_x
    pts[:, 1] -= tile_y

    # normalize to tile size
    pts[:, 0] /= tile_w
    pts[:, 1] /= tile_h

    # clip numeric noise
    pts = np.clip(pts, 0.0, 1.0)

    if len(pts) < 3:
        return None

    flat = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
    return f"{cls} {flat}"

def clip_polygon_to_tile(coords, tile_rect):
    """
    Clip polygon to tile rectangle using shapely.
    Returns list of polygons (usually one, sometimes multiple).
    """
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)

    inter = poly.intersection(tile_rect)

    if inter.is_empty:
        return []

    polys = []
    if isinstance(inter, Polygon):
        polys = [inter]
    elif isinstance(inter, MultiPolygon):
        polys = list(inter.geoms)
    else:
        return []

    out = []
    for p in polys:
        if p.area <= 1.0:   # skip tiny fragments
            continue
        xy = np.array(p.exterior.coords[:-1], dtype=np.float32)  # remove repeated last point
        if len(xy) >= 3:
            out.append(xy)
    return out

# --------------------------------------------------
# Main tiling
# --------------------------------------------------
for split in splits:
    img_dir = src_root / "images" / split
    lbl_dir = src_root / "labels" / split

    out_img_dir = dst_root / "images" / split
    out_lbl_dir = dst_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

    for img_path in image_files:
        label_path = lbl_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read {img_path}")
            continue

        H, W = img.shape[:2]
        if (H, W) != (img_size, img_size):
            print(f"Skipping {img_path.name}: expected {img_size}x{img_size}, got {W}x{H}")
            continue

        lines = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

        objects = []
        for line in lines:
            parsed = parse_yolo_seg_line(line, W, H)
            if parsed is not None:
                objects.append(parsed)

        # 4 non-overlapping tiles
        tiles = [
            (0,   0,   tile_size, tile_size),   # top-left
            (512, 0,   tile_size, tile_size),   # top-right
            (0,   512, tile_size, tile_size),   # bottom-left
            (512, 512, tile_size, tile_size),   # bottom-right
        ]
        tile_names = ["tl", "tr", "bl", "br"]

        for (tx, ty, tw, th), suffix in zip(tiles, tile_names):
            tile_img = img[ty:ty+th, tx:tx+tw].copy()
            tile_box = box(tx, ty, tx + tw, ty + th)

            out_lines = []
            for cls, coords in objects:
                clipped_polys = clip_polygon_to_tile(coords, tile_box)
                for cp in clipped_polys:
                    yolo_line = polygon_to_yolo_line(cls, cp, tx, ty, tw, th)
                    if yolo_line is not None:
                        out_lines.append(yolo_line)

            out_img_path = out_img_dir / f"{img_path.stem}_{suffix}{img_path.suffix}"
            out_lbl_path = out_lbl_dir / f"{img_path.stem}_{suffix}.txt"

            cv2.imwrite(str(out_img_path), tile_img)

            with open(out_lbl_path, "w", encoding="utf-8") as f:
                for ln in out_lines:
                    f.write(ln + "\n")

        print(f"Done: {img_path.name}")
# %%
