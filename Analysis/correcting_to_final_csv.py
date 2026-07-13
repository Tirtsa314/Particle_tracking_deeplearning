
# %%
from pathlib import Path
import numpy as np
import pandas as pd


from scipy.spatial import cKDTree
import numpy as np
import pandas as pd


def add_working_center_columns(df):
    """
    Adds x_center_work, y_center_work.

    Uses refined center if available.
    Falls back to original YOLO/detection center if refined center is NaN.
    """

    df = df.copy()

    if {"x_refined", "y_refined", "x", "y"}.issubset(df.columns):
        df["x_center_work"] = df["x_refined"].where(df["x_refined"].notna(), df["x"])
        df["y_center_work"] = df["y_refined"].where(df["y_refined"].notna(), df["y"])

    elif {"x_refined", "y_refined"}.issubset(df.columns):
        df["x_center_work"] = df["x_refined"]
        df["y_center_work"] = df["y_refined"]

    elif {"x", "y"}.issubset(df.columns):
        df["x_center_work"] = df["x"]
        df["y_center_work"] = df["y"]

    else:
        raise ValueError("Could not find x/y or x_refined/y_refined columns.")

    return df

def remove_close_duplicate_particles(
    df,
    x_col="x_refined",
    y_col="y_refined",
    duplicate_tol_px=3.0,
    prefer_existing=True,
):
    """
    Remove duplicate particles whose centers are within duplicate_tol_px.

    If prefer_existing=True:
        keep original fitted particles over manually added center-only particles.

    This is usually good because original particles have theta/poly/fit info,
    while manual particles only have x/y.
    """

    df = df.copy()

    coords = df[[x_col, y_col]].to_numpy(dtype=float)

    valid = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        return df

    valid_coords = coords[valid_indices]

    tree = cKDTree(valid_coords)
    close_pairs = tree.query_pairs(r=duplicate_tol_px)

    if len(close_pairs) == 0:
        print("No close duplicate particles found.")
        return df

    # Union-find to group connected duplicate particles
    parent = np.arange(len(valid_indices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in close_pairs:
        union(a, b)

    groups = {}
    for local_i, global_i in enumerate(valid_indices):
        root = find(local_i)
        groups.setdefault(root, []).append(global_i)

    rows_to_drop = []

    for group in groups.values():
        if len(group) <= 1:
            continue

        group_df = df.iloc[group].copy()

        # Decide which row to keep
        if prefer_existing and "manual_added" in group_df.columns:
            # Existing particles first, manual-added particles second
            group_df["_manual_sort"] = group_df["manual_added"].fillna(False).astype(bool).astype(int)
        else:
            group_df["_manual_sort"] = 0

        # Prefer rows with more information
        info_score = np.zeros(len(group_df))

        for col in ["poly_json", "vertices_json", "theta", "success"]:
            if col in group_df.columns:
                info_score += group_df[col].notna().to_numpy(dtype=float)

        group_df["_info_score"] = info_score

        # Sort:
        # 1. non-manual first
        # 2. rows with more info first
        group_df = group_df.sort_values(
            by=["_manual_sort", "_info_score"],
            ascending=[True, False],
        )

        keep_index = group_df.index[0]
        drop_indices = [idx for idx in group_df.index if idx != keep_index]

        rows_to_drop.extend(drop_indices)

    rows_to_drop = sorted(set(rows_to_drop))

    print(f"Close duplicate groups found: {sum(len(g) > 1 for g in groups.values())}")
    print(f"Removing close duplicate particles: {len(rows_to_drop)}")

    df = df.drop(index=rows_to_drop).copy()
    df = df.reset_index(drop=True)

    return df

def add_xcol_ycol(df):
    """
    Adds two columns:
    - x_col
    - y_col

    They contain:
    - x_refined/y_refined if both are valid numbers
    - otherwise x_old/y_old
    """

    df = df.copy()

    required = {"x_refined", "y_refined", "x_old", "y_old"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    refined_ok = (
        np.isfinite(df["x_refined"])
        & np.isfinite(df["y_refined"])
    )

    df["x_col"] = df["x_refined"].where(refined_ok, df["x_old"])
    df["y_col"] = df["y_refined"].where(refined_ok, df["y_old"])

    return df

def apply_manual_edits_by_coordinates(
    refined_csv,
    manual_edits_csv,
    output_csv=None,
    remove_tol_px=1.0,
):
    """
    Apply manual edits where:

    - remove_x, remove_y define particles to remove by coordinate
    - missing_x, missing_y define new particles to add

    This does NOT use remove_particle_id.
    """

    refined_csv = Path(refined_csv)
    manual_edits_csv = Path(manual_edits_csv)

    if output_csv is None:
        output_csv = refined_csv.with_name(refined_csv.stem + "_CORRECTED.csv")
    else:
        output_csv = Path(output_csv)

    det_df = pd.read_csv(refined_csv)
    edit_df = pd.read_csv(manual_edits_csv)

    print(f"Original detections: {len(det_df)}")
    print(f"Manual edit rows: {len(edit_df)}")

    # ------------------------------------------------------------
    # Decide which center columns to use in the detection CSV
    # ------------------------------------------------------------

    det_df = det_df.copy()
    det_df = add_xcol_ycol(det_df)
    x_col = "x_col"
    y_col = "y_col"
    # ------------------------------------------------------------
    # 1. Remove particles by coordinates
    # ------------------------------------------------------------
    remove_xy = (
        edit_df[["remove_x", "remove_y"]]
        .dropna()
        .to_numpy(dtype=float)
    )

    rows_to_remove = []

    for rx, ry in remove_xy:
        coords = det_df[[x_col, y_col]].to_numpy(dtype=float)

        if len(coords) == 0:
            break

        d = np.sqrt((coords[:, 0] - rx)**2 + (coords[:, 1] - ry)**2)
        nearest_i = int(np.argmin(d))
        nearest_dist = d[nearest_i]

        if nearest_dist <= remove_tol_px:
            rows_to_remove.append(det_df.index[nearest_i])
        else:
            print(
                f"Warning: no particle found near remove point "
                f"({rx:.2f}, {ry:.2f}); nearest distance = {nearest_dist:.2f} px"
            )

    rows_to_remove = sorted(set(rows_to_remove))

    det_df = det_df.drop(index=rows_to_remove).copy()

    print(f"Removed particles by coordinate: {len(rows_to_remove)}")

    # ------------------------------------------------------------
    # 2. Add missing particles as new rows
    # ------------------------------------------------------------
    missing_xy = (
        edit_df[["missing_x", "missing_y"]]
        .dropna()
        .to_numpy(dtype=float)
    )

    print(f"Particles to add: {len(missing_xy)}")

    if "manual_added" not in det_df.columns:
        det_df["manual_added"] = False

    if "manual_note" not in det_df.columns:
        det_df["manual_note"] = ""

    if "particle_id" not in det_df.columns:
        det_df["particle_id"] = np.arange(len(det_df), dtype=int)

    numeric_ids = pd.to_numeric(det_df["particle_id"], errors="coerce")
    next_particle_id = int(np.nanmax(numeric_ids)) + 1 if numeric_ids.notna().any() else 0

    new_rows = []

    for i, (mx, my) in enumerate(missing_xy):
        new_row = {col: np.nan for col in det_df.columns}

        new_row["particle_id"] = next_particle_id + i

        # Put the manually clicked center into all available center columns
        if "x" in det_df.columns:
            new_row["x"] = mx
        if "y" in det_df.columns:
            new_row["y"] = my

        if "x_old" in det_df.columns:
            new_row["x_old"] = mx
        if "y_old" in det_df.columns:
            new_row["y_old"] = my

        if "x_refined" in det_df.columns:
            new_row["x_refined"] = mx
        if "y_refined" in det_df.columns:
            new_row["y_refined"] = my

        # If your CSV has frame information, keep the same frame as the rest
        if "frame" in det_df.columns:
            frame_values = det_df["frame"].dropna()
            if len(frame_values) > 0:
                new_row["frame"] = frame_values.mode().iloc[0]

        new_row["manual_added"] = True
        new_row["manual_note"] = "manual_center_only"

        new_rows.append(new_row)

    added_df = pd.DataFrame(new_rows, columns=det_df.columns)

    corrected_df = pd.concat([det_df, added_df], ignore_index=True)

    # Sort by particle_id for neatness
    corrected_df = add_xcol_ycol(corrected_df)

    corrected_df = remove_close_duplicate_particles(
        corrected_df,
        x_col="x_col",
        y_col="y_col",
        duplicate_tol_px=3.0,
        prefer_existing=True,
    )

    corrected_df.to_csv(output_csv, index=False)

    print(f"Final corrected detections: {len(corrected_df)}")
    print(f"Saved to: {output_csv}")

    return corrected_df


# %%
refined_csv = r"c:\particle_csv_files\Hydrazine 010_refined_triangle_fits.csv"

manual_edits_csv = r"c:\particle_csv_files\Hydrazine 010_refined_triangle_fitsmanual_triangle_edits.csv"

output_csv = r"c:\particle_csv_files\Hydrazine 010_corrected_final.csv"

corrected_df = apply_manual_edits_by_coordinates(
    refined_csv=refined_csv,
    manual_edits_csv=manual_edits_csv,
    output_csv=output_csv,
    remove_tol_px=1.0,
)

corrected_df.head()
# %%
# %%
from pathlib import Path
import base64
import json
import webbrowser

import numpy as np
import pandas as pd
from PIL import Image
import nd2


def to_uint8(img, mode="percentile", p_low=1, p_high=99):
    """
    Convert image to uint8 for display.
    """
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img.copy()

    img = img.astype(np.float32)

    if mode == "divide256":
        img8 = np.clip(img / 256, 0, 255).astype(np.uint8)

    elif mode == "minmax":
        img = img - np.nanmin(img)
        mx = np.nanmax(img)
        if mx > 0:
            img = img / mx
        img8 = np.clip(255 * img, 0, 255).astype(np.uint8)

    elif mode == "percentile":
        lo, hi = np.nanpercentile(img, [p_low, p_high])
        if hi <= lo:
            hi = lo + 1.0
        img = (img - lo) / (hi - lo)
        img8 = np.clip(255 * img, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return img8


def load_nd2_frame_uint8(
    nd2_path,
    t=0,
    c=0,
    z=None,
    p=0,
    normalize_mode="percentile",
    p_low=1,
    p_high=99,
):
    """
    Load one 2D frame from an ND2 and convert it to uint8.
    """
    nd2_path = Path(nd2_path)

    with nd2.ND2File(nd2_path) as f:
        print("ND2 sizes:", f.sizes)
        print("ND2 shape:", f.shape)

        try:
            data = f.to_dask()
        except Exception:
            data = f.asarray()

        axes = list(f.sizes.keys())
        selection = []

        for ax in axes:
            size = f.sizes[ax]

            if ax in ["Y", "X"]:
                selection.append(slice(None))

            elif ax == "T":
                selection.append(size + t if t < 0 else t)

            elif ax == "C":
                selection.append(c)

            elif ax == "Z":
                selection.append(size // 2 if z is None else z)

            elif ax == "P":
                selection.append(p)

            else:
                selection.append(0)

        frame = data[tuple(selection)]

        if hasattr(frame, "compute"):
            frame = frame.compute()

    frame = np.asarray(frame)

    while frame.ndim > 2:
        frame = frame[0]

    arr8 = to_uint8(
        frame,
        mode=normalize_mode,
        p_low=p_low,
        p_high=p_high,
    )
    return arr8


def make_final_check_html(
    image,
    final_csv,
    output_html="final_particle_check.html",
):
    """
    Make a read-only HTML viewer for final particle checking.

    Features
    --------
    - image display
    - overlay final centers
    - overlay YOLO outlines if poly_json exists
    - zoom + pan
    - hover to see image pixel coordinates
    - optional particle IDs
    - optional highlighting of manual-added particles
    """

    # ------------------------------------------------------------
    # 1) Load image
    # ------------------------------------------------------------
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("L")
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = to_uint8(arr)
        img = Image.fromarray(arr).convert("L")

    W, H = img.size

    temp_png = Path("_temp_final_check_image.png")
    img.save(temp_png)

    with open(temp_png, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    temp_png.unlink()

    # ------------------------------------------------------------
    # 2) Load final CSV
    # ------------------------------------------------------------
    df = pd.read_csv(final_csv).copy()

    df = add_xcol_ycol(df)

    x_col = "x_col"
    y_col = "y_col"

    df = df[df[x_col].notna() & df[y_col].notna()].copy()

    if "particle_id" not in df.columns:
        df["particle_id"] = np.arange(len(df), dtype=int)

    if "manual_added" not in df.columns:
        df["manual_added"] = False

    detections = []
    for _, row in df.iterrows():
        poly = []

        if "poly_json" in df.columns and pd.notna(row.get("poly_json", np.nan)):
            try:
                raw_poly = json.loads(row["poly_json"])
                poly = [[round(float(x), 2), round(float(y), 2)] for x, y in raw_poly]
            except Exception:
                poly = []

        detections.append(
            {
                "particle_id": int(row["particle_id"]),
                "x": round(float(row[x_col]), 3),
                "y": round(float(row[y_col]), 3),
                "manual_added": bool(row["manual_added"]),
                "has_poly": len(poly) > 0,
                "poly": poly,
            }
        )

    detections_json = json.dumps(detections, separators=(",", ":"))

    # ------------------------------------------------------------
    # 3) Build HTML
    # ------------------------------------------------------------
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Final particle check</title>

<style>
    body {{
        font-family: Arial, sans-serif;
        background: #111;
        color: white;
        margin: 20px;
    }}

    canvas {{
        cursor: crosshair;
        border: 2px solid white;
        background: black;
        display: block;
        margin-top: 10px;
    }}

    button {{
        margin: 8px 5px 8px 0;
        padding: 8px 14px;
        font-size: 14px;
    }}

    label {{
        margin-right: 18px;
    }}

    #info {{
        margin-top: 10px;
        font-size: 14px;
        line-height: 1.5;
    }}

    .toolbar {{
        margin-bottom: 8px;
    }}

    .hint {{
        color: #bbb;
    }}
</style>
</head>

<body>

<h2>Final particle check</h2>

<div class="toolbar">
    <button onclick="resetView()">Reset view</button>
</div>

<div class="toolbar">
    <label><input type="checkbox" id="showPolys" checked onchange="draw()"> Show YOLO outlines</label>
    <label><input type="checkbox" id="showCenters" checked onchange="draw()"> Show centers</label>
    <label><input type="checkbox" id="showIds" onchange="draw()"> Show particle IDs</label>
    <label><input type="checkbox" id="showManual" checked onchange="draw()"> Highlight manual-added particles</label>
</div>

<p class="hint">
Mouse wheel = zoom<br>
Drag = pan<br>
Hover = show image pixel coordinates
</p>

<canvas id="canvas"></canvas>

<div id="info">
    Image size: {W} × {H} px<br>
    Total particles: <span id="nDetections">0</span><br>
    With polygon outline: <span id="nPolys">0</span><br>
    Manual-added: <span id="nManual">0</span><br>
    Zoom: <span id="zoom">1.00</span>x<br>
    Hover pixel: x = <span id="hoverX">-</span>, y = <span id="hoverY">-</span><br>
    Nearest particle: <span id="nearestInfo">-</span>
</div>

<script>
const imageWidth = {W};
const imageHeight = {H};
const detections = {detections_json};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

document.getElementById("nDetections").innerText = detections.length;
document.getElementById("nPolys").innerText = detections.filter(d => d.has_poly).length;
document.getElementById("nManual").innerText = detections.filter(d => d.manual_added).length;

const img = new Image();
img.src = "data:image/png;base64,{img_base64}";

let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

let hoverScreenX = null;
let hoverScreenY = null;
let hoverImgX = null;
let hoverImgY = null;

canvas.width = Math.min(window.innerWidth * 0.92, 1400);
canvas.height = Math.min(window.innerHeight * 0.75, 950);

img.onload = function() {{
    resetView();
}};

function resetView() {{
    const scaleX = canvas.width / imageWidth;
    const scaleY = canvas.height / imageHeight;
    scale = Math.min(scaleX, scaleY);

    offsetX = (canvas.width - imageWidth * scale) / 2;
    offsetY = (canvas.height - imageHeight * scale) / 2;

    draw();
}}

function imageToScreen(x, y) {{
    return {{
        x: offsetX + x * scale,
        y: offsetY + y * scale
    }};
}}

function screenToImage(x, y) {{
    return {{
        x: (x - offsetX) / scale,
        y: (y - offsetY) / scale
    }};
}}

function drawPolygon(poly, strokeStyle, lineWidth=1) {{
    if (!poly || poly.length < 2) return;

    ctx.beginPath();
    const p0 = imageToScreen(poly[0][0], poly[0][1]);
    ctx.moveTo(p0.x, p0.y);

    for (let i = 1; i < poly.length; i++) {{
        const p = imageToScreen(poly[i][0], poly[i][1]);
        ctx.lineTo(p.x, p.y);
    }}

    ctx.closePath();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
}}

function drawCenter(x, y, color="cyan", radius=3) {{
    const p = imageToScreen(x, y);

    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(p.x - radius - 3, p.y);
    ctx.lineTo(p.x + radius + 3, p.y);
    ctx.moveTo(p.x, p.y - radius - 3);
    ctx.lineTo(p.x, p.y + radius + 3);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();
}}

function findNearestDetectionInScreen(screenX, screenY, maxScreenDist=20) {{
    let best = null;
    let bestDist2 = maxScreenDist * maxScreenDist;

    for (const det of detections) {{
        const p = imageToScreen(det.x, det.y);
        const dx = p.x - screenX;
        const dy = p.y - screenY;
        const d2 = dx * dx + dy * dy;

        if (d2 <= bestDist2) {{
            bestDist2 = d2;
            best = {{
                det: det,
                distScreen: Math.sqrt(d2),
            }};
        }}
    }}

    return best;
}}

function drawHoverCrosshair() {{
    if (hoverScreenX === null || hoverScreenY === null) return;

    ctx.beginPath();
    ctx.moveTo(hoverScreenX, 0);
    ctx.lineTo(hoverScreenX, canvas.height);
    ctx.moveTo(0, hoverScreenY);
    ctx.lineTo(canvas.width, hoverScreenY);
    ctx.strokeStyle = "rgba(255,255,0,0.25)";
    ctx.lineWidth = 1;
    ctx.stroke();
}}

function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.drawImage(
        img,
        offsetX,
        offsetY,
        imageWidth * scale,
        imageHeight * scale
    );

    const showPolys = document.getElementById("showPolys").checked;
    const showCenters = document.getElementById("showCenters").checked;
    const showIds = document.getElementById("showIds").checked;
    const showManual = document.getElementById("showManual").checked;

    for (const det of detections) {{
        const color = (showManual && det.manual_added) ? "yellow" : "cyan";
        const polyColor = det.manual_added ? "rgba(255,220,0,0.7)" : "rgba(0,255,0,0.5)";

        if (showPolys && det.poly && det.poly.length > 0) {{
            drawPolygon(det.poly, polyColor, det.manual_added ? 2 : 1);
        }}

        if (showCenters) {{
            drawCenter(det.x, det.y, color, det.manual_added ? 4 : 3);
        }}

        if (showIds) {{
            const p = imageToScreen(det.x, det.y);
            ctx.fillStyle = det.manual_added ? "yellow" : "white";
            ctx.font = "10px Arial";
            ctx.fillText(det.particle_id, p.x + 6, p.y - 6);
        }}
    }}

    drawHoverCrosshair();

    document.getElementById("zoom").innerText = scale.toFixed(2);

    if (
        hoverImgX !== null &&
        hoverImgY !== null &&
        hoverImgX >= 0 && hoverImgX <= imageWidth &&
        hoverImgY >= 0 && hoverImgY <= imageHeight
    ) {{
        document.getElementById("hoverX").innerText = hoverImgX.toFixed(2);
        document.getElementById("hoverY").innerText = hoverImgY.toFixed(2);

        const nearest = findNearestDetectionInScreen(hoverScreenX, hoverScreenY, 18);
        if (nearest) {{
            const d = nearest.det;
            document.getElementById("nearestInfo").innerText =
                `id=${{d.particle_id}}, x=${{d.x.toFixed(2)}}, y=${{d.y.toFixed(2)}}, manual=${{d.manual_added}}`;
        }} else {{
            document.getElementById("nearestInfo").innerText = "-";
        }}
    }} else {{
        document.getElementById("hoverX").innerText = "-";
        document.getElementById("hoverY").innerText = "-";
        document.getElementById("nearestInfo").innerText = "-";
    }}
}}

// zoom
canvas.addEventListener("wheel", function(event) {{
    event.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const imgPt = screenToImage(mouseX, mouseY);

    const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;
    const newScale = Math.min(Math.max(scale * zoomFactor, 0.1), 60);

    scale = newScale;

    offsetX = mouseX - imgPt.x * scale;
    offsetY = mouseY - imgPt.y * scale;

    draw();
}}, {{ passive: false }});

// pan start
canvas.addEventListener("mousedown", function(event) {{
    isDragging = true;

    const rect = canvas.getBoundingClientRect();
    dragStartX = event.clientX - rect.left;
    dragStartY = event.clientY - rect.top;
}});

// pan move
canvas.addEventListener("mousemove", function(event) {{
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    hoverScreenX = mouseX;
    hoverScreenY = mouseY;

    const pt = screenToImage(mouseX, mouseY);
    hoverImgX = pt.x;
    hoverImgY = pt.y;

    if (isDragging) {{
        const dx = mouseX - dragStartX;
        const dy = mouseY - dragStartY;

        offsetX += dx;
        offsetY += dy;

        dragStartX = mouseX;
        dragStartY = mouseY;
    }}

    draw();
}});

// pan end
canvas.addEventListener("mouseup", function() {{
    isDragging = false;
}});
canvas.addEventListener("mouseleave", function() {{
    isDragging = false;
}});
</script>

</body>
</html>
"""

    output_html = Path(output_html)
    output_html.write_text(html, encoding="utf-8")

    print(f"Created: {output_html.resolve()}")
    return output_html.resolve()


def open_final_check_from_nd2(
    nd2_path,
    final_csv,
    output_html="final_particle_check.html",
    t=0,
    c=0,
    z=None,
    p=0,
    normalize_mode="percentile",
    p_low=1,
    p_high=99,
):
    """
    Convenience wrapper:
    - load frame from ND2
    - build final-check HTML
    - open it in browser
    """
    arr8 = load_nd2_frame_uint8(
        nd2_path=nd2_path,
        t=t,
        c=c,
        z=z,
        p=p,
        normalize_mode=normalize_mode,
        p_low=p_low,
        p_high=p_high,
    )

    html_path = make_final_check_html(
        image=arr8,
        final_csv=final_csv,
        output_html=output_html,
    )

    webbrowser.open(Path(html_path).as_uri())
    return html_path
# %%
# %%
nd2_path = r"c:\Users\Public\Hydrazine 010.nd2"
final_csv = r"c:\particle_csv_files\Hydrazine 010_corrected_final.csv"

html_path = open_final_check_from_nd2(
    nd2_path=nd2_path,
    final_csv=final_csv,
    output_html="Hydrazine_010_final_check.html",
    t=0,                  # or -1 for last frame
    c=0,
    z=None,
    p=0,
    normalize_mode="percentile",
    p_low=1,
    p_high=99,
)

print(html_path)
# %%
