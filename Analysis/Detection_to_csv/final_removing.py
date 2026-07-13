import json
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _image_to_base64_png(image):
    """
    Convert grayscale or RGB image array to base64 PNG for embedding in HTML.
    """
    arr = np.asarray(image)

    if arr.dtype != np.uint8:
        arr = arr.astype(float)
        arr = arr - np.nanmin(arr)
        arr = arr / max(np.nanmax(arr), 1e-12)
        arr = (255 * arr).astype(np.uint8)

    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3:
        im = Image.fromarray(arr)
    else:
        raise ValueError("image must be 2D grayscale or 3D RGB/RGBA")

    buffer = BytesIO()
    im.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_polygon_value(value):
    """
    Safely parse polygon columns that may be:
    - JSON string
    - Python list
    - NaN
    """
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    if isinstance(value, str):
        try:
            poly = json.loads(value)
        except Exception:
            return None
    else:
        poly = value

    try:
        poly = np.asarray(poly, dtype=float)
    except Exception:
        return None

    if poly.ndim != 2 or poly.shape[1] != 2:
        return None

    return poly.tolist()

def make_remove_particles_html(
    image,
    roi_df,
    output_html,
    output_csv_name="particles_to_remove.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    polygon_col=None,
    point_radius=4,
):
    """
    Make an interactive HTML file for manually marking particles for removal.

    Controls
    --------
    Left click particle : toggle remove / keep
    Mouse wheel         : zoom
    Drag mouse          : pan
    U button            : undo last click
    Download CSV button : saves particles_to_remove.csv

    Output CSV columns
    ------------------
    particle_id, x, y, remove
    """

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    df = roi_df.copy().reset_index(drop=True)

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    records = []

    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        rec = {
            "particle_id": row[id_col],
            "x": float(x),
            "y": float(y),
            "polygon": None,
        }

        if polygon_col is not None and polygon_col in df.columns:
            rec["polygon"] = _parse_polygon_value(row[polygon_col])

        records.append(rec)

    image_b64 = _image_to_base64_png(image)

    data_json = json.dumps(records)
    image_src = f"data:image/png;base64,{image_b64}"

    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Remove particles</title>

<style>
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
}

#toolbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: #222;
    padding: 8px;
    z-index: 10;
    display: flex;
    gap: 10px;
    align-items: center;
}

button {
    padding: 6px 10px;
    font-size: 14px;
}

#status {
    margin-left: 10px;
}

canvas {
    position: absolute;
    top: 48px;
    left: 0;
    background: black;
    cursor: crosshair;
}
</style>
</head>

<body>

<div id="toolbar">
    <button onclick="downloadCSV()">Download removal CSV</button>
    <button onclick="undoLast()">Undo</button>
    <button onclick="resetView()">Reset view</button>
    <span id="status">Loading...</span>
</div>

<canvas id="canvas"></canvas>

<script>
const particles = __PARTICLE_DATA__;
const outputCsvName = "__OUTPUT_CSV_NAME__";
const imageSrc = "__IMAGE_SRC__";
const pointRadius = __POINT_RADIUS__;

let removed = new Set();
let history = [];

let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let img = new Image();

let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragOffsetX = 0;
let dragOffsetY = 0;
let movedDuringDrag = false;

img.onload = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 48;

    offsetX = 0;
    offsetY = 0;
    scale = Math.min(canvas.width / img.width, canvas.height / img.height);

    draw();
};

img.src = imageSrc;

window.addEventListener("resize", function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 48;
    draw();
});

function imageToScreen(x, y) {
    return {
        x: x * scale + offsetX,
        y: y * scale + offsetY
    };
}

function screenToImage(x, y) {
    return {
        x: (x - offsetX) / scale,
        y: (y - offsetY) / scale
    };
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(
        img,
        offsetX,
        offsetY,
        img.width * scale,
        img.height * scale
    );

    for (const p of particles) {
        const isRemoved = removed.has(String(p.particle_id));

        if (p.polygon !== null) {
            ctx.beginPath();

            for (let k = 0; k < p.polygon.length; k++) {
                const q = imageToScreen(p.polygon[k][0], p.polygon[k][1]);
                if (k === 0) {
                    ctx.moveTo(q.x, q.y);
                } else {
                    ctx.lineTo(q.x, q.y);
                }
            }

            ctx.closePath();
            ctx.lineWidth = isRemoved ? 2.5 : 1.0;
            ctx.strokeStyle = isRemoved ? "red" : "cyan";
            ctx.globalAlpha = isRemoved ? 0.95 : 0.5;
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }

        const s = imageToScreen(p.x, p.y);

        ctx.beginPath();
        ctx.arc(s.x, s.y, pointRadius, 0, 2 * Math.PI);

        if (isRemoved) {
            ctx.fillStyle = "red";
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1.5;
        } else {
            ctx.fillStyle = "lime";
            ctx.strokeStyle = "black";
            ctx.lineWidth = 1.0;
        }

        ctx.fill();
        ctx.stroke();
    }

    document.getElementById("status").innerText =
        "Marked for removal: " + removed.size + " / " + particles.length;
}

function findNearestParticle(screenX, screenY) {
    let best = null;
    let bestDist = Infinity;

    for (const p of particles) {
        const s = imageToScreen(p.x, p.y);
        const dx = s.x - screenX;
        const dy = s.y - screenY;
        const d = Math.sqrt(dx * dx + dy * dy);

        if (d < bestDist) {
            bestDist = d;
            best = p;
        }
    }

    const maxClickDist = Math.max(10, pointRadius * 3);

    if (bestDist <= maxClickDist) {
        return best;
    }

    return null;
}

canvas.addEventListener("mousedown", function(e) {
    isDragging = true;
    movedDuringDrag = false;

    dragStartX = e.clientX;
    dragStartY = e.clientY;

    dragOffsetX = offsetX;
    dragOffsetY = offsetY;
});

canvas.addEventListener("mousemove", function(e) {
    if (!isDragging) return;

    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;

    if (Math.abs(dx) + Math.abs(dy) > 3) {
        movedDuringDrag = true;
    }

    offsetX = dragOffsetX + dx;
    offsetY = dragOffsetY + dy;

    draw();
});

canvas.addEventListener("mouseup", function(e) {
    if (!isDragging) return;

    isDragging = false;

    if (movedDuringDrag) return;

    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    const p = findNearestParticle(sx, sy);

    if (p === null) return;

    const id = String(p.particle_id);

    if (removed.has(id)) {
        removed.delete(id);
        history.push({id: id, action: "keep"});
    } else {
        removed.add(id);
        history.push({id: id, action: "remove"});
    }

    draw();
});

canvas.addEventListener("wheel", function(e) {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const before = screenToImage(mouseX, mouseY);

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    scale *= zoomFactor;

    const after = imageToScreen(before.x, before.y);

    offsetX += mouseX - after.x;
    offsetY += mouseY - after.y;

    draw();
});

function undoLast() {
    if (history.length === 0) return;

    const last = history.pop();

    if (last.action === "remove") {
        removed.delete(last.id);
    } else {
        removed.add(last.id);
    }

    draw();
}

function resetView() {
    scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    offsetX = 0;
    offsetY = 0;
    draw();
}

function downloadCSV() {
    let lines = [];
    lines.push("particle_id,x,y,remove");

    for (const p of particles) {
        const id = String(p.particle_id);

        if (removed.has(id)) {
            lines.push(
                p.particle_id + "," +
                p.x + "," +
                p.y + "," +
                "1"
            );
        }
    }

    const csv = lines.join("\n");
    const blob = new Blob([csv], {type: "text/csv"});
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = outputCsvName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}
</script>

</body>
</html>
"""

    html = (
        html_template
        .replace("__PARTICLE_DATA__", data_json)
        .replace("__OUTPUT_CSV_NAME__", output_csv_name)
        .replace("__IMAGE_SRC__", image_src)
        .replace("__POINT_RADIUS__", str(point_radius))
    )

    output_html.write_text(html, encoding="utf-8")

    return str(output_html)

def remove_particles_from_roi_df(
    roi_df_or_csv,
    removal_csv,
    id_col="particle_id",
    x_col="x",
    y_col="y",
    save_path=None,
):
    """
    Remove particles from an ROI dataframe using the CSV exported from
    make_remove_particles_html.

    Parameters
    ----------
    roi_df_or_csv : pandas.DataFrame or str/path
        Original ROI dataframe or path to ROI CSV.

    removal_csv : str/path
        CSV downloaded from the HTML tool.

    id_col : str
        Particle ID column used to match particles.

    x_col, y_col : str
        Coordinate columns. Only used as fallback if id_col is unavailable.

    save_path : str/path or None
        If given, saves final ROI dataframe to this path.

    Returns
    -------
    final_roi_df : pandas.DataFrame
        ROI dataframe with selected particles removed.
    """

    if isinstance(roi_df_or_csv, (str, Path)):
        roi_df = pd.read_csv(roi_df_or_csv)
    else:
        roi_df = roi_df_or_csv.copy()

    removal_df = pd.read_csv(removal_csv)

    removal_df = removal_df[removal_df["remove"].astype(int) == 1].copy()

    if len(removal_df) == 0:
        final_roi_df = roi_df.copy()
    else:
        if id_col in roi_df.columns and id_col in removal_df.columns:
            remove_ids = set(removal_df[id_col].astype(str))
            keep_mask = ~roi_df[id_col].astype(str).isin(remove_ids)
            final_roi_df = roi_df.loc[keep_mask].copy()

        else:
            # fallback: remove by exact x/y coordinate match
            remove_xy = set(
                zip(
                    removal_df[x_col].round(6),
                    removal_df[y_col].round(6),
                )
            )

            roi_xy = list(
                zip(
                    roi_df[x_col].round(6),
                    roi_df[y_col].round(6),
                )
            )

            keep_mask = [xy not in remove_xy for xy in roi_xy]
            final_roi_df = roi_df.loc[keep_mask].copy()

    final_roi_df = final_roi_df.reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_roi_df.to_csv(save_path, index=False)

    print(f"Original ROI particles: {len(roi_df)}")
    print(f"Removed particles:      {len(roi_df) - len(final_roi_df)}")
    print(f"Final ROI particles:    {len(final_roi_df)}")

    return final_roi_df

