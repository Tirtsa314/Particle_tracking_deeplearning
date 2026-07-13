import base64
import io
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image



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


def make_particle_roi_selection_html(
    image,
    det_df,
    output_html="select_particle_region.html",
    output_csv_name="selected_particles.csv",
    output_roi_csv_name="roi_vertices.csv",
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    point_radius=2.0,
):
    """
    Create an HTML tool to manually draw a free region/polygon around particles.

    The HTML can download:
    - selected_particles.csv  -> same columns as det_df, only particles inside ROI
    - roi_vertices.csv        -> polygon vertices in pixel and micron coordinates
    """

    # -------------------------
    # Prepare image
    # -------------------------
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("L")
        arr = np.asarray(img)
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = to_uint8(arr)
        img = Image.fromarray(arr)

    W, H = img.size

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    # -------------------------
    # Prepare dataframe
    # -------------------------
    df = det_df.copy()

    valid = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    df = df[valid].reset_index(drop=True)

    columns = list(df.columns)

    # Store whole dataframe in the HTML
    rows_json = df.to_json(orient="records", double_precision=15)
    columns_json = json.dumps(columns)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Particle ROI Selector</title>

<style>
body {{
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    margin: 18px;
}}

button {{
    margin: 4px;
    padding: 8px 12px;
    font-size: 14px;
}}

canvas {{
    background: white;
    border: 2px solid #ccc;
    cursor: crosshair;
}}

#info {{
    margin-top: 10px;
    line-height: 1.5;
}}

.hint {{
    color: #bbb;
    font-size: 14px;
}}
</style>
</head>

<body>

<h2>Particle ROI Selector</h2>

<div>
    <button onclick="setMode('polygon')">Polygon click mode</button>
    <button onclick="setMode('freehand')">Freehand draw mode</button>
    <button onclick="setMode('pan')">Pan mode</button>
    <button onclick="closePolygon()">Close polygon</button>
    <button onclick="clearROI()">Clear ROI</button>
    <button onclick="resetView()">Reset view</button>
    <button onclick="downloadSelectedCSV()">Download selected particles CSV</button>
    <button onclick="downloadROICSV()">Download ROI vertices CSV</button>
</div>

<p class="hint">
<b>Polygon mode:</b> click points around your region, then click “Close polygon”.<br>
<b>Freehand mode:</b> drag with the mouse to draw a free shape.<br>
<b>Pan mode:</b> drag to move the image. Mouse wheel zooms in/out in all modes.
</p>

<canvas id="canvas"></canvas>

<div id="info">
    Mode: <span id="modeInfo"></span><br>
    Total particles: <span id="totalParticles"></span><br>
    Selected particles: <span id="selectedParticles">0</span><br>
    Mouse image position: x = <span id="mouseX">-</span> px,
    y = <span id="mouseY">-</span> px
</div>

<script>
const imgW = {W};
const imgH = {H};
const umPerPx = {um_per_px};

const rows = {rows_json};
const columns = {columns_json};

const xCol = "{x_col}";
const yCol = "{y_col}";

const outputCsvName = "{output_csv_name}";
const outputRoiCsvName = "{output_roi_csv_name}";

const pointRadius = {point_radius};

const image = new Image();
image.src = "data:image/png;base64,{image_b64}";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

canvas.width = Math.min(window.innerWidth * 0.92, 1200);
canvas.height = Math.min(window.innerHeight * 0.75, 850);

let scale = 1.0;
let offsetX = 0.0;
let offsetY = 0.0;

let mode = "polygon";

let polygon = [];
let polygonClosed = false;

let isDrawingFreehand = false;
let isPanning = false;
let lastMouseX = 0;
let lastMouseY = 0;

document.getElementById("totalParticles").innerText = rows.length;
document.getElementById("modeInfo").innerText = mode;

function setMode(newMode) {{
    mode = newMode;
    document.getElementById("modeInfo").innerText = mode;

    if (mode === "pan") {{
        canvas.style.cursor = "grab";
    }} else {{
        canvas.style.cursor = "crosshair";
    }}
}}

function imageToScreen(x, y) {{
    return {{
        x: offsetX + x * scale,
        y: offsetY + y * scale
    }};
}}

function screenToImage(sx, sy) {{
    return {{
        x: (sx - offsetX) / scale,
        y: (sy - offsetY) / scale
    }};
}}

function resetView() {{
    const fitScale = Math.min(canvas.width / imgW, canvas.height / imgH);
    scale = fitScale;

    offsetX = 0.5 * (canvas.width - imgW * scale);
    offsetY = 0.5 * (canvas.height - imgH * scale);

    draw();
}}

function pointInPolygon(x, y, poly) {{
    let inside = false;

    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {{
        const xi = poly[i].x;
        const yi = poly[i].y;
        const xj = poly[j].x;
        const yj = poly[j].y;

        const intersect =
            ((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi);

        if (intersect) inside = !inside;
    }}

    return inside;
}}

function getSelectedRows() {{
    if (polygon.length < 3) return [];

    return rows.filter(row => {{
        const x = Number(row[xCol]);
        const y = Number(row[yCol]);
        return pointInPolygon(x, y, polygon);
    }});
}}

function updateSelectedCount() {{
    const selected = getSelectedRows();
    document.getElementById("selectedParticles").innerText = selected.length;
}}

function drawParticles() {{
    const selected = new Set();

    if (polygon.length >= 3) {{
        for (let i = 0; i < rows.length; i++) {{
            const x = Number(rows[i][xCol]);
            const y = Number(rows[i][yCol]);

            if (pointInPolygon(x, y, polygon)) {{
                selected.add(i);
            }}
        }}
    }}

    for (let i = 0; i < rows.length; i++) {{
        const x = Number(rows[i][xCol]);
        const y = Number(rows[i][yCol]);

        const p = imageToScreen(x, y);

        ctx.beginPath();
        ctx.arc(p.x, p.y, pointRadius, 0, 2 * Math.PI);

        if (selected.has(i)) {{
            ctx.fillStyle = "cyan";
        }} else {{
            ctx.fillStyle = "red";
        }}

        ctx.fill();
    }}
}}

function drawPolygon() {{
    if (polygon.length === 0) return;

    ctx.lineWidth = 2;
    ctx.strokeStyle = "lime";
    ctx.fillStyle = "rgba(0, 255, 0, 0.15)";

    ctx.beginPath();

    const p0 = imageToScreen(polygon[0].x, polygon[0].y);
    ctx.moveTo(p0.x, p0.y);

    for (let i = 1; i < polygon.length; i++) {{
        const p = imageToScreen(polygon[i].x, polygon[i].y);
        ctx.lineTo(p.x, p.y);
    }}

    if (polygonClosed || polygon.length > 2) {{
        ctx.closePath();
        ctx.fill();
    }}

    ctx.stroke();

    for (const vertex of polygon) {{
        const p = imageToScreen(vertex.x, vertex.y);

        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = "yellow";
        ctx.fill();
        ctx.strokeStyle = "black";
        ctx.stroke();
    }}
}}

function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.drawImage(
        image,
        offsetX,
        offsetY,
        imgW * scale,
        imgH * scale
    );

    drawParticles();
    drawPolygon();
    updateSelectedCount();
}}

function closePolygon() {{
    if (polygon.length >= 3) {{
        polygonClosed = true;
    }}
    draw();
}}

function clearROI() {{
    polygon = [];
    polygonClosed = false;
    draw();
}}

function csvEscape(value) {{
    if (value === null || value === undefined) return "";

    let s = String(value);

    if (s.includes('"') || s.includes(",") || s.includes("\\n") || s.includes("\\r")) {{
        s = '"' + s.replaceAll('"', '""') + '"';
    }}

    return s;
}}

function downloadText(filename, text) {{
    const blob = new Blob([text], {{ type: "text/csv;charset=utf-8;" }});
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();

    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}}

function downloadSelectedCSV() {{
    const selected = getSelectedRows();

    let lines = [];
    lines.push(columns.map(csvEscape).join(","));

    for (const row of selected) {{
        const line = columns.map(col => csvEscape(row[col])).join(",");
        lines.push(line);
    }}

    downloadText(outputCsvName, lines.join("\\n"));
}}

function downloadROICSV() {{
    let lines = [];
    lines.push("vertex_index,x_px,y_px,x_um,y_um");

    for (let i = 0; i < polygon.length; i++) {{
        const xPx = polygon[i].x;
        const yPx = polygon[i].y;

        const xUm = xPx * umPerPx;
        const yUm = yPx * umPerPx;

        lines.push([
            i,
            xPx,
            yPx,
            xUm,
            yUm
        ].join(","));
    }}

    downloadText(outputRoiCsvName, lines.join("\\n"));
}}

canvas.addEventListener("wheel", function(event) {{
    event.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;

    const before = screenToImage(sx, sy);

    const factor = event.deltaY < 0 ? 1.15 : 0.87;
    scale = Math.min(Math.max(scale * factor, 0.02), 100);

    offsetX = sx - before.x * scale;
    offsetY = sy - before.y * scale;

    draw();
}}, {{ passive: false }});

canvas.addEventListener("mousedown", function(event) {{
    const rect = canvas.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;

    lastMouseX = sx;
    lastMouseY = sy;

    if (mode === "pan") {{
        isPanning = true;
        canvas.style.cursor = "grabbing";
        return;
    }}

    if (mode === "freehand") {{
        const p = screenToImage(sx, sy);
        polygon = [p];
        polygonClosed = false;
        isDrawingFreehand = true;
        return;
    }}
}});

canvas.addEventListener("mousemove", function(event) {{
    const rect = canvas.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;

    const imgP = screenToImage(sx, sy);

    document.getElementById("mouseX").innerText = imgP.x.toFixed(1);
    document.getElementById("mouseY").innerText = imgP.y.toFixed(1);

    if (isPanning) {{
        const dx = sx - lastMouseX;
        const dy = sy - lastMouseY;

        offsetX += dx;
        offsetY += dy;

        lastMouseX = sx;
        lastMouseY = sy;

        draw();
        return;
    }}

    if (isDrawingFreehand) {{
        const last = polygon[polygon.length - 1];

        const dx = imgP.x - last.x;
        const dy = imgP.y - last.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > 2.0) {{
            polygon.push(imgP);
            draw();
        }}
    }}
}});

canvas.addEventListener("mouseup", function(event) {{
    if (isPanning) {{
        isPanning = false;
        if (mode === "pan") canvas.style.cursor = "grab";
    }}

    if (isDrawingFreehand) {{
        isDrawingFreehand = false;

        if (polygon.length >= 3) {{
            polygonClosed = true;
        }}

        draw();
    }}
}});

canvas.addEventListener("click", function(event) {{
    if (mode !== "polygon") return;
    if (isPanning || isDrawingFreehand) return;

    const rect = canvas.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;

    const p = screenToImage(sx, sy);

    polygon.push(p);
    polygonClosed = false;

    draw();
}});

image.onload = function() {{
    resetView();
}};
</script>

</body>
</html>
"""

    output_html = Path(output_html)
    output_html.write_text(html, encoding="utf-8")

    print("Created:", output_html.resolve())
    webbrowser.open(output_html.resolve().as_uri())

    return output_html.resolve()