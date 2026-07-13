from pathlib import Path
import base64
import json
import pandas as pd
import numpy as np
from PIL import Image
import webbrowser


def make_refined_review_html(
    image,
    df_det_fitted,
    output_html="review_refined_triangles.html",
    output_csv_name="manual_triangle_edits.csv",
    success_only=True,
):
    """
    Create an interactive HTML page to inspect refined triangle fits.

    Features
    --------
    - Shows image
    - Overlays YOLO polygon outlines (poly_json)
    - Overlays refined centers (x_refined, y_refined)
    - Keyboard modes:
        t = add missing center
        r = mark existing refined center for removal
        p / Esc = pan mode
    - Mouse wheel = zoom
    - Drag = pan
    - Click behavior depends on mode:
        * in 't' mode: click adds missing center
        * in 'r' mode: click near an existing refined center toggles removal
    - Downloads CSV with separate columns for:
        missing centers and removed centers
    """

    # ------------------------------------------------------------
    # 1) Load image
    # ------------------------------------------------------------
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("L")
    else:
        arr = np.asarray(image)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            arr = arr - np.nanmin(arr)
            mx = np.nanmax(arr)
            if mx > 0:
                arr = arr / mx
            arr = (255 * arr).astype(np.uint8)

        img = Image.fromarray(arr).convert("L")

    W, H = img.size

    # encode image as base64
    temp_png = Path("_temp_review_image.png")
    img.save(temp_png)

    with open(temp_png, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    temp_png.unlink()

    # ------------------------------------------------------------
    # 2) Load refined CSV
    # ------------------------------------------------------------
    df = df_det_fitted

    if success_only and "success" in df.columns:
        df = df[df["success"] == True].copy()

    df = df[df["x_refined"].notna() & df["y_refined"].notna()].copy()

    detections = []
    for _, row in df.iterrows():
        poly = []
        if "poly_json" in row and pd.notna(row["poly_json"]):
            try:
                raw_poly = json.loads(row["poly_json"])
                # round a bit so the HTML stays smaller
                poly = [[round(float(x), 2), round(float(y), 2)] for x, y in raw_poly]
            except Exception:
                poly = []

        detections.append(
            {
                "particle_id": int(row["particle_id"]),
                "x": round(float(row["x_refined"]), 3),
                "y": round(float(row["y_refined"]), 3),
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
<title>Review refined triangles</title>

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

    .modebox {{
        display: inline-block;
        padding: 4px 10px;
        border: 1px solid #888;
        border-radius: 6px;
        margin-left: 10px;
        background: #222;
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

<h2>Review refined triangle fits</h2>

<div class="toolbar">
    <button onclick="setMode('pan')">Pan mode (P / Esc)</button>
    <button onclick="setMode('add')">Missing center mode (T)</button>
    <button onclick="setMode('remove')">Remove center mode (R)</button>
    <button onclick="undoLast()">Undo last</button>
    <button onclick="clearMissing()">Clear missing</button>
    <button onclick="clearRemoved()">Clear removed</button>
    <button onclick="resetView()">Reset view</button>
    <button onclick="downloadCSV()">Download CSV</button>

    <span class="modebox">
        Current mode: <span id="modeLabel">pan</span>
    </span>
</div>

<div class="toolbar">
    <label><input type="checkbox" id="showPolys" checked onchange="draw()"> Show YOLO outlines</label>
    <label><input type="checkbox" id="showCenters" checked onchange="draw()"> Show refined centers</label>
    <label><input type="checkbox" id="showIds" onchange="draw()"> Show particle IDs</label>
</div>

<p class="hint">
Keyboard:<br>
T = add missing center<br>
R = mark existing refined center for removal<br>
P or Esc = pan mode<br>
U = undo last action
</p>

<canvas id="canvas"></canvas>

<div id="info">
    Image size: {W} × {H} px<br>
    Existing refined detections: <span id="nDetections">0</span><br>
    Missing centers added: <span id="nMissing">0</span><br>
    Existing centers marked for removal: <span id="nRemoved">0</span><br>
    Zoom: <span id="zoom">1.00</span>x
</div>

<script>
const imageWidth = {W};
const imageHeight = {H};
const outputCsvName = "{output_csv_name}";
const detections = {detections_json};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

document.getElementById("nDetections").innerText = detections.length;

const img = new Image();
img.src = "data:image/png;base64,{img_base64}";

// view transform
let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

// modes: pan, add, remove
let mode = "pan";

// user annotations
let missingPoints = [];
let removedParticleIds = new Set();
let history = [];

// pan state
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragMoved = false;

// canvas size
canvas.width = Math.min(window.innerWidth * 0.92, 1400);
canvas.height = Math.min(window.innerHeight * 0.75, 950);

img.onload = function() {{
    resetView();
}};

function setMode(newMode) {{
    mode = newMode;
    document.getElementById("modeLabel").innerText = mode;
}}

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

    // crosshair
    ctx.beginPath();
    ctx.moveTo(p.x - radius - 3, p.y);
    ctx.lineTo(p.x + radius + 3, p.y);
    ctx.moveTo(p.x, p.y - radius - 3);
    ctx.lineTo(p.x, p.y + radius + 3);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();
}}

function drawRemovedMarker(x, y, color="red") {{
    const p = imageToScreen(x, y);
    const s = 7;

    ctx.beginPath();
    ctx.moveTo(p.x - s, p.y - s);
    ctx.lineTo(p.x + s, p.y + s);
    ctx.moveTo(p.x - s, p.y + s);
    ctx.lineTo(p.x + s, p.y - s);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
}}

function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // image
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

    // detections
    for (const det of detections) {{
        const removed = removedParticleIds.has(det.particle_id);

        if (showPolys) {{
            drawPolygon(
                det.poly,
                removed ? "rgba(255,80,80,0.85)" : "rgba(0,255,0,0.45)",
                removed ? 2 : 1
            );
        }}

        if (showCenters) {{
            if (removed) {{
                drawRemovedMarker(det.x, det.y, "red");
            }} else {{
                drawCenter(det.x, det.y, "cyan", 3);
            }}
        }}

        if (showIds) {{
            const p = imageToScreen(det.x, det.y);
            ctx.fillStyle = removed ? "red" : "white";
            ctx.font = "10px Arial";
            ctx.fillText(det.particle_id, p.x + 6, p.y - 6);
        }}
    }}

    // missing points
    for (let i = 0; i < missingPoints.length; i++) {{
        const p = missingPoints[i];
        drawCenter(p.x, p.y, "yellow", 4);

        const sp = imageToScreen(p.x, p.y);
        ctx.fillStyle = "yellow";
        ctx.font = "12px Arial";
        ctx.fillText("M" + i, sp.x + 7, sp.y - 7);
    }}

    document.getElementById("nMissing").innerText = missingPoints.length;
    document.getElementById("nRemoved").innerText = removedParticleIds.size;
    document.getElementById("zoom").innerText = scale.toFixed(2);
}}

function findNearestDetection(screenX, screenY, maxScreenDist=12) {{
    let best = null;
    let bestDist2 = maxScreenDist * maxScreenDist;

    for (const det of detections) {{
        const p = imageToScreen(det.x, det.y);
        const dx = p.x - screenX;
        const dy = p.y - screenY;
        const d2 = dx * dx + dy * dy;

        if (d2 <= bestDist2) {{
            bestDist2 = d2;
            best = det;
        }}
    }}

    return best;
}}

// Click behavior depends on mode
canvas.addEventListener("click", function(event) {{
    if (dragMoved) {{
        dragMoved = false;
        return;
    }}

    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const pt = screenToImage(mouseX, mouseY);

    if (pt.x < 0 || pt.x > imageWidth || pt.y < 0 || pt.y > imageHeight) {{
        return;
    }}

    if (mode === "add") {{
        const newPoint = {{
            x: pt.x,
            y: pt.y
        }};
        missingPoints.push(newPoint);
        history.push({{
            type: "add_missing"
        }});
        draw();
    }}

    else if (mode === "remove") {{
        const det = findNearestDetection(mouseX, mouseY, 14);
        if (!det) return;

        if (removedParticleIds.has(det.particle_id)) {{
            removedParticleIds.delete(det.particle_id);
            history.push({{
                type: "unremove_existing",
                particle_id: det.particle_id
            }});
        }} else {{
            removedParticleIds.add(det.particle_id);
            history.push({{
                type: "remove_existing",
                particle_id: det.particle_id
            }});
        }}
        draw();
    }}
}});

// Zoom
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

// Pan start
canvas.addEventListener("mousedown", function(event) {{
    isDragging = true;
    dragMoved = false;

    const rect = canvas.getBoundingClientRect();
    dragStartX = event.clientX - rect.left;
    dragStartY = event.clientY - rect.top;
}});

// Pan move
canvas.addEventListener("mousemove", function(event) {{
    if (!isDragging) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const dx = mouseX - dragStartX;
    const dy = mouseY - dragStartY;

    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {{
        dragMoved = true;
    }}

    offsetX += dx;
    offsetY += dy;

    dragStartX = mouseX;
    dragStartY = mouseY;

    draw();
}});

// Pan end
canvas.addEventListener("mouseup", function() {{
    isDragging = false;
}});
canvas.addEventListener("mouseleave", function() {{
    isDragging = false;
}});

function undoLast() {{
    if (history.length === 0) return;

    const last = history.pop();

    if (last.type === "add_missing") {{
        missingPoints.pop();
    }}
    else if (last.type === "remove_existing") {{
        removedParticleIds.delete(last.particle_id);
    }}
    else if (last.type === "unremove_existing") {{
        removedParticleIds.add(last.particle_id);
    }}

    draw();
}}

function clearMissing() {{
    missingPoints = [];
    history = [];
    draw();
}}

function clearRemoved() {{
    removedParticleIds = new Set();
    history = [];
    draw();
}}

function downloadCSV() {{
    const removedList = detections.filter(d => removedParticleIds.has(d.particle_id));

    // "wide" csv with separate columns for missing vs removed
    let csv = "missing_x,missing_y,remove_particle_id,remove_x,remove_y\\n";

    const nRows = Math.max(missingPoints.length, removedList.length);

    for (let i = 0; i < nRows; i++) {{
        const m = i < missingPoints.length ? missingPoints[i] : null;
        const r = i < removedList.length ? removedList[i] : null;

        const mx = m ? m.x.toFixed(3) : "";
        const my = m ? m.y.toFixed(3) : "";

        const rid = r ? r.particle_id : "";
        const rx = r ? r.x.toFixed(3) : "";
        const ry = r ? r.y.toFixed(3) : "";

        csv += `${{mx}},${{my}},${{rid}},${{rx}},${{ry}}\\n`;
    }}

    const blob = new Blob([csv], {{ type: "text/csv" }});
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = outputCsvName;
    a.click();

    URL.revokeObjectURL(url);
}}

// keyboard shortcuts
document.addEventListener("keydown", function(event) {{
    const key = event.key.toLowerCase();

    if (key === "t") {{
        setMode("add");
    }}
    else if (key === "r") {{
        setMode("remove");
    }}
    else if (key === "p" || key === "escape") {{
        setMode("pan");
    }}
    else if (key === "u") {{
        undoLast();
    }}
}});
</script>

</body>
</html>
"""

    output_html = Path(output_html)
    output_html.write_text(html, encoding="utf-8")

    print(f"Created: {output_html.resolve()}")
    return output_html.resolve()


