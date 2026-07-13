



from pathlib import Path
from io import BytesIO
import base64
import json
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import base64


def _image_to_base64_png(image):
    """
    Convert a numpy image array to a base64 PNG for embedding in HTML.
    """

    arr = np.asarray(image)

    if arr.dtype != np.uint8:
        arr = arr.astype(float)
        arr = arr - np.nanmin(arr)
        arr = arr / np.nanmax(arr)
        arr = (255 * arr).astype(np.uint8)

    if arr.ndim == 2:
        img = Image.fromarray(arr)
    elif arr.ndim == 3:
        img = Image.fromarray(arr)
    else:
        raise ValueError("Image must be 2D grayscale or 3D RGB/RGBA.")

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded


def make_interactive_voronoi_order_centers_html(
    image,
    vor_df,
    ordpar_df,
    output_html="interactive_voronoi_order_centers.html",
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    psi_col="psi_3",
    cell_value_col="voronoi_num_edges",
    cell_cmap_name="tab20",
    center_cmap_name="OrRd",
    cell_alpha=0.30,
    point_radius=3.5,
    n_center_bins=10,
    same_color_from_edge=6,
):
    """
    Make an interactive Voronoi HTML viewer.

    Voronoi cells:
        colored by cell_value_col, for example voronoi_num_edges.

    Particle centers:
        colored by order parameter psi_col, for example psi_3 or psi_6.

    The cell palette and center palette are intentionally different
    to avoid confusion.
    """
    cell_color_indices = {
        1: 0,
        2: 0,
        3: 4,
        4: 8,
        5: 10,
        6: 19,
    }
    

    image = np.asarray(image)
    H, W = image.shape[:2]

    if psi_col not in vor_df.columns:
        raise ValueError(
            f"Column '{psi_col}' not found in vor_df.\n"
            f"Available columns are:\n{list(vor_df.columns)}"
        )

    if poly_col not in vor_df.columns:
        raise ValueError(f"Column '{poly_col}' not found in vor_df.")

    if cell_value_col not in vor_df.columns:
        raise ValueError(f"Column '{cell_value_col}' not found in vor_df.")

    image_data_url = _image_to_base64_png(image)



    centers_xy = vor_df[[x_col, y_col]].to_numpy(dtype=float)
    psi_values = vor_df[psi_col].to_numpy(dtype=float)
    cell_values = vor_df[cell_value_col].to_numpy()

    if ordpar_df is not None:
        psi_values = ordpar_df[psi_col].to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Cell colors: discrete palette based on Voronoi edge number
    # but all edge counts >= same_color_from_edge share one color
    # ------------------------------------------------------------
    valid_cell_values = cell_values[np.isfinite(cell_values)].astype(int)

    # Map 6,7,8,... all to 6 if same_color_from_edge=6
    grouped_cell_values = np.array([
        v if v < same_color_from_edge else same_color_from_edge
        for v in valid_cell_values
    ], dtype=int)

    unique_cell_groups = np.sort(np.unique(grouped_cell_values))

    cell_cmap = cm.get_cmap(cell_cmap_name, 20)

    cell_value_to_rgb = {}

    for value in unique_cell_groups:
        value = int(value)

        color_index = cell_color_indices[value]

        r, g, b, _ = cell_cmap(color_index)

        cell_value_to_rgb[value] = {
            "r": int(255 * r),
            "g": int(255 * g),
            "b": int(255 * b),
            "rgb": f"rgb({int(255*r)}, {int(255*g)}, {int(255*b)})",
            "rgba": f"rgba({int(255*r)}, {int(255*g)}, {int(255*b)}, {cell_alpha})",
        }

    cell_legend_items = [
        {
            "value": f"{int(value)}+" if int(value) == same_color_from_edge else int(value),
            "color": cell_value_to_rgb[int(value)]["rgb"],
        }
        for value in unique_cell_groups
    ]

    # ------------------------------------------------------------
    # Center colors: continuous/discretized palette based on psi
    # ------------------------------------------------------------
    center_cmap = cm.get_cmap(center_cmap_name, n_center_bins)

    def psi_to_center_color(psi):
        if not np.isfinite(psi):
            psi = 0.0

        psi = float(np.clip(psi, 0.0, 1.0))

        bin_index = int(np.floor(psi * n_center_bins))

        if bin_index >= n_center_bins:
            bin_index = n_center_bins - 1

        r, g, b, _ = center_cmap(bin_index)

        return {
            "psi": psi,
            "bin": bin_index,
            "color": f"rgb({int(255*r)}, {int(255*g)}, {int(255*b)})",
        }

    center_legend_items = []

    for i in range(n_center_bins):
        low = i / n_center_bins
        high = (i + 1) / n_center_bins

        r, g, b, _ = center_cmap(i)

        center_legend_items.append({
            "label": f"{low:.1f} - {high:.1f}",
            "color": f"rgb({int(255*r)}, {int(255*g)}, {int(255*b)})",
        })

    # ------------------------------------------------------------
    # Build polygon data
    # ------------------------------------------------------------
    polygons = []

    for poly, cell_value, psi in zip(vor_df[poly_col], cell_values, psi_values):
        poly = np.asarray(poly, dtype=float)

        if poly.ndim != 2 or poly.shape[1] != 2:
            continue

        if not np.all(np.isfinite(poly)):
            continue

        if not np.isfinite(cell_value):
            continue

        raw_cell_value_int = int(cell_value)

        color_group = (
            raw_cell_value_int
            if raw_cell_value_int < same_color_from_edge
            else same_color_from_edge
        )

        color = cell_value_to_rgb[color_group]["rgba"]

        xmin = float(np.min(poly[:, 0]))
        xmax = float(np.max(poly[:, 0]))
        ymin = float(np.min(poly[:, 1]))
        ymax = float(np.max(poly[:, 1]))

        psi_clean = float(np.clip(psi, 0.0, 1.0)) if np.isfinite(psi) else 0.0

        polygons.append({
            "points": np.round(poly, 2).tolist(),
            "color": color,
            "bbox": [xmin, ymin, xmax, ymax],
            "cell_value": float(cell_value),   # keeps the real number for hover text
            "psi": psi_clean,
        })
    # ------------------------------------------------------------
    # Build center data
    # ------------------------------------------------------------
    centers = []

    for xy, psi in zip(centers_xy, psi_values):
        x, y = xy

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        center_color_info = psi_to_center_color(psi)

        centers.append({
            "x": float(x),
            "y": float(y),
            "psi": center_color_info["psi"],
            "color": center_color_info["color"],
        })

    centers_json = json.dumps(centers)
    polygons_json = json.dumps(polygons)
    cell_legend_json = json.dumps(cell_legend_items)
    center_legend_json = json.dumps(center_legend_items)
    image_json = json.dumps(image_data_url)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive Voronoi Order Center Viewer</title>

<style>
    body {{
        margin: 0;
        font-family: Arial, sans-serif;
        background: #111;
        color: white;
        overflow: hidden;
    }}

    #toolbar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 44px;
        background: #222;
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 0 12px;
        z-index: 10;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }}

    button {{
        padding: 5px 10px;
        cursor: pointer;
    }}

    label {{
        user-select: none;
    }}

    #info {{
        margin-left: auto;
        font-size: 13px;
        color: #ddd;
    }}

    #canvas {{
        position: absolute;
        left: 0;
        top: 44px;
        width: 100vw;
        height: calc(100vh - 44px);
        background: #111;
        cursor: grab;
    }}

    #canvas:active {{
        cursor: grabbing;
    }}

    #legend {{
        position: fixed;
        right: 18px;
        top: 70px;
        max-height: calc(100vh - 110px);
        overflow-y: auto;
        background: rgba(20, 20, 20, 0.88);
        color: white;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 13px;
        z-index: 20;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}

    .legend-title {{
        font-weight: bold;
        margin-top: 8px;
        margin-bottom: 8px;
    }}

    .legend-title:first-child {{
        margin-top: 0;
    }}

    .legend-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
    }}

    .legend-color {{
        width: 22px;
        height: 14px;
        border: 1px solid white;
    }}

    .small {{
        font-size: 12px;
        color: #bbb;
        margin-top: 8px;
        line-height: 1.3;
    }}
</style>
</head>

<body>

<div id="toolbar">
    <button onclick="resetView()">Reset view</button>

    <label>
        <input type="checkbox" id="showImage" checked>
        image
    </label>

    <label>
        <input type="checkbox" id="showCells" checked>
        Voronoi cells
    </label>

    <label>
        <input type="checkbox" id="showCenters" checked>
        centers
    </label>

    <span>Cells: {cell_value_col}</span>
    <span>Centers: {psi_col}</span>

    <div id="info"></div>
</div>

<canvas id="canvas"></canvas>
<div id="legend"></div>

<script>
const imageWidth = {W};
const imageHeight = {H};

const imageData = {image_json};
const polygons = {polygons_json};
const centers = {centers_json};

const cellLegendItems = {cell_legend_json};
const centerLegendItems = {center_legend_json};

const pointRadius = {point_radius};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const showImageBox = document.getElementById("showImage");
const showCellsBox = document.getElementById("showCells");
const showCentersBox = document.getElementById("showCenters");
const info = document.getElementById("info");
const legend = document.getElementById("legend");

function buildLegend() {{
    let html = "";

    html += "<div class='legend-title'>Cell color: {cell_value_col}</div>";

    for (const item of cellLegendItems) {{
        html += "<div class='legend-row'>";
        html += "<div class='legend-color' style='background:" + item.color + ";'></div>";
        html += "<span>" + item.value + "</span>";
        html += "</div>";
    }}

    html += "<div class='legend-title'>Center color: {psi_col}</div>";

    for (const item of centerLegendItems) {{
        html += "<div class='legend-row'>";
        html += "<div class='legend-color' style='background:" + item.color + ";'></div>";
        html += "<span>" + item.label + "</span>";
        html += "</div>";
    }}

    html += "<div class='small'>";
    html += "Cells and centers use different palettes.<br>";
    html += "Cell color = Voronoi property.<br>";
    html += "Center color = order parameter.";
    html += "</div>";

    legend.innerHTML = html;
}}

buildLegend();

let scale = 1.0;
let offsetX = 0.0;
let offsetY = 0.0;

let isDragging = false;
let lastX = 0;
let lastY = 0;

const img = new Image();
img.src = imageData;
img.onload = function() {{
    resizeCanvas();
    resetView();
}};

function resizeCanvas() {{
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    draw();
}}

window.addEventListener("resize", resizeCanvas);

function resetView() {{
    const sx = canvas.width / imageWidth;
    const sy = canvas.height / imageHeight;

    scale = Math.min(sx, sy);

    offsetX = 0.5 * (canvas.width - imageWidth * scale);
    offsetY = 0.5 * (canvas.height - imageHeight * scale);

    draw();
}}

function screenToWorld(screenX, screenY) {{
    return {{
        x: (screenX - offsetX) / scale,
        y: (screenY - offsetY) / scale
    }};
}}

function drawPolygon(poly) {{
    const pts = poly.points;

    if (pts.length < 3) {{
        return;
    }}

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);

    for (let i = 1; i < pts.length; i++) {{
        ctx.lineTo(pts[i][0], pts[i][1]);
    }}

    ctx.closePath();

    ctx.fillStyle = poly.color;
    ctx.fill();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
    ctx.lineWidth = 0.75 / scale;
    ctx.stroke();
}}

function drawCenter(center) {{
    const r = pointRadius / scale;

    ctx.beginPath();
    ctx.arc(center.x, center.y, r, 0, 2 * Math.PI);

    ctx.fillStyle = center.color;
    ctx.fill();

    ctx.strokeStyle = "black";
    ctx.lineWidth = 0.75 / scale;
    ctx.stroke();
}}

function draw() {{
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);

    const viewLeft = -offsetX / scale;
    const viewTop = -offsetY / scale;
    const viewRight = viewLeft + canvas.width / scale;
    const viewBottom = viewTop + canvas.height / scale;

    if (showImageBox.checked) {{
        ctx.drawImage(img, 0, 0, imageWidth, imageHeight);
    }}

    if (showCellsBox.checked) {{
        for (const poly of polygons) {{
            const b = poly.bbox;

            if (
                b[2] < viewLeft ||
                b[0] > viewRight ||
                b[3] < viewTop ||
                b[1] > viewBottom
            ) {{
                continue;
            }}

            drawPolygon(poly);
        }}
    }}

    if (showCentersBox.checked) {{
        for (const center of centers) {{
            if (
                center.x < viewLeft ||
                center.x > viewRight ||
                center.y < viewTop ||
                center.y > viewBottom
            ) {{
                continue;
            }}

            drawCenter(center);
        }}
    }}

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    info.innerHTML =
        "zoom: " + scale.toFixed(3) +
        " | visible x: " + viewLeft.toFixed(1) + " - " + viewRight.toFixed(1) +
        " | y: " + viewTop.toFixed(1) + " - " + viewBottom.toFixed(1);
}}

function findNearestCenter(worldX, worldY) {{
    let best = null;
    let bestDist = Infinity;

    for (const center of centers) {{
        const dx = center.x - worldX;
        const dy = center.y - worldY;
        const d = Math.sqrt(dx * dx + dy * dy);

        if (d < bestDist) {{
            bestDist = d;
            best = center;
        }}
    }}

    const maxDist = Math.max(6 / scale, pointRadius * 4 / scale);

    if (bestDist <= maxDist) {{
        return best;
    }}

    return null;
}}

function findPolygonAt(worldX, worldY) {{
    for (let i = polygons.length - 1; i >= 0; i--) {{
        const poly = polygons[i];
        const pts = poly.points;
        let inside = false;

        for (let j = 0, k = pts.length - 1; j < pts.length; k = j++) {{
            const xi = pts[j][0], yi = pts[j][1];
            const xj = pts[k][0], yj = pts[k][1];

            const intersect =
                ((yi > worldY) !== (yj > worldY)) &&
                (worldX < (xj - xi) * (worldY - yi) / (yj - yi + 1e-12) + xi);

            if (intersect) inside = !inside;
        }}

        if (inside) return poly;
    }}

    return null;
}}

canvas.addEventListener("mousedown", function(event) {{
    isDragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
}});

window.addEventListener("mouseup", function() {{
    isDragging = false;
}});

window.addEventListener("mousemove", function(event) {{
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    if (!isDragging) {{
        const world = screenToWorld(mouseX, mouseY);

        const center = findNearestCenter(world.x, world.y);
        const poly = findPolygonAt(world.x, world.y);

        let text =
            "x: " + world.x.toFixed(1) +
            " | y: " + world.y.toFixed(1) +
            " | zoom: " + scale.toFixed(3);

        if (poly !== null) {{
            text += " | {cell_value_col}: " + poly.cell_value;
        }}

        if (center !== null) {{
            text += " | {psi_col}: " + center.psi.toFixed(3);
        }}

        info.innerHTML = text;
        return;
    }}

    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;

    offsetX += dx;
    offsetY += dy;

    lastX = event.clientX;
    lastY = event.clientY;

    draw();
}});

canvas.addEventListener("wheel", function(event) {{
    event.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const worldBefore = screenToWorld(mouseX, mouseY);

    const zoomFactor = event.deltaY < 0 ? 1.20 : 1 / 1.20;

    scale *= zoomFactor;

    offsetX = mouseX - worldBefore.x * scale;
    offsetY = mouseY - worldBefore.y * scale;

    draw();
}}, {{ passive: false }});

showImageBox.addEventListener("change", draw);
showCellsBox.addEventListener("change", draw);
showCentersBox.addEventListener("change", draw);
</script>

</body>
</html>
"""

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    return output_html