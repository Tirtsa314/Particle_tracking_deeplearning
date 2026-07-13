

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


def make_interactive_voronoi_order_alpha_html(
    image,
    vor_df,
    output_html="interactive_voronoi_order_alpha.html",
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    psi_col="psi_3",
    color_value_col="voronoi_num_edges",
    cmap_name="tab20",
    min_alpha=0.08,
    max_alpha=0.75,
    point_radius=2.5,
    same_color_from_edge=6,
):
    """
    Make an interactive Voronoi HTML viewer where the transparency of each
    Voronoi cell is scaled by the order parameter.

    Higher psi => less transparent / more opaque.
    Lower psi  => more transparent.

    Parameters
    ----------
    image : ndarray
        Background microscopy image.

    vor_df : pandas.DataFrame
        DataFrame containing Voronoi polygons and psi values.
        Needs columns:
            x_col, y_col, poly_col, psi_col, color_value_col

    output_html : str or Path
        Path where the HTML file is saved.

    psi_col : str
        Column containing order parameter values, e.g. 'psi_3' or 'psi_6'.

    color_value_col : str
        Column used for color categories.
        Usually 'voronoi_num_edges'.

    min_alpha, max_alpha : float
        Alpha range. psi=0 gives min_alpha, psi=1 gives max_alpha.
    """

    import numpy as np
    import json
    from pathlib import Path
    import matplotlib.cm as cm

    image = np.asarray(image)
    H, W = image.shape[:2]

    if psi_col not in vor_df.columns:
        raise ValueError(
            f"Column '{psi_col}' not found in vor_df.\n"
            f"Available columns are:\n{list(vor_df.columns)}"
        )

    if poly_col not in vor_df.columns:
        raise ValueError(f"Column '{poly_col}' not found in vor_df.")

    if color_value_col not in vor_df.columns:
        raise ValueError(f"Column '{color_value_col}' not found in vor_df.")

    image_data_url = _image_to_base64_png(image)

    centers = vor_df[[x_col, y_col]].to_numpy(dtype=float)

    color_values = vor_df[color_value_col].to_numpy()
    psi_values = vor_df[psi_col].to_numpy(dtype=float)

    valid_color_values = color_values[np.isfinite(color_values)]
    unique_values = np.sort(np.unique(valid_color_values.astype(int)))

    cmap = cm.get_cmap(cmap_name, len(unique_values))

    value_to_rgb = {}

    for i, value in enumerate(unique_values):
        r, g, b, _ = cmap(i)
        value_to_rgb[int(value)] = {
            "r": int(255 * r),
            "g": int(255 * g),
            "b": int(255 * b),
            "rgb": f"rgb({int(255*r)}, {int(255*g)}, {int(255*b)})",
        }

    legend_items = [
        {
            "value": int(value),
            "color": value_to_rgb[int(value)]["rgb"],
        }
        for value in unique_values
    ]

    polygons = []

    for poly, color_value, psi in zip(vor_df[poly_col], color_values, psi_values):
        poly = np.asarray(poly, dtype=float)

        if poly.ndim != 2 or poly.shape[1] != 2:
            continue

        if not np.all(np.isfinite(poly)):
            continue

        if not np.isfinite(color_value):
            continue

        if not np.isfinite(psi):
            psi = 0.0

        # Keep psi inside [0, 1]
        psi_clipped = float(np.clip(psi, 0.0, 1.0))

        # Higher psi => higher alpha => less transparent
        alpha = min_alpha + psi_clipped * (max_alpha - min_alpha)

        color_value_int = int(color_value)
        rgb = value_to_rgb[color_value_int]

        color = (
            f"rgba({rgb['r']}, {rgb['g']}, {rgb['b']}, {alpha:.3f})"
        )

        xmin = float(np.min(poly[:, 0]))
        xmax = float(np.max(poly[:, 0]))
        ymin = float(np.min(poly[:, 1]))
        ymax = float(np.max(poly[:, 1]))

        polygons.append({
            "points": np.round(poly, 2).tolist(),
            "color": color,
            "bbox": [xmin, ymin, xmax, ymax],
            "value": float(color_value),
            "psi": float(psi_clipped),
            "alpha": float(alpha),
        })

    centers_json = json.dumps(np.round(centers, 2).tolist())
    polygons_json = json.dumps(polygons)
    legend_json = json.dumps(legend_items)
    image_json = json.dumps(image_data_url)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive Voronoi Order Viewer</title>

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
        margin-bottom: 8px;
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

    <span>Color: {color_value_col}</span>
    <span>Transparency: {psi_col}</span>

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
const legendItems = {legend_json};

const pointRadius = {point_radius};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const showImageBox = document.getElementById("showImage");
const showCellsBox = document.getElementById("showCells");
const showCentersBox = document.getElementById("showCenters");
const info = document.getElementById("info");
const legend = document.getElementById("legend");

function buildLegend() {{
    let html = "<div class='legend-title'>Voronoi cell color</div>";

    for (const item of legendItems) {{
        html += "<div class='legend-row'>";
        html += "<div class='legend-color' style='background:" + item.color + ";'></div>";
        html += "<span>" + item.value + "</span>";
        html += "</div>";
    }}

    html += "<div class='small'>";
    html += "Transparency is scaled by |ψ|:<br>";
    html += "low |ψ| = more transparent<br>";
    html += "high |ψ| = more opaque";
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
        ctx.fillStyle = "red";

        const r = pointRadius / scale;

        for (const p of centers) {{
            const x = p[0];
            const y = p[1];

            if (
                x < viewLeft ||
                x > viewRight ||
                y < viewTop ||
                y > viewBottom
            ) {{
                continue;
            }}

            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fill();
        }}
    }}

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    info.innerHTML =
        "zoom: " + scale.toFixed(3) +
        " | visible x: " + viewLeft.toFixed(1) + " - " + viewRight.toFixed(1) +
        " | y: " + viewTop.toFixed(1) + " - " + viewBottom.toFixed(1);
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
        const poly = findPolygonAt(world.x, world.y);

        if (poly === null) {{
            info.innerHTML =
                "x: " + world.x.toFixed(1) +
                " | y: " + world.y.toFixed(1) +
                " | zoom: " + scale.toFixed(3);
        }} else {{
            info.innerHTML =
                "x: " + world.x.toFixed(1) +
                " | y: " + world.y.toFixed(1) +
                " | {color_value_col}: " + poly.value +
                " | {psi_col}: " + poly.psi.toFixed(3) +
                " | alpha: " + poly.alpha.toFixed(3);
        }}

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