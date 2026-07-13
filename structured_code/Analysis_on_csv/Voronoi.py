import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
from io import BytesIO
import base64
import json
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def count_polygon_edges(poly, min_edge_length=0):

    poly = np.asarray(poly, dtype=float)

    if len(poly) < 3:
        return 0

    n = len(poly)
    edge_count = 0

    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]   # wraps last point back to first point

        edge_length = np.linalg.norm(p2 - p1)

        if edge_length >= min_edge_length:
            edge_count += 1

    return edge_count


def voronoi_finite_polygons_2d(vor, radius=None):


    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D Voronoi diagrams are supported.")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)

    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    # Map ridge vertices to each point
    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct each Voronoi region
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if all(v >= 0 for v in vertices):
            # Already finite
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1

            if v1 >= 0:
                continue

            # Tangent direction between points
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)

            # Normal direction
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)

            direction = np.sign(np.dot(midpoint - center, normal)) * normal

            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # Sort vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        centroid = vs.mean(axis=0)

        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def compute_voronoi_from_df(
    df,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.32,
    min_edge_length_um = 2
):


    df = df.copy()

    valid = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    df = df[valid].reset_index(drop=True)

    points = df[[x_col, y_col]].to_numpy(dtype=float)

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    areas_px2 = []
    areas_um2 = []
    edge_counts = []
    polygons = []
    min_edge_length_px = min_edge_length_um / um_per_px

    for region in regions:
        poly = vertices[region]
        polygons.append(poly)

        edge_counts.append(count_polygon_edges(poly, min_edge_length=min_edge_length_px))

        x = poly[:, 0]
        y = poly[:, 1]

        area_px2 = 0.5 * np.abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )

        areas_px2.append(area_px2)
        areas_um2.append(area_px2 * um_per_px**2)

    df["voronoi_area_px2"] = areas_px2
    df["voronoi_area_um2"] = areas_um2
    df["voronoi_num_edges"] = edge_counts
    df["voronoi_poly"] = polygons

    return df, vor, regions, vertices



def make_voronoi_neighbor_graph(
    vor_df,
    vor,
    x_col="x_col",
    y_col="y_col",
    particle_id_col=None,
    min_edge_length_um=0.5,
    um_per_px=0.65,
):
 

    graph_df = vor_df.copy().reset_index(drop=True)

    if particle_id_col is None:
        node_ids = graph_df.index.to_numpy()
    else:
        node_ids = graph_df[particle_id_col].to_numpy()

    G = nx.Graph()

    # Add nodes with useful attributes
    for i, node_id in enumerate(node_ids):
        G.add_node(
            node_id,
            vor_index=i,
            x=float(graph_df.loc[i, x_col]),
            y=float(graph_df.loc[i, y_col]),
        )


    min_edge_length_px = min_edge_length_um / um_per_px

    for (i, j), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):

        # Skip infinite ridges
        if -1 in ridge_vertices:
            continue

        ridge_pts = vor.vertices[ridge_vertices]

        if len(ridge_pts) < 2:
            continue

        # Length of the shared Voronoi edge
        ridge_length_px = np.linalg.norm(ridge_pts[1] - ridge_pts[0])

        if ridge_length_px < min_edge_length_px:
            continue

        node_i = node_ids[i]
        node_j = node_ids[j]
        G.add_edge(node_i, node_j)



    # Add graph coordination number back into dataframe
    coord_nums = []

    for i, node_id in enumerate(node_ids):
        if node_id in G:
            coord_nums.append(G.degree[node_id])
        else:
            coord_nums.append(np.nan)

    graph_df["voronoi_graph_coord_num"] = coord_nums

    return G, graph_df


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



def make_interactive_voronoi_html(
    image,
    vor_df,
    output_html="interactive_voronoi.html",
    x_col="x_col",
    y_col="y_col",
    poly_col="voronoi_poly",
    value_col="voronoi_num_edges",
    cmap_name="tab20",
    alpha=0.35,
    robust_percentiles=(2, 98),
    point_radius=2.5,
    um_per_px=0.65,
    scale_bar_um=50,
    scale_bar_corner="bottom-left",
):
    """
    Make a self-contained HTML file where you can zoom and pan through
    Voronoi cells on top of the image.

    Controls:
    - Mouse wheel: zoom
    - Left mouse drag: pan
    - Reset view button: go back to full image
    - Checkboxes: show/hide image, Voronoi cells, centers
    """

    image = np.asarray(image)
    H, W = image.shape[:2]

    image_data_url = _image_to_base64_png(image)

    centers = vor_df[[x_col, y_col]].to_numpy(dtype=float)

    values = vor_df[value_col].to_numpy(dtype=int)

    unique_edges = np.sort(np.unique(values))

    cmap = cm.get_cmap(cmap_name, len(unique_edges))

    edge_to_color = {}

    for i, edge_count in enumerate(unique_edges):
        r, g, b, _ = cmap(i)
        edge_to_color[int(edge_count)] = {
            "rgba": f"rgba({int(255*r)}, {int(255*g)}, {int(255*b)}, {alpha})",
            "rgb": f"rgb({int(255*r)}, {int(255*g)}, {int(255*b)})",
        }

    legend_items = [
        {
            "edge": int(edge_count),
            "color": edge_to_color[int(edge_count)]["rgb"],
        }
        for edge_count in unique_edges
    ]

    polygons = []

    for poly, value in zip(vor_df[poly_col], values):
        poly = np.asarray(poly, dtype=float)

        if poly.ndim != 2 or poly.shape[1] != 2:
            continue

        if not np.all(np.isfinite(poly)):
            continue

        edge_count = int(value)
        color = edge_to_color[edge_count]["rgba"]

        xmin = float(np.min(poly[:, 0]))
        xmax = float(np.max(poly[:, 0]))
        ymin = float(np.min(poly[:, 1]))
        ymax = float(np.max(poly[:, 1]))

        polygons.append({
            "points": np.round(poly, 2).tolist(),
            "color": color,
            "bbox": [xmin, ymin, xmax, ymax],
            "value": float(value),
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
<title>Interactive Voronoi Viewer</title>

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
        image-rendering: pixelated;
        image-rendering: crisp-edges;
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

    <span>Color: {value_col}</span>

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
const umPerPx = {um_per_px};
const scaleBarUm = {scale_bar_um};
const scaleBarCorner = "{scale_bar_corner}";
const edgeAlpha = 0.9;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;
ctx.imageSmoothingQuality = "low";

const showImageBox = document.getElementById("showImage");
const showCellsBox = document.getElementById("showCells");
const showCentersBox = document.getElementById("showCenters");
const info = document.getElementById("info");
const legend = document.getElementById("legend");

function buildLegend() {{
    let html = "<div class='legend-title'>Number of Voronoi cell edges</div>";

    for (const item of legendItems) {{
        html += "<div class='legend-row'>";
        html += "<div class='legend-color' style='background:" + item.color + ";'></div>";
        html += "<span>" + item.edge + "</span>";
        html += "</div>";
    }}

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

    ctx.strokeStyle = "rgba(255, 255, 0, " + edgeAlpha + ")";
    ctx.lineWidth = 0.75 / scale;
    ctx.stroke();
}}

function drawScaleBar() {{
    if (scaleBarUm <= 0 || umPerPx <= 0) {{
        return;
    }}

    // length of the scale bar in image pixels
    const barImagePx = scaleBarUm / umPerPx;

    // convert image pixels to screen pixels using current zoom
    const barScreenPx = barImagePx * scale;

    const margin = 30;
    const barHeight = 5;

    let x0;
    let y0;

    if (scaleBarCorner.includes("right")) {{
        x0 = canvas.width - margin - barScreenPx;
    }} else {{
        x0 = margin;
    }}

    if (scaleBarCorner.includes("top")) {{
        y0 = margin + 25;
    }} else {{
        y0 = canvas.height - margin;
    }}

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // black outline for visibility
    ctx.strokeStyle = "black";
    ctx.lineWidth = barHeight + 3;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + barScreenPx, y0);
    ctx.stroke();

    // white scale bar
    ctx.strokeStyle = "white";
    ctx.lineWidth = barHeight;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + barScreenPx, y0);
    ctx.stroke();

    // label
    ctx.font = "14px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    const label = scaleBarUm + " µm";
    const labelX = x0 + barScreenPx / 2;
    const labelY = y0 - 8;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "black";
    ctx.strokeText(label, labelX, labelY);

    ctx.fillStyle = "white";
    ctx.fillText(label, labelX, labelY);

    ctx.restore();
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
        ctx.imageSmoothingEnabled = false;
        ctx.imageSmoothingQuality = "low";
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
drawScaleBar();

    info.innerHTML =

        "zoom: " + scale.toFixed(3) +
        " | visible x: " + viewLeft.toFixed(1) + " - " + viewRight.toFixed(1) +
        " | y: " + viewTop.toFixed(1) + " - " + viewBottom.toFixed(1);
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
    if (!isDragging) {{
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        const world = screenToWorld(mouseX, mouseY);

        info.innerHTML =
            "x: " + world.x.toFixed(1) +
            " | y: " + world.y.toFixed(1) +
            " | zoom: " + scale.toFixed(3);

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
    output_html.write_text(html, encoding="utf-8")

    return output_html