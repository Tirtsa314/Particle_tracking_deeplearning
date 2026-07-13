# %%
"""
Make graph from Voronoi neighbours
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def make_voronoi_neighbor_graph(
    vor_df,
    vor,
    x_col="x_col",
    y_col="y_col",
    particle_id_col=None,
    exclude_border_cells=False,
    border_col="is_border_cell",
    min_edge_length_um=0.5,
    um_per_px=0.65,
):
    """
    Make a NetworkX graph where particles are connected if their Voronoi
    cells share an edge.

    Parameters
    ----------
    vor_df : pandas.DataFrame
        DataFrame returned by compute_voronoi_from_df.
        Must be in the same point order used to create the Voronoi object.

    vor : scipy.spatial.Voronoi
        Voronoi object returned by compute_voronoi_from_df.

    x_col, y_col : str
        Coordinate columns.

    particle_id_col : str or None
        Optional column to use as node labels.
        If None, dataframe index is used.

    exclude_border_cells : bool
        If True, remove border Voronoi cells from the graph.

    border_col : str
        Column indicating border cells, if available.

    Returns
    -------
    G : networkx.Graph
        Graph with Voronoi-neighbour edges.

    graph_df : pandas.DataFrame
        Copy of vor_df with added graph coordination number.
    """

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



    # Optionally remove border cells
    if exclude_border_cells:
        if border_col not in graph_df.columns:
            raise ValueError(
                f"exclude_border_cells=True, but '{border_col}' is not in vor_df."
            )

        border_mask = graph_df[border_col].astype(bool).to_numpy()
        border_nodes = node_ids[border_mask]

        G.remove_nodes_from(border_nodes)

    # Add graph coordination number back into dataframe
    coord_nums = []

    for i, node_id in enumerate(node_ids):
        if node_id in G:
            coord_nums.append(G.degree[node_id])
        else:
            coord_nums.append(np.nan)

    graph_df["voronoi_graph_coord_num"] = coord_nums

    return G, graph_df


# %%
def plot_voronoi_neighbor_graph(
    image,
    graph_df,
    G,
    x_col="x_col",
    y_col="y_col",
    node_size=8,
    edge_alpha=0.35,
    node_alpha=0.9,
):
    """
    Plot Voronoi-neighbour graph on top of the microscopy image.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image, cmap="gray")

    # Draw edges
    for u, v in G.edges:
        x1 = G.nodes[u]["x"]
        y1 = G.nodes[u]["y"]
        x2 = G.nodes[v]["x"]
        y2 = G.nodes[v]["y"]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=0.7,
            alpha=edge_alpha,
        )

    # Draw nodes
    xs = [G.nodes[n]["x"] for n in G.nodes]
    ys = [G.nodes[n]["y"] for n in G.nodes]

    ax.scatter(
        xs,
        ys,
        s=node_size,
        alpha=node_alpha,
    )

    ax.set_aspect("equal")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title("Voronoi neighbour graph")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    plt.tight_layout()
    plt.show()

    return fig, ax



import json
import base64
import io
from pathlib import Path

import numpy as np
import matplotlib.cm as cm
from PIL import Image


def _image_to_base64_png(image):
    """
    Convert image array to base64 PNG for embedding in HTML.
    """
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = image - np.nanmin(image)
        mx = np.nanmax(image)
        if mx > 0:
            image = image / mx
        image = np.clip(255 * image, 0, 255).astype(np.uint8)

    img = Image.fromarray(image)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

def make_interactive_voronoi_graph_html(
    image,
    graph_df,
    G,
    output_html="interactive_voronoi_graph.html",
    x_col="x_col",
    y_col="y_col",
    poly_col="voronoi_poly",
    value_col="voronoi_num_edges",
    cmap_name="tab20",
    alpha=0.35,
    point_radius=2.5,
    edge_line_width=0.6,
    edge_alpha=0.55,
):
    """
    Make an interactive HTML viewer with:
    - microscopy image
    - Voronoi cells
    - particle centers
    - graph edges between connected Voronoi neighbours

    Controls:
    - Mouse wheel: zoom
    - Left mouse drag: pan
    - Reset view button
    - Show/hide image, cells, centers, graph edges
    """

    image = np.asarray(image)
    H, W = image.shape[:2]

    image_data_url = _image_to_base64_png(image)

    # -------------------------
    # Centers
    # -------------------------
    centers = graph_df[[x_col, y_col]].to_numpy(dtype=float)

    # -------------------------
    # Voronoi polygons
    # -------------------------
    values = graph_df[value_col].to_numpy(dtype=int)
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

    for poly, value in zip(graph_df[poly_col], values):
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

    # -------------------------
    # Graph edges
    # -------------------------
    graph_edges = []

    for u, v in G.edges:
        x1 = float(G.nodes[u]["x"])
        y1 = float(G.nodes[u]["y"])
        x2 = float(G.nodes[v]["x"])
        y2 = float(G.nodes[v]["y"])

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        graph_edges.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bbox": [xmin, ymin, xmax, ymax],
        })

    centers_json = json.dumps(np.round(centers, 2).tolist())
    polygons_json = json.dumps(polygons)
    graph_edges_json = json.dumps(graph_edges)
    legend_json = json.dumps(legend_items)
    image_json = json.dumps(image_data_url)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive Voronoi Graph Viewer</title>

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
        <input type="checkbox" id="showGraph" checked>
        graph lines
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
const graphEdges = {graph_edges_json};
const legendItems = {legend_json};

const pointRadius = {point_radius};
const edgeLineWidth = {edge_line_width};
const edgeAlpha = {edge_alpha};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const showImageBox = document.getElementById("showImage");
const showCellsBox = document.getElementById("showCells");
const showGraphBox = document.getElementById("showGraph");
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

    ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
    ctx.lineWidth = 0.75 / scale;
    ctx.stroke();
}}

function drawGraphEdge(edge) {{
    ctx.beginPath();
    ctx.moveTo(edge.x1, edge.y1);
    ctx.lineTo(edge.x2, edge.y2);

    ctx.strokeStyle = "rgba(0, 255, 255, " + edgeAlpha + ")";
    ctx.lineWidth = edgeLineWidth / scale;
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

    if (showGraphBox.checked) {{
        for (const edge of graphEdges) {{
            const b = edge.bbox;

            if (
                b[2] < viewLeft ||
                b[0] > viewRight ||
                b[3] < viewTop ||
                b[1] > viewBottom
            ) {{
                continue;
            }}

            drawGraphEdge(edge);
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
        " | y: " + viewTop.toFixed(1) + " - " + viewBottom.toFixed(1) +
        " | graph edges: " + graphEdges.length;
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
            " | zoom: " + scale.toFixed(3) +
            " | graph edges: " + graphEdges.length;

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
showGraphBox.addEventListener("change", draw);
showCentersBox.addEventListener("change", draw);
</script>

</body>
</html>
"""

    output_html = Path(output_html)
    output_html.write_text(html, encoding="utf-8")

    return output_html





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import ast


def _safe_poly_to_array(poly):
    """
    Convert polygon data to a numpy array.
    Works if poly is already an array/list, or if it was loaded from CSV as a string.
    """
    if isinstance(poly, str):
        poly = ast.literal_eval(poly)

    poly = np.asarray(poly, dtype=float)

    if poly.ndim != 2 or poly.shape[1] != 2:
        return None

    if not np.all(np.isfinite(poly)):
        return None

    return poly


def plot_voronoi_cutout_um(
    image,
    graph_df,
    G=None,
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    value_col="voronoi_num_edges",
    um_per_px=0.65,
    xlim_um=None,
    ylim_um=None,
    xlim_px=None,
    ylim_px=None,
    show_image=True,
    show_voronoi=True,
    show_graph=True,
    show_centers=True,
    color_voronoi_by_value=True,
    cmap_name="tab20",
    voronoi_alpha=0.35,
    voronoi_edge_color="white",
    voronoi_edge_width=0.4,
    graph_color="cyan",
    graph_line_width=0.6,
    graph_alpha=0.8,
    center_color="red",
    center_size=6,
    image_cmap="gray",
    figsize=(7, 7),
    title=None,
    save_path=None,
    dpi=300,
    show_legend=True,
    legend_loc="center left",
    legend_bbox_to_anchor=(1.02, 0.5),
    cell_alpha=0.30,
    legend_alpha=0.8,
    show_scale_bar=True,
    scale_bar_um=50,
    scale_bar_loc="lower left",
    scale_bar_color="white",
    scale_bar_fontsize=11,
    scale_bar_height_um=2,
    scale_bar_frameon=True,
    axis_label_fontsize=26,
    tick_fontsize=22,
    title_fontsize=28,
):
    """
    Plot a cut-out of the microscopy image with optional Voronoi cells,
    neighbour graph lines, and particle centers.

    Coordinates on the axes are shown in micrometers.

    Parameters
    ----------
    image : array
        Microscopy image, e.g. arr8.

    graph_df : pandas.DataFrame
        DataFrame containing particle x/y positions and Voronoi polygons.

    G : networkx.Graph or None
        Voronoi neighbour graph. Needed only if show_graph=True.

    xlim_um, ylim_um : tuple or None
        Crop limits in micrometers, for example:
        xlim_um=(100, 250), ylim_um=(50, 180)

    xlim_px, ylim_px : tuple or None
        Alternative crop limits in pixels.
        If both px and um limits are given, micrometer limits are used.

    show_image, show_voronoi, show_graph, show_centers : bool
        Switch different layers on/off.

    save_path : str or Path or None
        If given, saves the figure.

    Returns
    -------
    fig, ax
    """

    image = np.asarray(image)
    H, W = image.shape[:2]

    # Convert pixel limits to micrometer limits if needed
    if xlim_um is None:
        if xlim_px is not None:
            xlim_um = (xlim_px[0] * um_per_px, xlim_px[1] * um_per_px)
        else:
            xlim_um = (0, W * um_per_px)

    if ylim_um is None:
        if ylim_px is not None:
            ylim_um = (ylim_px[0] * um_per_px, ylim_px[1] * um_per_px)
        else:
            ylim_um = (0, H * um_per_px)

    fig, ax = plt.subplots(figsize=figsize)

    # -------------------------
    # Image
    # -------------------------
    if show_image:
        ax.imshow(
            image,
            cmap=image_cmap,
            extent=[0, W * um_per_px, H * um_per_px, 0],
        )

    # -------------------------
    # Voronoi cells
    # -------------------------

    cell_color_indices = {
    0: 1,
    1: 1,
    2: 0,
    3: 4,
    4: 6,
    5: 8,
    6: 18,
    7: 3,
}
        
    legend_handles = []

    if show_voronoi:
        polygons_um = []
        values_in_crop = []

        for _, row in graph_df.iterrows():
            poly = _safe_poly_to_array(row[poly_col])

            if poly is None:
                continue

            poly_um = poly * um_per_px

            xmin, ymin = np.min(poly_um, axis=0)
            xmax, ymax = np.max(poly_um, axis=0)

            if xmax < xlim_um[0] or xmin > xlim_um[1]:
                continue
            if ymax < ylim_um[0] or ymin > ylim_um[1]:
                continue

            polygons_um.append(poly_um)
            values_in_crop.append(int(row[value_col]))

        same_color_from_edge = 7

        if len(polygons_um) > 0:
            if color_voronoi_by_value and value_col in graph_df.columns:

                values_in_crop = np.asarray(values_in_crop, dtype=int)

                grouped_cell_values = np.array([
                    v if v < same_color_from_edge else same_color_from_edge
                    for v in values_in_crop
                ], dtype=int)

                # Complete legend always: 2, 3, 4, 5, 6, 7+
                legend_values = np.array([0, 1, 2, 3, 4, 5, 6, 7])

                cell_cmap = cm.get_cmap(cmap_name, 20)

                cell_value_to_rgba = {}

                for value in legend_values:
                    value = int(value)

                    color_index = cell_color_indices.get(value, value % 20)
                    r, g, b, _ = cell_cmap(color_index)

                    cell_value_to_rgba[value] = (r, g, b, cell_alpha)

                facecolors = [
                    cell_value_to_rgba[int(v)]
                    for v in grouped_cell_values
                ]

                legend_handles = [
                    Patch(
                        facecolor=(
                            cell_value_to_rgba[int(value)][0],
                            cell_value_to_rgba[int(value)][1],
                            cell_value_to_rgba[int(value)][2],
                            legend_alpha,
                        ),
                        edgecolor=voronoi_edge_color,
                        label=f"{int(value)}+" if int(value) == same_color_from_edge else str(int(value)),
                    )
                    for value in legend_values
                ]

            else:
                facecolors = [(1, 1, 1, cell_alpha)] * len(polygons_um)

            poly_collection = PolyCollection(
                polygons_um,
                facecolors=facecolors,
                edgecolors=voronoi_edge_color,
                linewidths=voronoi_edge_width,
            )

            ax.add_collection(poly_collection)

    if show_voronoi and color_voronoi_by_value and show_legend and len(legend_handles) > 0:
        ax.legend(
            handles=legend_handles,
            title="Number of\nVoronoi cell edges",
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=True,
        )

    # -------------------------
    # Graph lines
    # -------------------------
    if show_graph:
        if G is None:
            raise ValueError("show_graph=True, but G=None. Pass your NetworkX graph G.")

        graph_lines = []

        for u, v in G.edges:
            x1 = G.nodes[u]["x"] * um_per_px
            y1 = G.nodes[u]["y"] * um_per_px
            x2 = G.nodes[v]["x"] * um_per_px
            y2 = G.nodes[v]["y"] * um_per_px

            # Skip lines completely outside crop
            if max(x1, x2) < xlim_um[0] or min(x1, x2) > xlim_um[1]:
                continue
            if max(y1, y2) < ylim_um[0] or min(y1, y2) > ylim_um[1]:
                continue

            graph_lines.append([(x1, y1), (x2, y2)])

        if len(graph_lines) > 0:
            line_collection = LineCollection(
                graph_lines,
                colors=graph_color,
                linewidths=graph_line_width,
                alpha=graph_alpha,
            )

            ax.add_collection(line_collection)

    # -------------------------
    # Centers
    # -------------------------
    if show_centers:
        xs_um = graph_df[x_col].to_numpy(dtype=float) * um_per_px
        ys_um = graph_df[y_col].to_numpy(dtype=float) * um_per_px

        crop_mask = (
            (xs_um >= xlim_um[0]) &
            (xs_um <= xlim_um[1]) &
            (ys_um >= ylim_um[0]) &
            (ys_um <= ylim_um[1])
        )

        ax.scatter(
            xs_um[crop_mask],
            ys_um[crop_mask],
            s=center_size,
            c=center_color,
            linewidths=0,
            zorder=10,
        )

    # -------------------------
    # Axes and layout
    # -------------------------
    ax.set_xlim(xlim_um)
    ax.set_ylim(ylim_um[1], ylim_um[0])  # reversed because image y-axis points downward

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$\mu$m]", fontsize=axis_label_fontsize)
    ax.set_ylabel(r"$y$ [$\mu$m]", fontsize=axis_label_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, pad=15)
        
    # -------------------------
    # Scale bar
    # -------------------------
    if show_scale_bar:
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm

        fontprops = fm.FontProperties(size=scale_bar_fontsize)

        scalebar = AnchoredSizeBar(
            ax.transData,
            scale_bar_um,
            rf"{scale_bar_um:g} $\mu$m",
            loc=scale_bar_loc,
            pad=0.4,
            borderpad=0.6,
            sep=5,
            color=scale_bar_color,
            frameon=scale_bar_frameon,
            size_vertical=scale_bar_height_um,
            fontproperties=fontprops,
        )

        if scale_bar_frameon:
            scalebar.patch.set_facecolor("black")
            scalebar.patch.set_alpha(0.45)
            scalebar.patch.set_edgecolor("none")

        ax.add_artist(scalebar)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax