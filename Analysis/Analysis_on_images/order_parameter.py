

from matplotlib.collections import LineCollection
import numpy as np


def compute_order_parameter_graph(
    df,
    G,
    order_n=6,
    x_col="x_col",
    y_col="y_col",
    particle_id_col=None,
    min_neighbours=2,
    required_coord_num=None,
):
    """
    Compute local bond-orientational order parameter psi_n using graph neighbours.

    Neighbours are taken from the NetworkX graph G.

    Returns a copy of df with:
    - psi_n
    - coord_num
    - psi_n_valid

    Parameters
    ----------
    df : pandas.DataFrame
        Particle dataframe.

    G : networkx.Graph
        Graph where connected nodes are neighbours.

    order_n : int
        Symmetry order. For hexagonal order use order_n=6.

    x_col, y_col : str
        Coordinate columns.

    particle_id_col : str or None
        If None, df index is used as graph node ID.
        If not None, this column is used as graph node ID.

    min_neighbours : int
        Minimum number of neighbours needed to calculate psi_n.
        For example, min_neighbours=2 means particles with 0 or 1 neighbour
        get psi_n = NaN and psi_n_valid = False.

    required_coord_num : int or None
        If None, calculate psi_n for all particles with at least min_neighbours.
        If an integer, for example 3, only calculate psi_n for particles with
        exactly that number of neighbours.
    """

    df = df.copy().reset_index(drop=True)

    valid_xy = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    df = df[valid_xy].reset_index(drop=True)

    coords = df[[x_col, y_col]].to_numpy(dtype=float)

    if particle_id_col is None:
        node_ids = df.index.to_numpy()
    else:
        node_ids = df[particle_id_col].to_numpy()

    node_to_row = {node_id: i for i, node_id in enumerate(node_ids)}

    psi_values = np.full(len(df), np.nan, dtype=float)
    coord_nums = np.zeros(len(df), dtype=int)
    psi_valid = np.zeros(len(df), dtype=bool)

    for i, node_id in enumerate(node_ids):

        if node_id not in G:
            coord_nums[i] = 0
            psi_values[i] = np.nan
            psi_valid[i] = False
            continue

        neighbours = list(G.neighbors(node_id))

        # Keep only neighbours that also exist in this dataframe
        neighbours = [n for n in neighbours if n in node_to_row]

        coord_nums[i] = len(neighbours)

        # Not enough neighbours, for example only 0 or 1 neighbour
        if len(neighbours) < min_neighbours:
            psi_values[i] = np.nan
            psi_valid[i] = False
            continue

        # Optional: only calculate for exactly a chosen coordination number
        if required_coord_num is not None:
            if len(neighbours) != required_coord_num:
                psi_values[i] = np.nan
                psi_valid[i] = False
                continue

        neighbour_rows = [node_to_row[n] for n in neighbours]

        dx = coords[neighbour_rows, 0] - coords[i, 0]
        dy = coords[neighbour_rows, 1] - coords[i, 1]

        angles = np.arctan2(dy, dx)

        psi_complex = np.mean(np.exp(1j * order_n * angles))

        psi_values[i] = np.abs(psi_complex)
        psi_valid[i] = True

    psi_col = f"psi_{order_n}"
    valid_col = f"{psi_col}_valid"

    df[psi_col] = psi_values
    df["coord_num"] = coord_nums
    df[valid_col] = psi_valid

    return df

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def fraction_limits_from_image(
    image_shape,
    um_per_px,
    x_frac=(0, 1),
    y_frac=(0, 1),
):
    """
    Convert image fractions to x/y axis limits in microns.

    image_shape is usually arr8.shape = (height, width).

    Examples
    --------
    x_frac=(0, 0.5) gives left half.
    x_frac=(0.5, 1) gives right half.
    y_frac=(0, 0.5) gives top half.
    y_frac=(0.5, 1) gives bottom half.
    """

    H, W = image_shape[:2]

    xlim = (
        x_frac[0] * W * um_per_px,
        x_frac[1] * W * um_per_px,
    )

    ylim = (
        y_frac[0] * H * um_per_px,
        y_frac[1] * H * um_per_px,
    )

    return xlim, ylim

def plot_order_parameter_graph_map(
    order_df,
    order_n=6,
    x_col="x",
    y_col="y",
    um_per_px=0.06,
    s=4,
    n_bins=10,
    image_shape=None,
    x_frac=(0, 1),
    y_frac=(0, 1),
    output_path=None,
):
    """
    Plot local bond-orientational order parameter from graph neighbours.

    This expects order_df to already contain a column like:
        psi_6, psi_3, etc.

    Example:
        order_df = compute_order_parameter_graph(...)
        plot_order_parameter_graph_map(order_df, order_n=6)
    """

    psi_col = f"psi_{order_n}"

    if psi_col not in order_df.columns:
        raise ValueError(
            f"{psi_col} not found in order_df.\n"
            f"Available columns are:\n{list(order_df.columns)}"
        )

    valid = (
        np.isfinite(order_df[x_col])
        & np.isfinite(order_df[y_col])
        & np.isfinite(order_df[psi_col])
    )

    plot_df = order_df[valid].copy()

    x_um = plot_df[x_col].to_numpy(dtype=float) * um_per_px
    y_um = plot_df[y_col].to_numpy(dtype=float) * um_per_px
    psi_values = plot_df[psi_col].to_numpy(dtype=float)

    bounds = np.linspace(0, 1, n_bins + 1)
    cmap = plt.cm.get_cmap("OrRd", n_bins)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    sc = ax.scatter(
        x_um,
        y_um,
        c=psi_values,
        s=s,
        cmap=cmap,
        norm=norm,
    )

    cb = plt.colorbar(
        sc,
        ax=ax,
        pad=0.02,
        boundaries=bounds,
        ticks=bounds,
        spacing="proportional",
    )

    cb.set_label(label=rf"$|\psi_{{{order_n}}}|$", size=16)
    cb.ax.tick_params(labelsize=12)

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    if image_shape is not None:
        xlim, ylim = fraction_limits_from_image(
            image_shape,
            um_per_px=um_per_px,
            x_frac=x_frac,
            y_frac=y_frac,
        )

        ax.set_xlim(xlim)

        # image-style coordinates: y = 0 at top
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.invert_yaxis()

    mean_psi = np.nanmean(plot_df[psi_col])
    mean_coord = np.nanmean(plot_df["coord_num"]) if "coord_num" in plot_df.columns else np.nan

    ax.set_title(
        rf"Local graph-based bond order $|\psi_{{{order_n}}}|$"
        + "\n"
        + f"mean {psi_col} = {mean_psi:.3f}, "
        + f"mean coord = {mean_coord:.2f}"
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt


def plot_order_parameter_histogram(
    df,
    order_n=6,
    psi_col=None,
    bins=30,
    output_path=None,
    title=None,
    show_mean=True,
    show_median=True,
    xlim=(0, 1),
):
    """
    Plot histogram of local bond-orientational order parameter psi_n.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing psi_n values.

    order_n : int
        Order parameter symmetry. Used to infer psi column name if psi_col is None.

    psi_col : str or None
        Column containing order parameter values.
        If None, uses f"psi_{order_n}".

    bins : int
        Number of histogram bins.

    output_path : str or Path or None
        If given, saves the figure.

    title : str or None
        Plot title.

    show_mean : bool
        Draw vertical line at mean psi_n.

    show_median : bool
        Draw vertical line at median psi_n.

    xlim : tuple or None
        x-axis limits.
    """

    if psi_col is None:
        psi_col = f"psi_{order_n}"

    if psi_col not in df.columns:
        raise ValueError(f"Column '{psi_col}' not found in dataframe.")

    values = df[psi_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)

    if show_mean and len(values) > 0:
        mean_val = np.mean(values)
        ax.axvline(
            mean_val,
            linestyle="--",
            linewidth=2,
            label=f"mean = {mean_val:.3f}",
        )

    if show_median and len(values) > 0:
        median_val = np.median(values)
        ax.axvline(
            median_val,
            linestyle=":",
            linewidth=2,
            label=f"median = {median_val:.3f}",
        )

    if title is None:
        title = rf"Histogram of $|\psi_{order_n}|$"

    ax.set_title(title)
    ax.set_xlabel(rf"$|\psi_{order_n}|$")
    ax.set_ylabel("Number of particles")

    if xlim is not None:
        ax.set_xlim(*xlim)

    if show_mean or show_median:
        ax.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax



def plot_order_parameter_graph_map_on_image(
    image,
    order_df,
    order_n=6,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    s=8,
    n_bins=10,
    image_shape=None,
    x_frac=(0, 1),
    y_frac=(0, 1),
    output_path=None,
):
    """
    Plot local bond-orientational order parameter on top of the microscopy image.

    order_df must contain:
        psi_6, psi_3, etc.
    """

    psi_col = f"psi_{order_n}"

    if psi_col not in order_df.columns:
        raise ValueError(
            f"{psi_col} not found in order_df.\n"
            f"Available columns are:\n{list(order_df.columns)}"
        )

    image = np.asarray(image)

    if image_shape is None:
        image_shape = image.shape

    valid = (
        np.isfinite(order_df[x_col])
        & np.isfinite(order_df[y_col])
        & np.isfinite(order_df[psi_col])
    )

    plot_df = order_df[valid].copy()

    # Your particle positions are in pixels, so scatter in pixel coordinates
    x_px = plot_df[x_col].to_numpy(dtype=float)
    y_px = plot_df[y_col].to_numpy(dtype=float)
    psi_values = plot_df[psi_col].to_numpy(dtype=float)

    bounds = np.linspace(0, 1, n_bins + 1)
    cmap = plt.cm.get_cmap("OrRd", n_bins)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # Image background
    ax.imshow(image, cmap="gray")

    # Order parameter overlay
    sc = ax.scatter(
        x_px,
        y_px,
        c=psi_values,
        s=s,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )

    cb = plt.colorbar(
        sc,
        ax=ax,
        pad=0.02,
        boundaries=bounds,
        ticks=bounds,
        spacing="proportional",
    )

    cb.set_label(label=rf"$|\psi_{{{order_n}}}|$", size=16)
    cb.ax.tick_params(labelsize=12)

    H, W = image_shape[:2]

    # Crop using image fractions, still in pixel coordinates
    xlim_px = (x_frac[0] * W, x_frac[1] * W)
    ylim_px = (y_frac[0] * H, y_frac[1] * H)

    ax.set_xlim(xlim_px)
    ax.set_ylim(ylim_px[1], ylim_px[0])  # image-style y-axis

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")

    mean_psi = np.nanmean(plot_df[psi_col])
    mean_coord = (
        np.nanmean(plot_df["coord_num"])
        if "coord_num" in plot_df.columns
        else np.nan
    )

    ax.set_title(
        rf"Local graph-based bond order $|\psi_{{{order_n}}}|$ on image"
        + "\n"
        + f"mean {psi_col} = {mean_psi:.3f}, "
        + f"mean coord = {mean_coord:.2f}"
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax




import numpy as np
import matplotlib.colors as mcolors

def psi_to_hex_colors(values, vmin=0.0, vmax=1.0, nan_color="#808080"):
    """
    Map psi values to colors:
    low  -> yellow
    high -> red
    NaN  -> gray
    """
    values = np.asarray(values, dtype=float)

    c_low = np.array(mcolors.to_rgb("#ffff00"))  # yellow
    c_high = np.array(mcolors.to_rgb("#ff0000")) # red

    out = []
    for v in values:
        if not np.isfinite(v):
            out.append(nan_color)
            continue

        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        t = np.clip(t, 0, 1)

        rgb = (1 - t) * c_low + t * c_high
        out.append(mcolors.to_hex(rgb))

    return out


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


def _psi_to_color(value, vmin=0.0, vmax=1.0, nan_color="rgb(150, 150, 150)"):
    """
    Map psi value to yellow-red color.
    low  -> yellow
    high -> red
    NaN  -> gray
    """
    try:
        v = float(value)
    except Exception:
        return nan_color

    if not np.isfinite(v):
        return nan_color

    if vmax <= vmin:
        t = 0.0
    else:
        t = (v - vmin) / (vmax - vmin)

    t = float(np.clip(t, 0.0, 1.0))

    # yellow = (255,255,0), red = (255,0,0)
    r = 255
    g = int(round(255 * (1.0 - t)))
    b = 0

    return f"rgb({r}, {g}, {b})"

def _parse_voronoi_polygon(poly):
    import json as _json
    import re as _re

    if poly is None:
        return None

    if not isinstance(poly, str):
        return np.asarray(poly, dtype=float)

    text = poly.strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None

    try:
        return np.asarray(_json.loads(text), dtype=float)
    except Exception:
        pass

    nums = _re.findall(
        r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
        text
    )

    if len(nums) < 6 or len(nums) % 2 != 0:
        return None

    return np.asarray(nums, dtype=float).reshape(-1, 2)


def make_interactive_voronoi_graph_order_html(
    image,
    graph_df,
    G,
    output_html="interactive_voronoi_graph_order.html",
    x_col="x_col",
    y_col="y_col",
    poly_col="voronoi_poly",
    value_col="voronoi_num_edges",
    psi_col="psi_3",
    cmap_name="tab20",
    alpha=0.35,
    point_radius=4.5,
    edge_line_width=0.8,
    edge_alpha=1.0,
    center_outline=True,
):
    """
    Make an interactive HTML viewer with:
    - microscopy image
    - Voronoi cells colored by value_col, usually voronoi_num_edges
    - graph edges between connected Voronoi neighbours
    - particle centers colored by local order parameter psi_col

    Center color scale:
    - yellow = low psi
    - red = high psi
    - gray = NaN psi

    Controls:
    - Mouse wheel: zoom
    - Left mouse drag: pan
    - Reset view button
    - Show/hide image, cells, centers, graph edges

    Notes
    -----
    graph_df should already contain psi_col, for example psi_3 or psi_6.
    So first run compute_order_parameter_graph(...), and pass the resulting order_df.
    """

    image = np.asarray(image)
    H, W = image.shape[:2]

    if psi_col not in graph_df.columns:
        raise ValueError(
            f"Column '{psi_col}' not found in graph_df.\n"
            f"Available columns are:\n{list(graph_df.columns)}"
        )

    image_data_url = _image_to_base64_png(image)

    # -------------------------
    # Centers: x/y plus psi color
    # -------------------------
    centers = []

    for _, row in graph_df.iterrows():
        x = row[x_col]
        y = row[y_col]
        psi = row[psi_col]

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        if np.isfinite(psi):
            psi_value = float(psi)
        else:
            psi_value = None

        centers.append({
            "x": float(x),
            "y": float(y),
            "psi": psi_value,
            "color": _psi_to_color(psi, vmin=0.0, vmax=1.0),
        })

    # -------------------------
    # Voronoi polygons, colored by value_col
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
        poly = _parse_voronoi_polygon(poly)

        if poly is None:
            continue

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

    centers_json = json.dumps(centers)
    polygons_json = json.dumps(polygons)
    graph_edges_json = json.dumps(graph_edges)
    legend_json = json.dumps(legend_items)
    image_json = json.dumps(image_data_url)

    # Display name for legend, e.g. psi_3 -> ψ₃-like HTML with subscript
    if psi_col.startswith("psi_"):
        psi_n_label = psi_col.split("psi_", 1)[1]
        psi_label_html = f"|ψ<sub>{psi_n_label}</sub>|"
    else:
        psi_label_html = psi_col

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive Voronoi Graph and Order Parameter Viewer</title>

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

    .legend-box {{
        position: fixed;
        right: 18px;
        background: rgba(20, 20, 20, 0.88);
        color: white;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 13px;
        z-index: 20;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}

    #edgeLegend {{
        top: 70px;
        max-height: 56vh;
        overflow-y: auto;
    }}

    #psiLegend {{
        top: 70px;
        right: 270px;
        width: 210px;
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

    .psi-gradient {{
        width: 100%;
        height: 20px;
        border: 1px solid white;
        border-radius: 4px;
        background: linear-gradient(to right, rgb(255,255,0), rgb(255,0,0));
        margin-bottom: 6px;
    }}

    .psi-labels {{
        display: flex;
        justify-content: space-between;
        font-size: 12px;
    }}

    .nan-row {{
        margin-top: 8px;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .nan-color {{
        width: 22px;
        height: 14px;
        border: 1px solid white;
        background: rgb(150,150,150);
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

    <span>Cells: {value_col}</span>
    <span>Centers: {psi_col}</span>

    <div id="info"></div>
</div>

<canvas id="canvas"></canvas>
<div id="edgeLegend" class="legend-box"></div>
<div id="psiLegend" class="legend-box">
    <div class="legend-title">Center color: {psi_label_html}</div>
    <div class="psi-gradient"></div>
    <div class="psi-labels">
        <span>0 low</span>
        <span>1 high</span>
    </div>
    <div class="nan-row">
        <div class="nan-color"></div>
        <span>NaN / missing</span>
    </div>
</div>

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
const centerOutline = {str(bool(center_outline)).lower()};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const showImageBox = document.getElementById("showImage");
const showCellsBox = document.getElementById("showCells");
const showGraphBox = document.getElementById("showGraph");
const showCentersBox = document.getElementById("showCenters");
const info = document.getElementById("info");
const edgeLegend = document.getElementById("edgeLegend");

function buildEdgeLegend() {{
    let html = "<div class='legend-title'>Number of Voronoi cell edges</div>";

    for (const item of legendItems) {{
        html += "<div class='legend-row'>";
        html += "<div class='legend-color' style='background:" + item.color + ";'></div>";
        html += "<span>" + item.edge + "</span>";
        html += "</div>";
    }}

    edgeLegend.innerHTML = html;
}}

buildEdgeLegend();

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
        const r = pointRadius / scale;

        for (const p of centers) {{
            const x = p.x;
            const y = p.y;

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
            ctx.fillStyle = p.color;
            ctx.fill();

            if (centerOutline) {{
                ctx.strokeStyle = "rgba(70, 0, 0, 0.9)";
                ctx.lineWidth = 0.5 / scale;
                ctx.stroke();
            }}
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




def plot_order_parameter_cutout_um(
    image,
    graph_df,
    G=None,
    x_col="x",
    y_col="y",
    psi_col="psi_6",
    um_per_px=0.65,
    xlim_um=None,
    ylim_um=None,
    xlim_px=None,
    ylim_px=None,
    show_image=True,
    show_graph=True,
    show_centers=True,
    cmap_name="tab20",
    graph_color="cyan",
    graph_line_width=0.6,
    graph_alpha=0.8,
    center_size=20,
    center_cmap="viridis",
    center_vmin=0.0,
    center_vmax=1.0,
    center_edge_color="black",
    center_edge_width=0.3,
    image_cmap="gray",
    figsize=(7, 6), 
    title=None,
    save_path=None,
    dpi=300,
    show_legend=True,
    legend_loc="center left",
    legend_bbox_to_anchor=(1.02, 0.5),
    cell_alpha=0.30,
    legend_alpha=0.8,
    show_colorbar=True,
    invalid_color="blue",
    titlefontsize = 35,
    labelfontsize = 30,
    tickfontsize = 30,
    fontsizecolorbar = 30,
    fontsizetickcbar = 50
):
    """
    and particle centers colored by local bond-orientational order parameter.

    Parameters
    ----------
    graph_df : pandas.DataFrame
        DataFrame containing x/y positions and optionally Voronoi polygons.
        Must contain psi_col if show_centers=True.

    psi_col : str
        Name of the order parameter column, e.g. "psi_6".
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
    # Centers colored by order parameter
    # -------------------------
    if show_centers:
        if psi_col not in graph_df.columns:
            raise ValueError(f"Column '{psi_col}' not found in graph_df.")

        xs_um = graph_df[x_col].to_numpy(dtype=float) * um_per_px
        ys_um = graph_df[y_col].to_numpy(dtype=float) * um_per_px
        psi_vals = graph_df[psi_col].to_numpy(dtype=float)

        # validity column, e.g. psi_6_valid
        valid_col = f"{psi_col}_valid"

        if valid_col in graph_df.columns:
            psi_valid = graph_df[valid_col].to_numpy(dtype=bool)
        else:
            # fallback: valid if psi is not NaN
            psi_valid = np.isfinite(psi_vals)       


        crop_mask = (
            (xs_um >= xlim_um[0]) &
            (xs_um <= xlim_um[1]) &
            (ys_um >= ylim_um[0]) &
            (ys_um <= ylim_um[1])
        )

        valid_mask = crop_mask & psi_valid
        invalid_mask = crop_mask & (~psi_valid)

        # First plot invalid points as blue
        ax.scatter(
            xs_um[invalid_mask],
            ys_um[invalid_mask],
            s=center_size,
            c=invalid_color,
            zorder=10,
            edgecolors=center_edge_color,
            linewidths=center_edge_width,
        )

        # Then plot valid points with colormap
        sc = ax.scatter(
            xs_um[valid_mask],
            ys_um[valid_mask],
            s=center_size,
            c=psi_vals[valid_mask],
            cmap=center_cmap,
            vmin=center_vmin,
            vmax=center_vmax,
            zorder=11,
            edgecolors=center_edge_color,
            linewidths=center_edge_width,
        )

        if show_colorbar and np.any(valid_mask):
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(rf"${psi_col}$",fontsize=fontsizecolorbar)
            cbar.ax.tick_params(labelsize=fontsizetickcbar)
    # -------------------------
    # Axes and layout
    # -------------------------
    ax.set_xlim(xlim_um)
    ax.set_ylim(ylim_um[1], ylim_um[0])

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$\mu$m]", fontsize=labelfontsize)
    ax.set_ylabel(r"$y$ [$\mu$m]", fontsize=labelfontsize)
    ax.tick_params(axis='both', labelsize=tickfontsize)

    if title is not None:
        ax.set_title(title, fontsize=titlefontsize)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_valid_order_parameter_histogram_colored(
    df,
    order_n=3,
    psi_col=None,
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    xlim_um=None,
    ylim_um=None,
    xlim_px=None,
    ylim_px=None,
    bins=40,
    center_cmap="plasma",
    center_vmin=0.0,
    center_vmax=1.0,
    figsize=(7, 5),
    title=None,
    show_mean=True,
    show_median=True,
    show_colorbar=True,
    save_path=None,
    dpi=300,
    titlefontsize=35,
    labelfontsize=30,
    tickfontsize=30,
    legendfontsize=30,
    borderpad=0.5,
    labelspacing=0.5,
    handlelength=2.0,
    normalize=False
):
    """
    Plot a histogram of only valid order-parameter values.

    The histogram bars are colored using the same colormap and value range
    as the order-parameter image overlay.

    If xlim/ylim are given, the histogram is made only for that crop.
    """

    if psi_col is None:
        psi_col = f"psi_{order_n}"

    if psi_col not in df.columns:
        raise ValueError(f"Column '{psi_col}' not found in dataframe.")

    valid_col = f"{psi_col}_valid"

    psi_vals = df[psi_col].to_numpy(dtype=float)

    # Use the validity column if available, otherwise use finite psi values
    if valid_col in df.columns:
        psi_valid = df[valid_col].to_numpy(dtype=bool)
    else:
        psi_valid = np.isfinite(psi_vals)

    mask = psi_valid & np.isfinite(psi_vals)

    # Optional crop, using the same convention as plot_order_parameter_cutout_um
    if xlim_px is not None:
        xlim_um = (xlim_px[0] * um_per_px, xlim_px[1] * um_per_px)

    if ylim_px is not None:
        ylim_um = (ylim_px[0] * um_per_px, ylim_px[1] * um_per_px)

    if xlim_um is not None or ylim_um is not None:
        xs_um = df[x_col].to_numpy(dtype=float) * um_per_px
        ys_um = df[y_col].to_numpy(dtype=float) * um_per_px

        if xlim_um is not None:
            mask &= (xs_um >= xlim_um[0]) & (xs_um <= xlim_um[1])

        if ylim_um is not None:
            mask &= (ys_um >= ylim_um[0]) & (ys_um <= ylim_um[1])

    values = psi_vals[mask]

    if len(values) == 0:
        raise ValueError("No valid order-parameter values found for this selection.")

    # Fixed range from 0 to 1, because |psi_n| should lie in this interval
    counts, bin_edges = np.histogram(
        values,
        bins=bins,
        range=(center_vmin, center_vmax),
    )

    if normalize:
        counts = counts / counts.sum()
        ylabel = "Normalized frequency"
    else:
        ylabel = "Number of particles"

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    cmap = plt.colormaps[center_cmap]
    norm = mcolors.Normalize(vmin=center_vmin, vmax=center_vmax)
    bar_colors = cmap(norm(bin_centers))

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        bin_centers,
        counts,
        width=bin_widths,
        align="center",
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )

    if show_mean:
        mean_val = np.mean(values)
        ax.axvline(
            mean_val,
            linestyle="--",
            linewidth=2,
            color="black",
            label=f"mean = {mean_val:.3f}",
        )

    if show_median:
        median_val = np.median(values)
        ax.axvline(
            median_val,
            linestyle=":",
            linewidth=2,
            color="black",
            label=f"median = {median_val:.3f}",
        )

    ax.set_xlim(center_vmin, center_vmax)
    ax.set_xlabel(rf"$|\psi_{{{order_n}}}|$", fontsize=labelfontsize)
    ax.set_ylabel(ylabel, fontsize=labelfontsize)
    ax.tick_params(axis='both', labelsize=tickfontsize)

    if title is None:
        title = rf"Histogram of valid $|\psi_{{{order_n}}}|$ values"

    ax.set_title(title, fontsize=titlefontsize)

    if show_mean or show_median:
        ax.legend(
            fontsize=legendfontsize,
            frameon=True,
            borderpad=borderpad,
            labelspacing=labelspacing,
            handlelength=handlelength,
        )

    if show_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(rf"$|\psi_{{{order_n}}}|$", fontsize=labelfontsize)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax



from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_valid_order_parameter_values_line(
    df,
    order_n=3,
    psi_col=None,
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    xlim_um=None,
    ylim_um=None,
    xlim_px=None,
    ylim_px=None,
    center_cmap="plasma",
    center_vmin=0.0,
    center_vmax=1.0,
    sort_values=True,
    show_points=True,
    point_size=18,
    line_width=2.0,
    figsize=(7, 5),
    title=None,
    show_mean=True,
    show_median=True,
    show_colorbar=True,
    save_path=None,
    dpi=300,
):
    """
    Plot all valid order-parameter values as a line, without binning.

    If sort_values=True, the values are sorted from low to high.
    If sort_values=False, the values are plotted in the order they appear
    in the dataframe after filtering.
    """

    if psi_col is None:
        psi_col = f"psi_{order_n}"

    if psi_col not in df.columns:
        raise ValueError(f"Column '{psi_col}' not found in dataframe.")

    valid_col = f"{psi_col}_valid"

    psi_vals = df[psi_col].to_numpy(dtype=float)

    if valid_col in df.columns:
        psi_valid = df[valid_col].to_numpy(dtype=bool)
    else:
        psi_valid = np.isfinite(psi_vals)

    mask = psi_valid & np.isfinite(psi_vals)

    # Optional crop, same convention as your image cutout function
    if xlim_px is not None:
        xlim_um = (xlim_px[0] * um_per_px, xlim_px[1] * um_per_px)

    if ylim_px is not None:
        ylim_um = (ylim_px[0] * um_per_px, ylim_px[1] * um_per_px)

    if xlim_um is not None or ylim_um is not None:
        xs_um = df[x_col].to_numpy(dtype=float) * um_per_px
        ys_um = df[y_col].to_numpy(dtype=float) * um_per_px

        if xlim_um is not None:
            mask &= (xs_um >= xlim_um[0]) & (xs_um <= xlim_um[1])

        if ylim_um is not None:
            mask &= (ys_um >= ylim_um[0]) & (ys_um <= ylim_um[1])

    values = psi_vals[mask]

    if len(values) == 0:
        raise ValueError("No valid order-parameter values found for this selection.")

    if sort_values:
        values = np.sort(values)

    x = np.arange(1, len(values) + 1)

    cmap = plt.colormaps[center_cmap]
    norm = mcolors.Normalize(vmin=center_vmin, vmax=center_vmax)

    fig, ax = plt.subplots(figsize=figsize)

    # Make colored line segments
    points = np.array([x, values]).T.reshape(-1, 1, 2)

    if len(values) > 1:
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_values = 0.5 * (values[:-1] + values[1:])

        line_collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=line_width,
        )
        line_collection.set_array(segment_values)
        ax.add_collection(line_collection)

    if show_points:
        ax.scatter(
            x,
            values,
            c=values,
            cmap=cmap,
            norm=norm,
            s=point_size,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

    if show_mean:
        mean_val = np.mean(values)
        ax.axhline(
            mean_val,
            linestyle="--",
            linewidth=1.8,
            color="black",
            label=f"mean = {mean_val:.3f}",
        )

    if show_median:
        median_val = np.median(values)
        ax.axhline(
            median_val,
            linestyle=":",
            linewidth=1.8,
            color="black",
            label=f"median = {median_val:.3f}",
        )

    ax.set_xlim(1, len(values))
    ax.set_ylim(center_vmin, center_vmax)

    if sort_values:
        ax.set_xlabel("Valid particle number, sorted by order parameter")
    else:
        ax.set_xlabel("Valid particle number")

    ax.set_ylabel(rf"$|\psi_{{{order_n}}}|$")

    if title is None:
        title = rf"Valid $|\psi_{{{order_n}}}|$ values"

    ax.set_title(title)

    if show_mean or show_median:
        ax.legend()

    if show_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(rf"$|\psi_{{{order_n}}}|$")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt


def plot_coordination_number_histogram(
    df,
    coord_col="coord_num",
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    xlim_um=None,
    ylim_um=None,
    xlim_px=None,
    ylim_px=None,
    normalize=False,
    show_values=True,
    figsize=(7, 5),
    title=None,
    save_path=None,
    dpi=300,
):
    """
    Plot coordination number against frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing a coordination-number column.

    coord_col : str
        Column containing the coordination number, usually "coord_num".

    normalize : bool
        If False, y-axis shows particle count.
        If True, y-axis shows relative frequency.

    xlim_px, ylim_px : tuple or None
        Optional crop in pixel coordinates, using the same convention
        as plot_order_parameter_cutout_um.
    """

    if coord_col not in df.columns:
        raise ValueError(f"Column '{coord_col}' not found in dataframe.")

    mask = np.isfinite(df[coord_col].to_numpy(dtype=float))

    # Convert pixel crop limits to micrometer limits
    if xlim_px is not None:
        xlim_um = (xlim_px[0] * um_per_px, xlim_px[1] * um_per_px)

    if ylim_px is not None:
        ylim_um = (ylim_px[0] * um_per_px, ylim_px[1] * um_per_px)

    # Optional crop
    if xlim_um is not None or ylim_um is not None:
        xs_um = df[x_col].to_numpy(dtype=float) * um_per_px
        ys_um = df[y_col].to_numpy(dtype=float) * um_per_px

        if xlim_um is not None:
            mask &= (xs_um >= xlim_um[0]) & (xs_um <= xlim_um[1])

        if ylim_um is not None:
            mask &= (ys_um >= ylim_um[0]) & (ys_um <= ylim_um[1])

    coord_values = df.loc[mask, coord_col].to_numpy(dtype=int)

    if len(coord_values) == 0:
        raise ValueError("No coordination numbers found for this selection.")

    unique_coord, counts = np.unique(coord_values, return_counts=True)

    if normalize:
        frequencies = counts / counts.sum()
        ylabel = "Relative frequency"
    else:
        frequencies = counts
        ylabel = "Number of particles"

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        unique_coord,
        frequencies,
        width=0.8,
        edgecolor="black",
        alpha=0.85,
    )

    if show_values:
        for x, y in zip(unique_coord, frequencies):
            if normalize:
                label = f"{y:.2f}"
            else:
                label = str(int(y))

            ax.text(
                x,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Coordination number")
    ax.set_ylabel(ylabel)

    if title is None:
        title = "Coordination number distribution"

    ax.set_title(title)

    ax.set_xticks(unique_coord)
    ax.set_xlim(unique_coord.min() - 0.7, unique_coord.max() + 0.7)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_coordination_vs_average_order_parameter(
    df,
    order_n=3,
    psi_col=None,
    coord_col="coord_num",
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    xlim_um=None,
    ylim_um=None,
    xlim_px=None,
    ylim_px=None,
    use_only_valid=True,
    show_counts=True,
    show_errorbars=False,
    errorbar_type="std",   # "std" or "sem"
    figsize=(7, 5),
    title=None,
    save_path=None,
    dpi=300,
):
    """
    Plot coordination number against the average order parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing coord_num and psi_n.

    order_n : int
        Used to infer psi_col if psi_col is None.

    psi_col : str or None
        For example "psi_3" or "psi_6".

    coord_col : str
        Column with coordination number, usually "coord_num".

    use_only_valid : bool
        If True, only use rows where psi_n is marked valid
        (via psi_n_valid if present).

    show_counts : bool
        If True, show the number of particles above each bar.

    show_errorbars : bool
        If True, add std or sem error bars.

    xlim_px, ylim_px : tuple or None
        Optional cutout in pixel coordinates.
    """

    if psi_col is None:
        psi_col = f"psi_{order_n}"

    if psi_col not in df.columns:
        raise ValueError(f"Column '{psi_col}' not found in dataframe.")

    if coord_col not in df.columns:
        raise ValueError(f"Column '{coord_col}' not found in dataframe.")

    mask = np.isfinite(df[coord_col].to_numpy(dtype=float))
    mask &= np.isfinite(df[psi_col].to_numpy(dtype=float))

    valid_col = f"{psi_col}_valid"
    if use_only_valid and valid_col in df.columns:
        mask &= df[valid_col].to_numpy(dtype=bool)

    # optional crop
    if xlim_px is not None:
        xlim_um = (xlim_px[0] * um_per_px, xlim_px[1] * um_per_px)
    if ylim_px is not None:
        ylim_um = (ylim_px[0] * um_per_px, ylim_px[1] * um_per_px)

    if xlim_um is not None or ylim_um is not None:
        xs_um = df[x_col].to_numpy(dtype=float) * um_per_px
        ys_um = df[y_col].to_numpy(dtype=float) * um_per_px

        if xlim_um is not None:
            mask &= (xs_um >= xlim_um[0]) & (xs_um <= xlim_um[1])

        if ylim_um is not None:
            mask &= (ys_um >= ylim_um[0]) & (ys_um <= ylim_um[1])

    plot_df = df.loc[mask, [coord_col, psi_col]].copy()

    if len(plot_df) == 0:
        raise ValueError("No valid data found for this selection.")

    grouped = plot_df.groupby(coord_col)[psi_col]
    mean_vals = grouped.mean()
    counts = grouped.count()

    if show_errorbars:
        if errorbar_type == "std":
            yerr = grouped.std().fillna(0).to_numpy()
        elif errorbar_type == "sem":
            yerr = grouped.sem().fillna(0).to_numpy()
        else:
            raise ValueError("errorbar_type must be 'std' or 'sem'")
    else:
        yerr = None

    x = mean_vals.index.to_numpy(dtype=int)
    y = mean_vals.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4 if show_errorbars else 0,
        edgecolor="black",
        alpha=0.85,
    )

    if show_counts:
        for xi, yi, n in zip(x, y, counts.to_numpy()):
            ax.text(
                xi,
                yi,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Coordination number")
    ax.set_ylabel(rf"Average $|\psi_{{{order_n}}}|$")
    ax.set_xticks(x)
    ax.set_ylim(0, max(1.0, np.nanmax(y) * 1.15))
    ax.grid(axis="y", alpha=0.3)

    if title is None:
        title = rf"Average $|\psi_{{{order_n}}}|$ per coordination number"

    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax