#%%

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import nd2
from pprint import pprint
import pandas as pd
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import networkx as nx
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.colors as mcolors

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Choose output folder
output_dir = Path(r"C:\Analysis_images\Hydrazine 010")
output_dir.mkdir(parents=True, exist_ok=True)


def fraction_limits_from_image(image_shape, um_per_px, x_frac=(0, 1), y_frac=(0, 1)):
    """
    Convert image fractions to x/y limits in microns.

    x_frac=(0, 0.5) means left half.
    y_frac=(0, 0.5) means top half.
    """
    H, W = image_shape

    W_um = W * um_per_px
    H_um = H * um_per_px

    x0 = x_frac[0] * W_um
    x1 = x_frac[1] * W_um

    y0 = y_frac[0] * H_um
    y1 = y_frac[1] * H_um

    return (x0, x1), (y0, y1)

def order_parameter(df, cutoff_distance, um_per_px  = 0.32, order_n=6):

    coords = df[['x_col', 'y_col']].values * um_per_px

    # Calculate all pairwise distances using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Mask to exclude self distances and those above cutoff
    mask = (distances < cutoff_distance) & (distances != 0)

    # Calculate angles using broadcasting and masked distances
    dy, dx = diff[:, :, 1], diff[:, :, 0]
    angles = np.arctan2(dy, dx)
    masked_angles = np.where(mask, angles, np.nan)

    # Calculate psi values
    exp_angles = np.exp(order_n * 1j * masked_angles)
    psi_values = np.nanmean(exp_angles, axis=1)
    psi_values = np.abs(np.nan_to_num(psi_values))  # Replace NaN with 0 and take absolute value

    # Ensure all arrays have the same length
    if len(psi_values) != len(coords):
        psi_values = psi_values[:len(coords)]

    # Extract x and y coordinates
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    # Set up the figure and axis with equal aspect ratio
    fig, ax1 = plt.subplots()
    ax1.set_aspect('equal')

    # Scatter plot for the data using psi values as color
    sc1 = ax1.scatter(x_coords, y_coords, c=psi_values, s=1, cmap=plt.cm.OrRd)

    # Custom limits for x and y axis
    # plt.xlim(150, 1150)
    # plt.ylim(-2000, -1000)

    # # Custom x and y ticks
    # plt.xticks([150, 650, 1150], [0, 50, 100], fontsize=14)
    # plt.yticks([-2000, -1500, -1000], [0, 50, 100], fontsize=14)

    # Adjust tick parameters
    plt.tick_params(axis='both', direction='in', length=10, width=2)

    # Add a colorbar
    cb = plt.colorbar(sc1, pad=0.02)
    cb.outline.set_linewidth(2)
    cb.ax.tick_params(length=8)
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.tick_params(direction='in')
    cb.set_label(label=rf'$\psi_{{{order_n}}}$', size=16)
    cb.ax.tick_params(labelsize=16)
    cb.ax.yaxis.set_label_coords(3.5, 0.5)

    # Set axis labels and padding
    plt.xlabel('X ($\mu$m)', size=14)
    plt.ylabel('Y ($\mu$m)', size=14)
    plt.gca().yaxis.labelpad = -10
    plt.gca().xaxis.labelpad = -2
    plt.gca().invert_yaxis()
    # Set spine thickness
    spine_thickness = 2  
    for spine in plt.gca().spines.values():
        spine.set_linewidth(spine_thickness)

    # Save the figure with high DPI and tight bounding box
    #plt.savefig('psi_6_GR_all_nn_crop_1.pdf', dpi=600, bbox_inches='tight')

    # Display the plot
    plt.show()



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

def compute_coord_num_from_df(det_df, cutoff):
    particles_xy = det_df[["x_col", "y_col"]].to_numpy()

    if len(particles_xy) == 0:
        return np.nan, np.array([])

    tree = cKDTree(particles_xy)
    coord_num = tree.query_ball_point(particles_xy, r=cutoff, return_length=True) - 1
    mean_coordination = np.mean(coord_num)

    return mean_coordination, coord_num

def compute_coord_num(particles_csv, cutoff, min_persist=2):
    particles = pd.read_csv(particles_csv)

    particles_xy = particles[['x_col', 'y_col']].values

    if len(particles_xy) == 0:
        return np.nan, np.array([])

    particles_tree = cKDTree(particles_xy)

    coord_num = particles_tree.query_ball_point(particles_xy, r=cutoff, return_length=True) - 1
    mean_coordination = np.mean(coord_num)

    return mean_coordination, coord_num


def radial_profile_from_tree_single_particle(
    det_df,
    particle_index,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.32,
    dr_um=2.0,
    r_max_um=100.0,
    image_shape=None,
    boundary_safe=True,
):
    """
    Radial density profile around one particle using one pre-built cKDTree.

    Returns density in particles / µm² and normalized g(r).
    """

    coords_px = det_df[[x_col, y_col]].to_numpy()
    coords_um = coords_px * um_per_px

    tree = cKDTree(coords_um)

    center = coords_um[particle_index]
    center_x, center_y = center

    # avoid edge effects by limiting r_max to nearest image boundary
    if image_shape is not None and boundary_safe:
        H, W = image_shape
        W_um = W * um_per_px
        H_um = H * um_per_px

        dist_to_edge = min(
            center_x,
            W_um - center_x,
            center_y,
            H_um - center_y,
        )

        r_max_um = min(r_max_um, dist_to_edge)

    # radii / cutoffs
    r_edges = np.arange(0, r_max_um + dr_um, dr_um)

    cumulative_counts = []

    for r in r_edges[1:]:
        neighbours = tree.query_ball_point(center, r=r)

        # remove the selected particle itself
        count = len(neighbours) - 1

        cumulative_counts.append(count)

    cumulative_counts = np.array(cumulative_counts)

    # convert cumulative counts into shell counts
    shell_counts = np.diff(np.r_[0, cumulative_counts])

    r_inner = r_edges[:-1]
    r_outer = r_edges[1:]
    r_mid = 0.5 * (r_inner + r_outer)

    shell_area = np.pi * (r_outer**2 - r_inner**2)

    radial_density = shell_counts / shell_area

    # average density of full image, for g(r)-like normalization
    if image_shape is not None:
        H, W = image_shape
        total_area_um2 = (H * um_per_px) * (W * um_per_px)
        average_density = len(det_df) / total_area_um2
        g_r = radial_density / average_density
    else:
        average_density = np.nan
        g_r = np.full_like(radial_density, np.nan)

    profile_df = pd.DataFrame({
        "r_um": r_mid,
        "r_inner_um": r_inner,
        "r_outer_um": r_outer,
        "cumulative_count": cumulative_counts,
        "shell_count": shell_counts,
        "shell_area_um2": shell_area,
        "density": radial_density,
        "g_r": g_r,
    })

    return profile_df
#%%
"""
    Load one 2D frame from an ND2 and convert it to uint8.
    """

nd2_path = r"c:\Users\Public\Hydrazine 010.nd2"

arr8 = load_nd2_frame_uint8(
    nd2_path=nd2_path,
    t=-1,          # use -1 if your detections were made on the last frame
    c=0,
    z=None,
    p=0,
    normalize_mode="percentile",
    p_low=1,
    p_high=99,
)
Image.fromarray(arr8).save(output_dir / "Hydrazine_010_last_frame.png")

#%%
corrected_csv = r"c:\particle_csv_files\Hydrazine 010_corrected_final.csv"

det_df = pd.read_csv(corrected_csv)

particle_id = 1421

valid_df = det_df[
    np.isfinite(det_df["x_col"])
    & np.isfinite(det_df["y_col"])
].reset_index(drop=True)


particle_index = valid_df.index[valid_df["particle_id"] == particle_id][0]

profile_df = radial_profile_from_tree_single_particle(
    valid_df,
    particle_index=particle_index,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    dr_um=0.25,
    r_max_um=10.0,
    image_shape=arr8.shape,
    boundary_safe=True,
)

plt.figure(figsize=(6, 4))
plt.plot(profile_df["r_um"], profile_df["g_r"], marker="o")
plt.xlabel("r (µm)")
plt.ylabel("g(r)")
plt.title(f"Radial profile around particle {particle_id}")
plt.grid(True, alpha=0.3)
plt.show()

#%%

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(profile_df["r_um"], profile_df["g_r"], marker="o")

ax.set_xlabel("r (µm)")
ax.set_ylabel("g(r)")
ax.set_title(f"Radial profile around particle {particle_id}")
ax.grid(True, alpha=0.3)

plt.tight_layout()

fig.savefig(
    output_dir / f"radial_profile_particle_{particle_id}.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    output_dir / f"radial_profile_particle_{particle_id}.pdf",
    bbox_inches="tight",
)

plt.show()
# %%
order_parameter(valid_df, cutoff_distance=2.0, um_per_px=0.06, order_n=3)

#%%
def compute_order_parameter_kdtree(
    df,
    cutoff_um,
    um_per_px=0.06,
    order_n=6,
    x_col="x_col",
    y_col="y_col",
):
    """
    Compute local bond-orientational order parameter psi_n using cKDTree.

    Returns a copy of df with:
    - psi_n
    - coord_num
    """

    df = df.copy()

    valid = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    df = df[valid].reset_index(drop=True)

    coords_px = df[[x_col, y_col]].to_numpy(dtype=float)
    coords_um = coords_px * um_per_px

    tree = cKDTree(coords_um)
    neighbours_all = tree.query_ball_point(coords_um, r=cutoff_um)

    psi_values = np.zeros(len(coords_um), dtype=float)
    coord_nums = np.zeros(len(coords_um), dtype=int)

    for i, neighbours in enumerate(neighbours_all):
        # remove self
        neighbours = [j for j in neighbours if j != i]

        coord_nums[i] = len(neighbours)

        if len(neighbours) == 0:
            psi_values[i] = 0.0
            continue

        dx = coords_um[neighbours, 0] - coords_um[i, 0]
        dy = coords_um[neighbours, 1] - coords_um[i, 1]

        angles = np.arctan2(dy, dx)

        psi_complex = np.mean(np.exp(1j * order_n * angles))
        psi_values[i] = np.abs(psi_complex)

    psi_col = f"psi_{order_n}"
    df[psi_col] = psi_values
    df["coord_num"] = coord_nums

    return df


def plot_order_parameter(
    order_df,
    order_n=6,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    s=1,
    n_bins=10,
):
    """
    Plot local bond order with a discrete colorbar.
    """

    psi_col = f"psi_{order_n}"

    x_um = order_df[x_col].to_numpy(dtype=float) * um_per_px
    y_um = order_df[y_col].to_numpy(dtype=float) * um_per_px
    psi_values = order_df[psi_col].to_numpy(dtype=float)

    # Discrete color bins from 0 to 1
    bounds = np.linspace(0, 1, n_bins + 1)

    # Discrete version of OrRd
    cmap = plt.cm.get_cmap("OrRd", n_bins)

    # Norm maps psi values into discrete bins
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

    cb.set_label(label=rf"$\psi_{{{order_n}}}$", size=16)
    cb.ax.tick_params(labelsize=12)

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.invert_yaxis()

    ax.set_title(
        rf"Local bond order $\psi_{{{order_n}}}$" + "\n"
        + f"mean {psi_col} = {order_df[psi_col].mean():.3f}, "
        + f"mean coord = {order_df['coord_num'].mean():.2f}"
    )

    plt.tight_layout()
    plt.show()

#%%
def plot_order_parameter(
    order_df,
    order_n=6,
    cutoff_um=1.0,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.65,
    s=1,
    n_bins=10,
    image_shape=None,
    x_frac=(0, 1),
    y_frac=(0, 1),
    output_path=None,
):
    """
    Plot local bond order with optional crop based on image fractions.

    Example:
    x_frac=(0, 0.5)  -> left half
    y_frac=(0, 0.5)  -> top half
    """

    psi_col = f"psi_{order_n}"

    if psi_col not in order_df.columns:
        raise ValueError(f"{psi_col} not found in order_df.")

    x_um = order_df[x_col].to_numpy(dtype=float) * um_per_px
    y_um = order_df[y_col].to_numpy(dtype=float) * um_per_px
    psi_values = order_df[psi_col].to_numpy(dtype=float)

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

    cb.set_label(label=rf"$\psi_{{{order_n}}}$", size=16)
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

        # This keeps image-style coordinates:
        # y = 0 at the top, larger y lower down.
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.invert_yaxis()

    ax.set_title(
        rf"Local bond order $\psi_{{{order_n}}}$" + "\n"
        + f"mean {psi_col} = {order_df[psi_col].mean():.3f}, "
        + f"cutoff = {cutoff_um} µm",
        # + f"mean coord = {order_df['coord_num'].mean():.2f}"
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
# %%

#%%

order_n = 6
cutoff_um=0.9

order_df = compute_order_parameter_kdtree(
    valid_df,
    cutoff_um=cutoff_um, #measured in NIS
    um_per_px=0.65,
    order_n=order_n,
    x_col="x_col",
    y_col="y_col",
)

# plot_order_parameter(
#     order_df,
#     order_n=6,
#     x_col="x_col",
#     y_col="y_col",
#     um_per_px=0.06,
#     s=1,
# )
#%%
plot_order_parameter(
    order_df,
    order_n=order_n,
    cutoff_um=cutoff_um,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.65,
    s=2,
    image_shape=arr8.shape,
    x_frac=(0.0, 0.8),
    y_frac=(0.1, 0.8),
    output_path=output_dir / f"Hydrazine_010_psi{order_n}_zoom_cutoff{cutoff_um}.png",
)



#%%
"""
HTML for delecting region where I take averages:


"""
#%%
import base64
import io
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


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

#%%


#%%
roi_html = make_particle_roi_selection_html(
    image=arr8,
    det_df=valid_df,
    output_html=output_dir / "Hydrazine_010_select_roi.html",
    output_csv_name="Hydrazine_010_selected_particles.csv",
    output_roi_csv_name="Hydrazine_010_roi_vertices.csv",
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    point_radius=2.0,
)

#%%

selected_csv = r"c:\Analysis_images\Hydrazine 010\Hydrazine_010_selected_particles (1).csv"

selected_df = pd.read_csv(selected_csv)

print("Selected particles:", len(selected_df))
selected_df.head()

#%%

#%%
def average_radial_profile_from_selected_particles(
    selected_df,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    dr_um=0.25,
    r_max_um=100.0,
):
    """
    Average radial density / g(r) over all particles in selected_df.

    Simple version:
    - centers = all selected particles
    - neighbours = selected particles
    - average density estimated from selected-particle bounding box

    For a quick comparison between regions, this is usually fine.
    """

    coords_px = selected_df[[x_col, y_col]].to_numpy(dtype=float)
    coords_um = coords_px * um_per_px

    if len(coords_um) < 2:
        raise ValueError("Need at least 2 particles for g(r).")

    tree = cKDTree(coords_um)

    r_edges = np.arange(0, r_max_um + dr_um, dr_um)
    r_inner = r_edges[:-1]
    r_outer = r_edges[1:]
    r_mid = 0.5 * (r_inner + r_outer)

    all_shell_counts = []

    for center in coords_um:
        cumulative_counts = []

        for r in r_edges[1:]:
            neighbours = tree.query_ball_point(center, r=r)
            count = len(neighbours) - 1
            cumulative_counts.append(count)

        cumulative_counts = np.array(cumulative_counts)
        shell_counts = np.diff(np.r_[0, cumulative_counts])

        all_shell_counts.append(shell_counts)

    all_shell_counts = np.array(all_shell_counts)

    mean_shell_count = np.mean(all_shell_counts, axis=0)
    std_shell_count = np.std(all_shell_counts, axis=0)

    shell_area = np.pi * (r_outer**2 - r_inner**2)
    radial_density = mean_shell_count / shell_area

    # approximate selected-region area using bounding box
    x_min, y_min = coords_um.min(axis=0)
    x_max, y_max = coords_um.max(axis=0)

    area_um2 = (x_max - x_min) * (y_max - y_min)
    average_density = len(coords_um) / area_um2

    g_r = radial_density / average_density

    profile_df = pd.DataFrame({
        "r_um": r_mid,
        "r_inner_um": r_inner,
        "r_outer_um": r_outer,
        "mean_shell_count": mean_shell_count,
        "std_shell_count": std_shell_count,
        "shell_area_um2": shell_area,
        "density": radial_density,
        "g_r": g_r,
        "n_center_particles": len(coords_um),
        "average_density": average_density,
    })

    return profile_df



#%%
triangle_side_um = 10.8
rin_um = triangle_side_um * np.sqrt(3) / 6
spacing_um = 2 * rin_um

#%%

from scipy.signal import find_peaks
import numpy as np

um_per_px=0.65
dr_um=0.5
r_max_um=200.0
selected_df = pd.read_csv(r"c:\Analysis_images\Hydrazine 010\Hydrazine_010_final_ROI(2).csv")
avg_profile_df = average_radial_profile_from_selected_particles(
    selected_df,
    x_col="x",
    y_col="y",
    um_per_px=um_per_px,
    dr_um=dr_um,
    r_max_um=r_max_um,
)
# x-axis used in your plot: r / 2a
x = avg_profile_df["r_um"].to_numpy() / spacing_um
y = avg_profile_df["g_r"].to_numpy()

# find local peaks
peaks, props = find_peaks(
    y,
    prominence=0.1,   # increase if it detects too many small wiggles
    distance=2        # minimum number of bins between peaks
)

# sort peaks by peak height, highest first
top2_peaks = peaks[np.argsort(y[peaks])[-2:]][::-1]

print("Two highest peaks:")
for i, peak_idx in enumerate(top2_peaks, start=1):
    print(
        f"Peak {i}: "
        f"x = {x[peak_idx]:.3f} in r/2a, "
        f"r = {avg_profile_df['r_um'].iloc[peak_idx]:.3f} µm, "
        f"g(r) = {y[peak_idx]:.3f}"
    )



fig, ax = plt.subplots(figsize=(6, 4))

ax.plot((avg_profile_df["r_um"])/spacing_um, avg_profile_df["g_r"])

# vertical dotted lines at n * spacing_um
# n_values = np.arange(1, int(np.floor((r_max_um/spacing_um) / spacing_um)) + 1)
n_values = np.arange(1, (r_max_um/spacing_um) + 1)

for n in n_values:
    xline = n * spacing_um
    ax.axvline(xline, linestyle=":", linewidth=1)



for peak_idx in top2_peaks:
    ax.axvline(x[peak_idx], linestyle="--", linewidth=1)

ax.set_xlabel("r/2a")
ax.set_ylabel("g(r)")
ax.set_title(f"Average g(r), selected region, N = {len(selected_df)}\n"
             +f"dr = {dr_um} µm"+ f"max radius = {r_max_um} µm")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, r_max_um/spacing_um)
plt.tight_layout()

fig.savefig(
    output_dir / f"Hydrazine_010_average_gr_selected_region_dr{dr_um}_rmax{r_max_um}.png",
    dpi=300,
    bbox_inches="tight",
)


plt.show()

#%%
"""
Order parameter selected region:"""


order_n = 5
cutoff_um=0.7

order_df = compute_order_parameter_kdtree(
    selected_df,
    cutoff_um=cutoff_um, #measured in NIS
    um_per_px=0.06,
    order_n=order_n,
    x_col="x_col",
    y_col="y_col",
)

# plot_order_parameter(
#     order_df,
#     order_n=6,
#     x_col="x_col",
#     y_col="y_col",
#     um_per_px=0.06,
#     s=1,
# )
#%%
plot_order_parameter(
    order_df,
    order_n=order_n,
    cutoff_um=cutoff_um,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    s=2,
    image_shape=arr8.shape,
    x_frac=(0.0, 0.8),
    y_frac=(0.1, 0.8),
    output_path=output_dir / f"Hydrazine_010_psi{order_n}_zoom_cutoff{cutoff_um}.png",
)



#%%
#%%
def compute_coordination_number_kdtree(
    df,
    cutoff_um,
    um_per_px=0.06,
    x_col="x_col",
    y_col="y_col",
):
    """
    Compute coordination number using a cutoff in microns.

    Returns a copy of df with:
    - coord_num
    """

    df = df.copy()

    valid = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    df = df[valid].reset_index(drop=True)

    coords_px = df[[x_col, y_col]].to_numpy(dtype=float)
    coords_um = coords_px * um_per_px

    tree = cKDTree(coords_um)

    coord_num = tree.query_ball_point(
        coords_um,
        r=cutoff_um,
        return_length=True,
    ) - 1

    df["coord_num"] = coord_num

    return df

#%%
def plot_coordination_number_map(
    coord_df,
    cutoff_um,
    x_col="x_col",
    y_col="y_col",
    um_per_px=0.06,
    image_shape=None,
    x_frac=(0, 1),
    y_frac=(0, 1),
    s=8,
    max_coord=None,
    output_path=None,
):
    """
    Plot particles colored by coordination number.
    """

    df = coord_df.copy()

    x_um = df[x_col].to_numpy(dtype=float) * um_per_px
    y_um = df[y_col].to_numpy(dtype=float) * um_per_px
    coord = df["coord_num"].to_numpy(dtype=int)

    if max_coord is None:
        max_coord = max(1, int(np.nanmax(coord)))

    bounds = np.arange(-0.5, max_coord + 1.5, 1)
    cmap = plt.cm.get_cmap("YlOrRd", max_coord + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    sc = ax.scatter(
        x_um,
        y_um,
        c=coord,
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
        ticks=np.arange(0, max_coord + 1),
    )

    cb.set_label("Coordination number", size=14)
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
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.invert_yaxis()

    ax.set_title(
        f"Coordination number\n"
        f"cutoff = {cutoff_um} µm, mean coord = {coord.mean():.2f}"
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

#%%
def plot_coordination_number_histogram(
    coord_df,
    cutoff_um,
    max_coord=None,
    output_path=None,
):
    """
    Plot normalized frequency of coordination numbers.
    """

    coord = coord_df["coord_num"].to_numpy(dtype=int)

    if max_coord is None:
        max_coord = max(1, int(np.nanmax(coord)))

    bins = np.arange(-0.5, max_coord + 1.5, 1)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(
        coord,
        bins=bins,
        weights=np.ones_like(coord) / len(coord),
        edgecolor="black",
    )

    ax.set_xlabel("Coordination Number")
    ax.set_ylabel("Normalized Frequency")
    ax.set_xticks(np.arange(0, max_coord + 1))
    ax.set_title(
        f"Coordination number distribution\n"
        f"cutoff = {cutoff_um} µm, N = {len(coord_df)}"
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
#%%
#%%
um_per_px = 0.06
coord_cutoff_um = 0.9   # choose based on g(r), usually first minimum after first peak

coord_df = compute_coordination_number_kdtree(
    selected_df,
    cutoff_um=coord_cutoff_um,
    um_per_px=um_per_px,
    x_col="x_col",
    y_col="y_col",
)

print("Mean coordination:", coord_df["coord_num"].mean())
print(coord_df["coord_num"].value_counts().sort_index())

#%%
plot_coordination_number_map(
    coord_df,
    cutoff_um=coord_cutoff_um,
    x_col="x_col",
    y_col="y_col",
    um_per_px=um_per_px,
    image_shape=arr8.shape,
    x_frac=(0.0, 0.8),
    y_frac=(0.1, 0.8),
    s=10,
    max_coord=6,
    output_path=output_dir / f"Hydrazine_010_coord_map_cutoff{coord_cutoff_um}.png",
)
#%%
plot_coordination_number_histogram(
    coord_df,
    cutoff_um=coord_cutoff_um,
    max_coord=6,
    output_path=output_dir / f"Hydrazine_010_coord_hist_cutoff{coord_cutoff_um}.png",
)

#%%
# # %%
# import json
# import webbrowser
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors


# def make_order_parameter_html(
#     order_df,
#     output_html="order_parameter_zoom.html",
#     order_n=3,
#     x_col="x_col",
#     y_col="y_col",
#     um_per_px=0.06,
#     n_bins=10,
#     point_radius=1.6,
#     title=None,
# ):
#     """
#     Create interactive HTML viewer for bond-orientational order.

#     Features:
#     - zoom with mouse wheel
#     - pan by dragging
#     - reset view button
#     - hover pixel/position + nearest particle info
#     - discrete colorbar for psi_n
#     """

#     psi_col = f"psi_{order_n}"

#     if psi_col not in order_df.columns:
#         raise ValueError(f"{psi_col} not found in dataframe.")

#     df = order_df.copy()

#     valid = (
#         np.isfinite(df[x_col])
#         & np.isfinite(df[y_col])
#         & np.isfinite(df[psi_col])
#     )
#     df = df[valid].reset_index(drop=True)

#     x_um = df[x_col].to_numpy(dtype=float) * um_per_px
#     y_um = df[y_col].to_numpy(dtype=float) * um_per_px
#     psi = df[psi_col].to_numpy(dtype=float)

#     if "particle_id" in df.columns:
#         particle_ids = df["particle_id"].to_numpy()
#     else:
#         particle_ids = np.arange(len(df))

#     if "coord_num" in df.columns:
#         coord_nums = df["coord_num"].to_numpy()
#     else:
#         coord_nums = np.full(len(df), np.nan)

#     psi = np.clip(psi, 0, 1)

#     # Discrete color bins
#     bin_edges = np.linspace(0, 1, n_bins + 1)
#     cmap = plt.cm.OrRd
#     colors = [
#         mcolors.to_hex(cmap((i + 0.5) / n_bins))
#         for i in range(n_bins)
#     ]

#     points = []
#     for i in range(len(df)):
#         bin_index = np.searchsorted(bin_edges, psi[i], side="right") - 1
#         bin_index = int(np.clip(bin_index, 0, n_bins - 1))

#         points.append({
#             "id": int(particle_ids[i]),
#             "x": float(x_um[i]),
#             "y": float(y_um[i]),
#             "psi": float(psi[i]),
#             "coord": None if np.isnan(coord_nums[i]) else float(coord_nums[i]),
#             "bin": bin_index,
#         })

#     points_json = json.dumps(points, separators=(",", ":"))
#     colors_json = json.dumps(colors)
#     bin_edges_json = json.dumps([float(v) for v in bin_edges])

#     x_min, x_max = float(np.min(x_um)), float(np.max(x_um))
#     y_min, y_max = float(np.min(y_um)), float(np.max(y_um))

#     pad_x = 0.04 * (x_max - x_min)
#     pad_y = 0.04 * (y_max - y_min)

#     x_min -= pad_x
#     x_max += pad_x
#     y_min -= pad_y
#     y_max += pad_y

#     mean_psi = float(np.mean(psi))
#     mean_coord = float(np.nanmean(coord_nums)) if np.isfinite(coord_nums).any() else np.nan

#     if title is None:
#         title = (
#             f"Local bond order ψ{order_n} | "
#             f"mean ψ{order_n} = {mean_psi:.3f}, mean coord = {mean_coord:.2f}"
#         )

#     html = f"""
# <!DOCTYPE html>
# <html>
# <head>
# <meta charset="UTF-8">
# <title>Order parameter viewer</title>

# <style>
# body {{
#     font-family: Arial, sans-serif;
#     background: #111;
#     color: white;
#     margin: 20px;
# }}

# #layout {{
#     display: flex;
#     gap: 20px;
#     align-items: flex-start;
# }}

# canvas {{
#     background: white;
#     border: 2px solid #ccc;
# }}

# button {{
#     margin: 5px 8px 5px 0;
#     padding: 8px 14px;
#     font-size: 14px;
# }}

# label {{
#     margin-right: 18px;
# }}

# #info {{
#     margin-top: 12px;
#     font-size: 14px;
#     line-height: 1.5;
# }}

# .hint {{
#     color: #bbb;
# }}
# </style>
# </head>

# <body>

# <h2>{title}</h2>

# <div>
#     <button onclick="resetView()">Reset view</button>
#     <button onclick="zoomIn()">Zoom in</button>
#     <button onclick="zoomOut()">Zoom out</button>

#     <label>
#         <input type="checkbox" id="showIds" onchange="draw()"> Show particle IDs
#     </label>

#     <label>
#         <input type="checkbox" id="showAxes" checked onchange="draw()"> Show axes
#     </label>
# </div>

# <p class="hint">
# Mouse wheel = zoom<br>
# Drag = pan<br>
# Hover = nearest particle info
# </p>

# <div id="layout">
#     <canvas id="canvas"></canvas>
#     <canvas id="colorbar"></canvas>
# </div>

# <div id="info">
#     Number of particles: <span id="nParticles"></span><br>
#     Zoom: <span id="zoomValue">1.00</span>x<br>
#     Hover position: x = <span id="hoverX">-</span> µm,
#     y = <span id="hoverY">-</span> µm<br>
#     Nearest particle: <span id="nearestInfo">-</span>
# </div>

# <script>
# const points = {points_json};
# const colors = {colors_json};
# const binEdges = {bin_edges_json};

# const xMin0 = {x_min};
# const xMax0 = {x_max};
# const yMin0 = {y_min};
# const yMax0 = {y_max};

# const orderN = {order_n};
# const pointRadiusBase = {point_radius};

# const canvas = document.getElementById("canvas");
# const ctx = canvas.getContext("2d");

# const colorbar = document.getElementById("colorbar");
# const cctx = colorbar.getContext("2d");

# canvas.width = Math.min(window.innerWidth * 0.72, 1100);
# canvas.height = Math.min(window.innerHeight * 0.72, 800);

# colorbar.width = 90;
# colorbar.height = canvas.height;

# document.getElementById("nParticles").innerText = points.length;

# let scale = 1.0;
# let offsetX = 0;
# let offsetY = 0;

# let baseScale = 1.0;
# let plotLeft = 70;
# let plotRight = canvas.width - 20;
# let plotTop = 20;
# let plotBottom = canvas.height - 60;

# let isDragging = false;
# let dragStartX = 0;
# let dragStartY = 0;

# let hoverScreenX = null;
# let hoverScreenY = null;

# function resetView() {{
#     const dataW = xMax0 - xMin0;
#     const dataH = yMax0 - yMin0;

#     const plotW = plotRight - plotLeft;
#     const plotH = plotBottom - plotTop;

#     baseScale = Math.min(plotW / dataW, plotH / dataH);

#     scale = baseScale;

#     offsetX = plotLeft - xMin0 * scale;
#     offsetY = plotTop - yMin0 * scale;

#     draw();
# }}

# function dataToScreen(x, y) {{
#     return {{
#         x: offsetX + x * scale,
#         y: offsetY + y * scale
#     }};
# }}

# function screenToData(sx, sy) {{
#     return {{
#         x: (sx - offsetX) / scale,
#         y: (sy - offsetY) / scale
#     }};
# }}

# function drawAxes() {{
#     const showAxes = document.getElementById("showAxes").checked;
#     if (!showAxes) return;

#     ctx.strokeStyle = "black";
#     ctx.fillStyle = "black";
#     ctx.lineWidth = 1;
#     ctx.font = "12px Arial";

#     // axis frame
#     ctx.strokeRect(plotLeft, plotTop, plotRight - plotLeft, plotBottom - plotTop);

#     const nTicks = 5;

#     // x ticks
#     for (let i = 0; i <= nTicks; i++) {{
#         const x = xMin0 + i * (xMax0 - xMin0) / nTicks;
#         const p = dataToScreen(x, yMax0);

#         ctx.beginPath();
#         ctx.moveTo(p.x, plotBottom);
#         ctx.lineTo(p.x, plotBottom + 5);
#         ctx.stroke();

#         ctx.fillText(x.toFixed(0), p.x - 10, plotBottom + 20);
#     }}

#     // y ticks
#     for (let i = 0; i <= nTicks; i++) {{
#         const y = yMin0 + i * (yMax0 - yMin0) / nTicks;
#         const p = dataToScreen(xMin0, y);

#         ctx.beginPath();
#         ctx.moveTo(plotLeft - 5, p.y);
#         ctx.lineTo(plotLeft, p.y);
#         ctx.stroke();

#         ctx.fillText(y.toFixed(0), plotLeft - 42, p.y + 4);
#     }}

#     ctx.fillText("x (µm)", 0.5 * (plotLeft + plotRight) - 20, canvas.height - 18);

#     ctx.save();
#     ctx.translate(18, 0.5 * (plotTop + plotBottom) + 20);
#     ctx.rotate(-Math.PI / 2);
#     ctx.fillText("y (µm)", 0, 0);
#     ctx.restore();
# }}

# function drawPoints() {{
#     const showIds = document.getElementById("showIds").checked;

#     for (const pnt of points) {{
#         const p = dataToScreen(pnt.x, pnt.y);

#         ctx.beginPath();
#         ctx.arc(p.x, p.y, Math.max(pointRadiusBase, 1.2), 0, 2 * Math.PI);
#         ctx.fillStyle = colors[pnt.bin];
#         ctx.fill();

#         if (showIds) {{
#             ctx.fillStyle = "black";
#             ctx.font = "10px Arial";
#             ctx.fillText(pnt.id, p.x + 4, p.y - 4);
#         }}
#     }}
# }}

# function drawColorbar() {{
#     cctx.clearRect(0, 0, colorbar.width, colorbar.height);

#     cctx.fillStyle = "#111";
#     cctx.fillRect(0, 0, colorbar.width, colorbar.height);

#     const barX = 20;
#     const barY = 40;
#     const barW = 28;
#     const barH = colorbar.height - 100;

#     const n = colors.length;
#     const binH = barH / n;

#     for (let i = 0; i < n; i++) {{
#         // draw high values at top
#         const drawI = n - 1 - i;
#         cctx.fillStyle = colors[drawI];
#         cctx.fillRect(barX, barY + i * binH, barW, binH + 1);

#         cctx.strokeStyle = "black";
#         cctx.lineWidth = 0.5;
#         cctx.strokeRect(barX, barY + i * binH, barW, binH);
#     }}

#     cctx.strokeStyle = "white";
#     cctx.lineWidth = 1.5;
#     cctx.strokeRect(barX, barY, barW, barH);

#     cctx.fillStyle = "white";
#     cctx.font = "12px Arial";
#     cctx.fillText("1.0", barX + 36, barY + 4);
#     cctx.fillText("0.0", barX + 36, barY + barH + 4);

#     cctx.save();
#     cctx.translate(78, barY + barH / 2 + 20);
#     cctx.rotate(-Math.PI / 2);
#     cctx.font = "16px Arial";
#     cctx.fillText("ψ" + orderN, 0, 0);
#     cctx.restore();
# }}

# function findNearest(screenX, screenY, maxDist=18) {{
#     let best = null;
#     let bestD2 = maxDist * maxDist;

#     for (const pnt of points) {{
#         const p = dataToScreen(pnt.x, pnt.y);
#         const dx = p.x - screenX;
#         const dy = p.y - screenY;
#         const d2 = dx * dx + dy * dy;

#         if (d2 < bestD2) {{
#             bestD2 = d2;
#             best = pnt;
#         }}
#     }}

#     return best;
# }}

# function updateHoverInfo() {{
#     if (hoverScreenX === null || hoverScreenY === null) {{
#         document.getElementById("hoverX").innerText = "-";
#         document.getElementById("hoverY").innerText = "-";
#         document.getElementById("nearestInfo").innerText = "-";
#         return;
#     }}

#     const d = screenToData(hoverScreenX, hoverScreenY);

#     document.getElementById("hoverX").innerText = d.x.toFixed(2);
#     document.getElementById("hoverY").innerText = d.y.toFixed(2);

#     const nearest = findNearest(hoverScreenX, hoverScreenY);

#     if (nearest) {{
#         document.getElementById("nearestInfo").innerText =
#             "id=" + nearest.id +
#             ", ψ" + orderN + "=" + nearest.psi.toFixed(3) +
#             ", coord=" + nearest.coord +
#             ", x=" + nearest.x.toFixed(2) +
#             ", y=" + nearest.y.toFixed(2);
#     }} else {{
#         document.getElementById("nearestInfo").innerText = "-";
#     }}
# }}

# function draw() {{
#     ctx.clearRect(0, 0, canvas.width, canvas.height);

#     ctx.fillStyle = "white";
#     ctx.fillRect(0, 0, canvas.width, canvas.height);

#     drawPoints();
#     drawAxes();
#     drawColorbar();

#     document.getElementById("zoomValue").innerText = (scale / baseScale).toFixed(2);
#     updateHoverInfo();
# }}

# function zoomAt(screenX, screenY, factor) {{
#     const before = screenToData(screenX, screenY);

#     scale = Math.min(Math.max(scale * factor, baseScale * 0.2), baseScale * 80);

#     offsetX = screenX - before.x * scale;
#     offsetY = screenY - before.y * scale;

#     draw();
# }}

# function zoomIn() {{
#     zoomAt(canvas.width / 2, canvas.height / 2, 1.25);
# }}

# function zoomOut() {{
#     zoomAt(canvas.width / 2, canvas.height / 2, 0.8);
# }}

# canvas.addEventListener("wheel", function(event) {{
#     event.preventDefault();

#     const rect = canvas.getBoundingClientRect();
#     const mx = event.clientX - rect.left;
#     const my = event.clientY - rect.top;

#     const factor = event.deltaY < 0 ? 1.15 : 0.87;
#     zoomAt(mx, my, factor);
# }}, {{ passive: false }});

# canvas.addEventListener("mousedown", function(event) {{
#     isDragging = true;

#     const rect = canvas.getBoundingClientRect();
#     dragStartX = event.clientX - rect.left;
#     dragStartY = event.clientY - rect.top;
# }});

# canvas.addEventListener("mousemove", function(event) {{
#     const rect = canvas.getBoundingClientRect();
#     const mx = event.clientX - rect.left;
#     const my = event.clientY - rect.top;

#     hoverScreenX = mx;
#     hoverScreenY = my;

#     if (isDragging) {{
#         const dx = mx - dragStartX;
#         const dy = my - dragStartY;

#         offsetX += dx;
#         offsetY += dy;

#         dragStartX = mx;
#         dragStartY = my;
#     }}

#     draw();
# }});

# canvas.addEventListener("mouseup", function() {{
#     isDragging = false;
# }});

# canvas.addEventListener("mouseleave", function() {{
#     isDragging = false;
# }});

# resetView();
# </script>

# </body>
# </html>
# """

#     output_html = Path(output_html)
#     output_html.write_text(html, encoding="utf-8")

#     print("Created:", output_html.resolve())
#     webbrowser.open(output_html.resolve().as_uri())

#     return output_html.resolve()

# html_path = make_order_parameter_html(
#     order_df,
#     output_html="Hydrazine_010_psi3_zoom.html",
#     order_n=3,
#     x_col="x_col",
#     y_col="y_col",
#     um_per_px=0.06,
#     n_bins=10,          # discrete colorbar bins
#     point_radius=1.4,
# )
# # %%


# %%
