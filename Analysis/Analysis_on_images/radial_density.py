
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
from scipy.signal import find_peaks
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree, ConvexHull
from shapely.geometry import Point, Polygon
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.path import Path
#%%

from shapely.geometry import Point


def clipped_shell_area_inside_hull(
    hull_polygon,
    center,
    r_inner,
    r_outer,
    circle_resolution=64,
):
    """
    Calculate the area of one radial shell that lies inside the convex hull.

    Parameters
    ----------
    hull_polygon : shapely.geometry.Polygon
        Convex hull polygon of the selected cluster.

    center : array-like
        Particle center in microns, for example [x_um, y_um].

    r_inner : float
        Inner radius of the shell in microns.

    r_outer : float
        Outer radius of the shell in microns.

    circle_resolution : int
        Smoothness of the circle approximation.

    Returns
    -------
    clipped_area : float
        Area of the shell inside the hull, in µm².
    """

    center_point = Point(center[0], center[1])

    outer_disk = center_point.buffer(r_outer, resolution=circle_resolution)

    if r_inner == 0:
        annulus = outer_disk
    else:
        inner_disk = center_point.buffer(r_inner, resolution=circle_resolution)
        annulus = outer_disk.difference(inner_disk)

    clipped_annulus = annulus.intersection(hull_polygon)

    clipped_area = clipped_annulus.area

    return clipped_area

def full_shell_area(
    hull_polygon,
    center,
    r_inner,
    r_outer,
    circle_resolution=64,
):
    center_point = Point(center[0], center[1])

    outer_disk = center_point.buffer(r_outer, resolution=circle_resolution)

    if r_inner == 0:
        annulus = outer_disk
    else:
        inner_disk = center_point.buffer(r_inner, resolution=circle_resolution)
        annulus = outer_disk.difference(inner_disk)

    area = annulus.area

    return area



def average_radial_profile_from_selected_particles(
    selected_df,
    x_col="x",
    y_col="y",
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

    # -----------------------------
    # Convex hull of selected cluster
    # -----------------------------
    hull = ConvexHull(coords_um)
    hull_points = coords_um[hull.vertices]
    hull_polygon = Polygon(hull_points)

    hull_area_um2 = hull_polygon.area
    average_density = len(coords_um) / hull_area_um2


    if len(coords_um) < 2:
        raise ValueError("Need at least 2 particles for g(r).")


    tree = cKDTree(coords_um)

    r_edges = np.arange(0, r_max_um + dr_um, dr_um)
    r_inner = r_edges[:-1]
    r_outer = r_edges[1:]
    r_mid = 0.5 * (r_inner + r_outer)

    all_shell_counts = []
    radial_density_list = []
    

    for center in coords_um:
        cumulative_counts = []
        clipped_shell_areas = []

        for r in r_edges[1:]:
            neighbours = tree.query_ball_point(center, r=r)
            count = len(neighbours) - 1
            cumulative_counts.append(count)
            rout = r
            rin = r - dr_um
            clipped_area = clipped_shell_area_inside_hull(
            hull_polygon=hull_polygon,
            center=center,
            r_inner=rin,
            r_outer=rout,
            circle_resolution=64,
        )
            clipped_shell_areas.append(clipped_area)

        cumulative_counts = np.array(cumulative_counts)
        clipped_shell_areas = np.array(clipped_shell_areas)
        shell_counts = np.diff(np.r_[0, cumulative_counts])

        all_shell_counts.append(shell_counts)
        radial_density = shell_counts / clipped_shell_areas
        radial_density_list.append(radial_density)


    all_shell_counts = np.array(all_shell_counts)
    radial_density_list = np.array(radial_density_list)
    mean_radial_density = np.mean(radial_density_list, axis=0)

    g_r = mean_radial_density / average_density

    profile_df = pd.DataFrame({
        "r_um": r_mid,
        "r_inner_um": r_inner,
        "r_outer_um": r_outer,
        "density": mean_radial_density,
        "g_r": g_r,
        "n_center_particles": len(coords_um),
        "average_density": average_density,
    })

    return profile_df







def average_radial_profile_from_selected_particles_full_circle(
    selected_df,
    x_col="x",
    y_col="y",
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

    # -----------------------------
    # Convex hull of selected cluster
    # -----------------------------
    hull = ConvexHull(coords_um)
    hull_points = coords_um[hull.vertices]
    hull_polygon = Polygon(hull_points)

    hull_area_um2 = hull_polygon.area
    average_density = len(coords_um) / hull_area_um2


    if len(coords_um) < 2:
        raise ValueError("Need at least 2 particles for g(r).")


    tree = cKDTree(coords_um)

    r_edges = np.arange(0, r_max_um + dr_um, dr_um)
    r_inner = r_edges[:-1]
    r_outer = r_edges[1:]
    r_mid = 0.5 * (r_inner + r_outer)

    all_shell_counts = []
    radial_density_list = []
    

    for center in coords_um:
        cumulative_counts = []
        clipped_shell_areas = []

        for r in r_edges[1:]:
            neighbours = tree.query_ball_point(center, r=r)
            count = len(neighbours) - 1
            cumulative_counts.append(count)
            rout = r
            rin = r - dr_um
            area = full_shell_area(
            hull_polygon=hull_polygon,
            center=center,
            r_inner=rin,
            r_outer=rout,
            circle_resolution=64,
        )
            clipped_shell_areas.append(area)

        cumulative_counts = np.array(cumulative_counts)
        clipped_shell_areas = np.array(clipped_shell_areas)
        shell_counts = np.diff(np.r_[0, cumulative_counts])

        all_shell_counts.append(shell_counts)
        radial_density = shell_counts / clipped_shell_areas
        radial_density_list.append(radial_density)


    all_shell_counts = np.array(all_shell_counts)
    radial_density_list = np.array(radial_density_list)
    mean_radial_density = np.mean(radial_density_list, axis=0)

    g_r = mean_radial_density / average_density

    profile_df = pd.DataFrame({
        "r_um": r_mid,
        "r_inner_um": r_inner,
        "r_outer_um": r_outer,
        "density": mean_radial_density,
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


def plot_radial_density(        
    um_per_px=0.65,
    dr_um=0.5,
    r_max_um=200.0,
    selected_df = None,
    output_dir = None
    ):


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
    # n_values = np.arange(1, (r_max_um/spacing_um) + 1)
    n_values = np.arange(1, int(np.max(x))+1)

    # for n in n_values:
    #     xline = n 
    #     ax.axvline(xline, linestyle=":", linewidth=1)



    # for peak_idx in top2_peaks:
    #     ax.axvline(x[peak_idx], linestyle="--", linewidth=1)

    ax.set_xlabel("r/2a")
    ax.set_ylabel("g(r)")
    ax.set_title(f"Average g(r) with circles only in selected region")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max_um/spacing_um)
    plt.tight_layout()

    fig.savefig(
        output_dir / f"Hydrazine_010_average_gr_selected_region_dr{dr_um}_rmax{r_max_um}_reduced_area.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.show()




import numpy as np
import pandas as pd


def make_perfect_hexagonal_df(
    n_rows=40,
    n_cols=40,
    spacing_um=6.24,
    um_per_px=0.65,
    center=True,
):
    """
    Make a dataframe of perfect hexagonal packing centers.

    The output has x and y in pixels, so it can be used directly in your
    average_radial_profile_from_selected_particles function.

    Parameters
    ----------
    n_rows, n_cols : int
        Number of lattice rows and columns.

    spacing_um : float
        Nearest-neighbour center-to-center distance in microns.
        For your code this is probably:
            spacing_um = 2 * rin_um

    um_per_px : float
        Pixel size in microns per pixel.

    center : bool
        If True, shifts the lattice so the minimum x and y are near zero.

    Returns
    -------
    hex_df : pandas.DataFrame
        DataFrame with columns x, y.
    """

    points_um = []

    dy_um = spacing_um * np.sqrt(3) / 2

    for row in range(n_rows):
        for col in range(n_cols):
            x_um = col * spacing_um

            # Shift every second row by half a spacing
            if row % 2 == 1:
                x_um += spacing_um / 2

            y_um = row * dy_um

            points_um.append([x_um, y_um])

    points_um = np.array(points_um)

    if center:
        points_um[:, 0] -= points_um[:, 0].min()
        points_um[:, 1] -= points_um[:, 1].min()

    # Convert microns to pixels, because your function expects x/y in pixels
    points_px = points_um / um_per_px

    hex_df = pd.DataFrame({
        "x": points_px[:, 0],
        "y": points_px[:, 1],
    })

    return hex_df


import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.path import Path

import numpy as np
import pandas as pd
import alphashape
from shapely.geometry import Point


def make_hexagonal_image_selected_by_alpha_shape(
    real_df,
    spacing_px,
    image_shape=(2048, 2048),
    x_col="x",
    y_col="y",
    alpha=0.02,
    lattice_origin_px=(0, 0),
    return_shape=False,
):
    """
    Make a perfect hexagonal lattice over the full image and keep only
    theoretical centres inside a concave hull / alpha shape of the real centres.

    Parameters
    ----------
    real_df : pandas.DataFrame
        Selected real particle centres in pixel coordinates.

    spacing_px : float
        Nearest-neighbour centre-to-centre distance in pixels.

    image_shape : tuple
        Image shape as (height_px, width_px), e.g. (2048, 2048).

    alpha : float
        Controls how tightly the boundary follows the points.
        Smaller/larger values can change the concavity strongly.
        You need to tune this visually.

    lattice_origin_px : tuple
        Origin of the theoretical lattice in image coordinates.

    return_shape : bool
        If True, also return the alpha-shape polygon.

    Returns
    -------
    hex_selected_df : pandas.DataFrame
        Theoretical hexagonal centres inside the alpha shape.
    """

    real_points_px = real_df[[x_col, y_col]].dropna().to_numpy()

    if len(real_points_px) < 4:
        raise ValueError("Need at least 4 real points for an alpha shape.")

    # Make concave hull / alpha shape
    alpha_shape = alphashape.alphashape(real_points_px, alpha)

    height_px, width_px = image_shape

    dy_px = spacing_px * np.sqrt(3) / 2
    x0, y0 = lattice_origin_px

    row_min = int(np.floor((0 - y0) / dy_px)) - 1
    row_max = int(np.ceil((height_px - y0) / dy_px)) + 1

    col_min = int(np.floor((0 - x0) / spacing_px)) - 1
    col_max = int(np.ceil((width_px - x0) / spacing_px)) + 1

    points = []

    for row in range(row_min, row_max + 1):
        y = y0 + row * dy_px

        for col in range(col_min, col_max + 1):
            x = x0 + col * spacing_px

            if row % 2 != 0:
                x += spacing_px / 2

            if 0 <= x <= width_px and 0 <= y <= height_px:
                if alpha_shape.contains(Point(x, y)):
                    points.append([x, y])

    points = np.array(points)

    hex_selected_df = pd.DataFrame({
        x_col: points[:, 0],
        y_col: points[:, 1],
    })

    if return_shape:
        return hex_selected_df, alpha_shape

    return hex_selected_df