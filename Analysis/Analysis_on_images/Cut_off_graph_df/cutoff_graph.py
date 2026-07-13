
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def compute_cutoff_graph_from_df(
    df,
    cutoff_um,
    um_per_px=0.65,
    x_col="x",
    y_col="y",
    particle_id_col=None,
):
    """
    Compute neighbour graph based on cutoff distance using cKDTree.

    Returns
    -------
    G : networkx.Graph
        Graph where particles are connected if their distance is <= cutoff_um.

    graph_df : pandas.DataFrame
        Copy of df with added cutoff coordination number.
    """

    graph_df = df.copy().reset_index(drop=True)

    valid = np.isfinite(graph_df[x_col]) & np.isfinite(graph_df[y_col])
    graph_df = graph_df[valid].reset_index(drop=True)

    coords_px = graph_df[[x_col, y_col]].to_numpy(dtype=float)
    coords_um = coords_px * um_per_px

    if particle_id_col is None:
        node_ids = graph_df.index.to_numpy()
    else:
        node_ids = graph_df[particle_id_col].to_numpy()

    G = nx.Graph()

    # Add nodes
    for i, node_id in enumerate(node_ids):
        G.add_node(
            node_id,
            cutoff_index=i,
            x=float(graph_df.loc[i, x_col]),
            y=float(graph_df.loc[i, y_col]),
        )

    # Find neighbours within cutoff
    tree = cKDTree(coords_um)
    neighbours_all = tree.query_ball_point(coords_um, r=cutoff_um)

    # Add edges
    for i, neighbours in enumerate(neighbours_all):
        for j in neighbours:

            # Skip self
            if i == j:
                continue

            # Avoid adding each edge twice
            if j <= i:
                continue

            node_i = node_ids[i]
            node_j = node_ids[j]

            distance_um = np.linalg.norm(coords_um[j] - coords_um[i])

            G.add_edge(
                node_i,
                node_j,
                distance_um=distance_um,
            )

    # Add coordination number to dataframe
    coord_nums = []

    for node_id in node_ids:
        if node_id in G:
            coord_nums.append(G.degree[node_id])
        else:
            coord_nums.append(np.nan)

    graph_df["cutoff_graph_coord_num"] = coord_nums

    return G, graph_df



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import matplotlib.cm as cm
from pathlib import Path


def plot_cutoff_graph_cutout_um(
    image,
    graph_df,
    G=None,
    x_col="x",
    y_col="y",
    value_col="cutoff_graph_coord_num",
    um_per_px=0.65,
    xlim_um=None,
    ylim_um=None,
    xlim_px=None,
    ylim_px=None,
    show_image=True,
    show_graph=True,
    show_centers=True,
    color_centers_by_value=True,
    cmap_name="tab20",
    graph_color="cyan",
    graph_line_width=0.6,
    graph_alpha=0.8,
    center_color="red",
    center_size=8,
    center_alpha=1.0,
    image_cmap="gray",
    figsize=(7, 7),
    title=None,
    save_path=None,
    dpi=300,
    show_legend=True,
    legend_loc="center left",
    legend_bbox_to_anchor=(1.02, 0.5),
    legend_alpha=0.8,
):
    """
    Plot a cut-out of the microscopy image with cutoff-neighbour graph lines
    and particle centers.

    Coordinates on the axes are shown in micrometers.

    Parameters
    ----------
    image : array
        Microscopy image, e.g. arr8.

    graph_df : pandas.DataFrame
        DataFrame returned by compute_cutoff_graph_from_df.
        Should contain x/y positions and cutoff_graph_coord_num.

    G : networkx.Graph or None
        Cutoff neighbour graph returned by compute_cutoff_graph_from_df.

    value_col : str
        Column used to color centers, usually "cutoff_graph_coord_num".

    xlim_um, ylim_um : tuple or None
        Crop limits in micrometers.

    xlim_px, ylim_px : tuple or None
        Alternative crop limits in pixels.
        If both px and um limits are given, micrometer limits are used.

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
                zorder=5,
            )

            ax.add_collection(line_collection)

    # -------------------------
    # Centers
    # -------------------------
    legend_handles = []

    if show_centers:
        xs_um = graph_df[x_col].to_numpy(dtype=float) * um_per_px
        ys_um = graph_df[y_col].to_numpy(dtype=float) * um_per_px

        crop_mask = (
            (xs_um >= xlim_um[0]) &
            (xs_um <= xlim_um[1]) &
            (ys_um >= ylim_um[0]) &
            (ys_um <= ylim_um[1])
        )

        if color_centers_by_value and value_col in graph_df.columns:

            values = graph_df[value_col].to_numpy()
            values_crop = values[crop_mask]

            cmap = cm.get_cmap(cmap_name, 20)

            # Only use finite values for coloring
            finite_values = values_crop[np.isfinite(values_crop)]

            if len(finite_values) > 0:
                unique_values = np.array(sorted(np.unique(finite_values).astype(int)))

                value_to_rgba = {}

                for value in unique_values:
                    color_index = int(value) % 20
                    r, g, b, _ = cmap(color_index)
                    value_to_rgba[int(value)] = (r, g, b, center_alpha)

                center_colors = []

                for value in values_crop:
                    if np.isfinite(value):
                        center_colors.append(value_to_rgba[int(value)])
                    else:
                        center_colors.append((0.5, 0.5, 0.5, center_alpha))

                ax.scatter(
                    xs_um[crop_mask],
                    ys_um[crop_mask],
                    s=center_size,
                    c=center_colors,
                    linewidths=0,
                    zorder=10,
                )

                legend_handles = [
                    Patch(
                        facecolor=(
                            value_to_rgba[int(value)][0],
                            value_to_rgba[int(value)][1],
                            value_to_rgba[int(value)][2],
                            legend_alpha,
                        ),
                        edgecolor="black",
                        label=str(int(value)),
                    )
                    for value in unique_values
                ]

            else:
                ax.scatter(
                    xs_um[crop_mask],
                    ys_um[crop_mask],
                    s=center_size,
                    c=center_color,
                    linewidths=0,
                    alpha=center_alpha,
                    zorder=10,
                )

        else:
            ax.scatter(
                xs_um[crop_mask],
                ys_um[crop_mask],
                s=center_size,
                c=center_color,
                linewidths=0,
                alpha=center_alpha,
                zorder=10,
            )

    # -------------------------
    # Legend
    # -------------------------
    if (
        show_centers
        and color_centers_by_value
        and show_legend
        and len(legend_handles) > 0
    ):
        ax.legend(
            handles=legend_handles,
            title="Cutoff graph\ncoordination",
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=True,
        )

    # -------------------------
    # Axes and layout
    # -------------------------
    ax.set_xlim(xlim_um)
    ax.set_ylim(ylim_um[1], ylim_um[0])  # reversed because image y-axis points downward

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$\mu$m]")
    ax.set_ylabel(r"$y$ [$\mu$m]")

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax