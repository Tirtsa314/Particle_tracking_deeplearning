from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
