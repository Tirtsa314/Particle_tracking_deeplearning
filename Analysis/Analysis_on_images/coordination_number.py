
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_coordination_number_on_image(
    image,
    coord_df,
    coord_col="coord_num",
    x_col="x",
    y_col="y",
    s=10,
    cmap_name="tab10",
    image_shape=None,
    x_frac=(0, 1),
    y_frac=(0, 1),
    output_path=None,
):
    """
    Plot coordination number on top of the microscopy image.

    Parameters
    ----------
    image : ndarray
        Image array, e.g. arr8.

    coord_df : pandas.DataFrame
        DataFrame containing x/y coordinates and coordination number.

    coord_col : str
        Column with coordination number values.

    x_col, y_col : str
        Coordinate columns in pixels.

    s : float
        Scatter marker size.

    cmap_name : str
        Matplotlib colormap name for discrete colors.

    image_shape : tuple or None
        Usually arr8.shape. If None, taken from image.

    x_frac, y_frac : tuple
        Fractions for cropping, e.g. (0, 0.5) for left/top half.

    output_path : str or Path or None
        If given, saves the figure.
    """

    if coord_col not in coord_df.columns:
        raise ValueError(
            f"{coord_col} not found in coord_df.\n"
            f"Available columns are:\n{list(coord_df.columns)}"
        )

    image = np.asarray(image)

    if image_shape is None:
        image_shape = image.shape

    valid = (
        np.isfinite(coord_df[x_col])
        & np.isfinite(coord_df[y_col])
        & np.isfinite(coord_df[coord_col])
    )

    plot_df = coord_df[valid].copy()

    x_px = plot_df[x_col].to_numpy(dtype=float)
    y_px = plot_df[y_col].to_numpy(dtype=float)
    coord_values = plot_df[coord_col].to_numpy(dtype=int)

    unique_coord = np.sort(np.unique(coord_values))

    boundaries = np.arange(
        unique_coord.min() - 0.5,
        unique_coord.max() + 1.5,
        1
    )

    cmap = plt.get_cmap(cmap_name, len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    ax.imshow(image, cmap="gray")

    sc = ax.scatter(
        x_px,
        y_px,
        c=coord_values,
        s=s,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )

    cbar = plt.colorbar(
        sc,
        ax=ax,
        boundaries=boundaries,
        ticks=unique_coord,
        pad=0.02,
    )
    cbar.set_label("Coordination number", size=14)
    cbar.ax.tick_params(labelsize=12)

    H, W = image_shape[:2]

    xlim_px = (x_frac[0] * W, x_frac[1] * W)
    ylim_px = (y_frac[0] * H, y_frac[1] * H)

    ax.set_xlim(xlim_px)
    ax.set_ylim(ylim_px[1], ylim_px[0])   # image-style y axis

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")

    mean_coord = np.nanmean(coord_values)

    ax.set_title(
        "Coordination number on image\n"
        + f"mean coordination number = {mean_coord:.2f}"
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax