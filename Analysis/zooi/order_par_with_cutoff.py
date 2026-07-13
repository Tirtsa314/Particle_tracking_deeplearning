from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt


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
