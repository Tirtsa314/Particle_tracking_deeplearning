#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import cKDTree

def remove_large_overlaps(
    df,
    side_length_um,
    max_overlap_fraction=0.10,
):
    """
    Remove particles until no pair overlaps by more than max_overlap_fraction
    of a single triangle area.

    Overlap fraction is:
        intersection_area / triangle_area

    Returns a filtered DataFrame.
    """
    if len(df) == 0:
        return df.copy()

    coords = df[["x_um", "y_um"]].to_numpy()

    # Circumradius of an equilateral triangle
    R = side_length_um / np.sqrt(3)

    # Only nearby centers can overlap, so use KDTree to find candidate pairs
    tree = cKDTree(coords)
    candidate_pairs = list(tree.query_pairs(r=2 * R))

    # Area of one equilateral triangle
    tri_area = (np.sqrt(3) / 4) * side_length_um**2
    max_allowed_overlap = max_overlap_fraction * tri_area

    # Build shapely polygons once
    polys = []
    for _, row in df.iterrows():
        verts = triangle_vertices(
            center=(row["x_um"], row["y_um"]),
            side_length=side_length_um,
            theta_deg=row["theta_deg"]
        )
        polys.append(ShapelyPolygon(verts))

    keep = np.ones(len(df), dtype=bool)

    while True:
        bad_pairs = []
        bad_count = np.zeros(len(df), dtype=int)

        for i, j in candidate_pairs:
            if not (keep[i] and keep[j]):
                continue

            overlap_area = polys[i].intersection(polys[j]).area

            if overlap_area > max_allowed_overlap:
                bad_pairs.append((i, j, overlap_area))
                bad_count[i] += 1
                bad_count[j] += 1

        if not bad_pairs:
            break

        # Remove one particle from the worst offending pair
        i_worst, j_worst, _ = max(bad_pairs, key=lambda x: x[2])

        if bad_count[i_worst] > bad_count[j_worst]:
            remove_idx = i_worst
        elif bad_count[j_worst] > bad_count[i_worst]:
            remove_idx = j_worst
        else:
            # tie: remove the second one
            remove_idx = j_worst

        keep[remove_idx] = False

    return df.loc[keep].reset_index(drop=True)

def triangle_vertices(center, side_length, theta_deg):
    """
    Vertices of an equilateral triangle centered at its centroid.
    theta_deg rotates the triangle around its center.
    """
    cx, cy = center

    # For an equilateral triangle:
    # distance from center to vertex = side / sqrt(3)
    R = side_length / np.sqrt(3)

    # Default triangle: one vertex points upward
    angles = np.deg2rad(np.array([90, 210, 330]) + theta_deg)

    xs = cx + R * np.cos(angles)
    ys = cy + R * np.sin(angles)

    return np.column_stack([xs, ys])

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import cKDTree


def triangle_vertices(center, side_length, theta_deg):
    """
    Vertices of an equilateral triangle centered at its centroid.
    theta_deg rotates the triangle around its center.
    """
    cx, cy = center

    # distance from center to vertex
    R = side_length / np.sqrt(3)

    # Default triangle: one vertex points upward
    angles = np.deg2rad(np.array([90, 210, 330]) + theta_deg)

    xs = cx + R * np.cos(angles)
    ys = cy + R * np.sin(angles)

    return np.column_stack([xs, ys])


def remove_large_overlaps(
    df,
    side_length_um,
    max_overlap_fraction=0.10,
):
    """
    Remove particles until no pair overlaps by more than max_overlap_fraction
    of a single triangle area.
    """
    if len(df) == 0:
        return df.copy()

    coords = df[["x_um", "y_um"]].to_numpy()

    # Only nearby particles can overlap
    R = side_length_um / np.sqrt(3)
    tree = cKDTree(coords)
    candidate_pairs = list(tree.query_pairs(r=2 * R))

    tri_area = (np.sqrt(3) / 4) * side_length_um**2
    max_allowed_overlap = max_overlap_fraction * tri_area

    polys = []
    for _, row in df.iterrows():
        verts = triangle_vertices(
            center=(row["x_um"], row["y_um"]),
            side_length=side_length_um,
            theta_deg=row["theta_deg"]
        )
        polys.append(ShapelyPolygon(verts))

    keep = np.ones(len(df), dtype=bool)

    while True:
        bad_pairs = []
        bad_count = np.zeros(len(df), dtype=int)

        for i, j in candidate_pairs:
            if not (keep[i] and keep[j]):
                continue

            overlap_area = polys[i].intersection(polys[j]).area

            if overlap_area > max_allowed_overlap:
                bad_pairs.append((i, j, overlap_area))
                bad_count[i] += 1
                bad_count[j] += 1

        if not bad_pairs:
            break

        # remove one particle from the worst pair
        i_worst, j_worst, _ = max(bad_pairs, key=lambda x: x[2])

        if bad_count[i_worst] > bad_count[j_worst]:
            remove_idx = i_worst
        elif bad_count[j_worst] > bad_count[i_worst]:
            remove_idx = j_worst
        else:
            remove_idx = j_worst

        keep[remove_idx] = False

    return df.loc[keep].reset_index(drop=True)


def generate_triangle_crystal(
    image_size_um=(160, 160),
    side_length_um=10.0,
    gap_um=0.3,
    vacancy_fraction=0.03,
    pos_jitter_um=0.15,
    rot_jitter_deg=2.0,
    n_domains=3,
    domain_angle_std_deg=4.0,
    domain_shift_um=0.8,
    line_defect=True,
    line_defect_width_um=2.0,
    line_defect_remove_probability=0.5,
    remove_overlaps=True,
    max_overlap_fraction=0.10,
    seed=1,
):
    rng = np.random.default_rng(seed)

    W, H = image_size_um

    # Effective lattice spacing
    a = side_length_um + gap_um
    h = np.sqrt(3) / 2 * a

    margin = 2 * a
    particles = []

    j_min = int(-margin / h) - 4
    j_max = int((H + margin) / h) + 4

    i_min = int(-margin / a) - j_max - 4
    i_max = int((W + margin) / a) + j_max + 4

    for j in range(j_min, j_max):
        for i in range(i_min, i_max):

            x0 = a * (i + 0.5 * j)
            y0 = h * j

            # Up triangle
            x_up = x0 + 0.5 * a
            y_up = y0 + h / 3
            particles.append([x_up, y_up, 0.0, "up"])

            # Down triangle
            x_down = x0 + a
            y_down = y0 + 2 * h / 3
            particles.append([x_down, y_down, 180.0, "down"])

    df = pd.DataFrame(
        particles,
        columns=["x_um", "y_um", "theta_deg", "orientation"]
    )

    # Crop to image region
    df = df[
        (df["x_um"] >= 0) & (df["x_um"] <= W) &
        (df["y_um"] >= 0) & (df["y_um"] <= H)
    ].copy()

    # Add crystal domains
    if n_domains > 1:
        domain_centers = np.column_stack([
            rng.uniform(0, W, n_domains),
            rng.uniform(0, H, n_domains)
        ])

        domain_angles = rng.normal(0, domain_angle_std_deg, n_domains)
        domain_shifts = rng.normal(0, domain_shift_um, size=(n_domains, 2))

        positions = df[["x_um", "y_um"]].to_numpy()

        distances = np.linalg.norm(
            positions[:, None, :] - domain_centers[None, :, :],
            axis=2
        )
        domain_id = np.argmin(distances, axis=1)

        new_positions = positions.copy()
        new_thetas = df["theta_deg"].to_numpy().copy()

        for d in range(n_domains):
            mask = domain_id == d

            angle = np.deg2rad(domain_angles[d])
            c, s = np.cos(angle), np.sin(angle)
            Rmat = np.array([[c, -s], [s, c]])

            local = positions[mask] - domain_centers[d]
            rotated = local @ Rmat.T

            new_positions[mask] = rotated + domain_centers[d] + domain_shifts[d]
            new_thetas[mask] += domain_angles[d]

        df["x_um"] = new_positions[:, 0]
        df["y_um"] = new_positions[:, 1]
        df["theta_deg"] = new_thetas
        df["domain"] = domain_id

    else:
        df["domain"] = 0

    # Add position jitter
    df["x_um"] += rng.normal(0, pos_jitter_um, len(df))
    df["y_um"] += rng.normal(0, pos_jitter_um, len(df))

    # Add rotational jitter
    df["theta_deg"] += rng.normal(0, rot_jitter_deg, len(df))

    # Add vacancies
    keep = rng.random(len(df)) > vacancy_fraction
    df = df[keep].copy()

    # Add line defect
    if line_defect:
        x = df["x_um"].to_numpy()
        y = df["y_um"].to_numpy()

        y0 = rng.uniform(0.25 * H, 0.75 * H)
        slope = rng.normal(0, 0.2)

        line_y = y0 + slope * (x - W / 2)
        dist_to_line = np.abs(y - line_y) / np.sqrt(1 + slope**2)

        near_line = dist_to_line < line_defect_width_um
        remove = near_line & (rng.random(len(df)) < line_defect_remove_probability)

        df = df[~remove].copy()

    # Final crop after all modifications
    df = df[
        (df["x_um"] >= 0) & (df["x_um"] <= W) &
        (df["y_um"] >= 0) & (df["y_um"] <= H)
    ].copy()

    # Remove too-large overlaps
    if remove_overlaps:
        df = remove_large_overlaps(
            df,
            side_length_um=side_length_um,
            max_overlap_fraction=max_overlap_fraction,
        )

    df = df.reset_index(drop=True)

    return df


def plot_triangle_crystal(
    df,
    image_size_um=(160, 160),
    side_length_um=10.0,
    color_by_domain=True,
):
    W, H = image_size_um

    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in df.iterrows():
        verts = triangle_vertices(
            center=(row["x_um"], row["y_um"]),
            side_length=side_length_um,
            theta_deg=row["theta_deg"]
        )

        if color_by_domain:
            facecolor = plt.cm.tab10(int(row["domain"]) % 10)
        else:
            facecolor = "tab:blue"

        patch = Polygon(
            verts,
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.75
        )
        ax.add_patch(patch)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"Triangle crystal with defects, N = {len(df)}")
    plt.show()


# %%
df = generate_triangle_crystal(
    image_size_um=(160, 160),
    side_length_um=10.0,
    gap_um=0.2,
    vacancy_fraction=0.04,
    pos_jitter_um=0.20,
    rot_jitter_deg=2.5,
    n_domains=4,
    domain_angle_std_deg=5.0,
    domain_shift_um=1.0,
    line_defect=True,
    remove_overlaps=True,
    max_overlap_fraction=0.10,
    seed=None,
)

plot_triangle_crystal(
    df,
    image_size_um=(160, 160),
    side_length_um=10.0
)

df.head()

def plot_triangle_crystal(
    df,
    image_size_um=(160, 160),
    side_length_um=10.0,
    color_by_domain=True,
):
    W, H = image_size_um

    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in df.iterrows():
        verts = triangle_vertices(
            center=(row["x_um"], row["y_um"]),
            side_length=side_length_um,
            theta_deg=row["theta_deg"]
        )

        if color_by_domain:
            facecolor = plt.cm.tab10(int(row["domain"]) % 10)
        else:
            facecolor = "tab:blue"

        patch = Polygon(
            verts,
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.75
        )
        ax.add_patch(patch)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"Triangle crystal with defects, N = {len(df)}")
    plt.show()



def plot_triangle_crystal_centre(
df,
image_size_um=(160, 160),
side_length_um=10.0,
color_by_domain=True,
show_centres=False,
centre_color="red",
centre_size=12,
):
    W, H = image_size_um

    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in df.iterrows():
        verts = triangle_vertices(
            center=(row["x_um"], row["y_um"]),
            side_length=side_length_um,
            theta_deg=row["theta_deg"]
        )

        if color_by_domain:
            facecolor = plt.cm.tab10(int(row["domain"]) % 10)
        else:
            facecolor = "tab:blue"

        patch = Polygon(
            verts,
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.75
        )
        ax.add_patch(patch)

    # Optional: show particle centres
    if show_centres:
        ax.scatter(
            df["x_um"],
            df["y_um"],
            s=centre_size,
            c=centre_color,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            label="centres"
        )
        ax.legend()

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"Triangle crystal with defects, N = {len(df)}")
    plt.show()
# %%
#%%
# df = generate_triangle_crystal(
#     image_size_um=(160, 160),
#     side_length_um=10.0,
#     gap_um=0.2,
#     vacancy_fraction=0.04,
#     pos_jitter_um=0.20,
#     rot_jitter_deg=2.5,
#     n_domains=4,
#     domain_angle_std_deg=5.0,
#     domain_shift_um=1.0,
#     line_defect=True,
#     seed=None,
# )

# plot_triangle_crystal(
#     df,
#     image_size_um=(160, 160),
#     side_length_um=10.0
# )

# df.head()

df = generate_triangle_crystal(
    image_size_um=(160, 160),
    side_length_um=10.0,
    gap_um=0,
    vacancy_fraction=0.0,
    pos_jitter_um=0,
    rot_jitter_deg=0,
    n_domains=1,
    domain_angle_std_deg=0,
    domain_shift_um=0,
    line_defect=False,
    seed=None,
)

plot_triangle_crystal(
    df,
    image_size_um=(160, 160),
    side_length_um=10.0
)

df.head()


# %%
plot_triangle_crystal_centre(
    df,
    image_size_um=(160, 160),
    side_length_um=10.0,
    show_centres=True,
    centre_size=6,
    centre_color="yellow"
)
# %%
