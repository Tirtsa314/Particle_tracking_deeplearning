#%%
import deeptrack as dt
import numpy as np
from deeptrack import units as u
from deeptrack.backend.units import ConversionTable
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

side_length=10e-6
thickness=2.5e-6
rotation=(0, 0, 60)
voxel_size=(0.16e-6, 0.16e-6, 0.16e-6)

R = side_length/np.sqrt(3) #R is centre to vertex distance, so furthest distance from centre

# Determine grid size
R_ceil = np.ceil(np.max(R) / np.min(voxel_size[:2])) #rounds to upper int needed to cover the object
thickness_ceil = np.ceil(thickness * 0.5 / voxel_size[2])
ceil = int(max(R_ceil, thickness_ceil))

# Create grid
x = np.arange(-ceil, ceil) * voxel_size[0]
y = np.arange(-ceil, ceil) * voxel_size[1]
z = np.arange(-ceil, ceil) * voxel_size[2]
Y, X, Z = np.meshgrid(x, y, z, indexing="ij")

# Rotate the grid
cos = np.cos(rotation)
sin = np.sin(rotation)
XR = (
    cos[0] * cos[1] * X
    + (cos[0] * sin[1] * sin[2] - sin[0] * cos[2]) * Y
    + (cos[0] * sin[1] * cos[2] + sin[0] * sin[2]) * Z
)
YR = (
    sin[0] * cos[1] * X
    + (sin[0] * sin[1] * sin[2] + cos[0] * cos[2]) * Y
    + (sin[0] * sin[1] * cos[2] - cos[0] * sin[2]) * Z
)
ZR = (-sin[1] * X) + cos[1] * sin[2] * Y + cos[1] * cos[2] * Z

# ---- signed distances to the prism faces (positive inside) ----
a = np.sqrt(3) / 3
norm_oblique = np.sqrt(a**2 + 1)

d_left  = XR + R / 2
d_right = R - XR
d_low   = (YR - (a * XR - a * R)) / norm_oblique
d_high  = ((-a * XR + a * R) - YR) / norm_oblique
d_z     = thickness / 2 - np.abs(ZR)

# # hard prism
# inside = (
#     (d_left  >= 0) &
#     (d_right >= 0) &
#     (d_low   >= 0) &
#     (d_high  >= 0) &
#     (d_z     >= 0)
# )

# # ---- scrape only edges, not flat surfaces ----
# # thickness of bevel region
# edge_scrape = 1.0 * min(voxel_size)   # try 1 voxel first
# # edge_scrape = 2.0 * min(voxel_size) # stronger bevel

# close_to_face = np.stack([
#     d_left  < edge_scrape,
#     d_right < edge_scrape,
#     d_low   < edge_scrape,
#     d_high  < edge_scrape,
#     d_z     < edge_scrape,
# ], axis=0)

# n_close = np.sum(close_to_face, axis=0)

# # near 2 or more faces => edge/corner region
# edge_region = inside & (n_close >= 2)

# # remove only that edge layer
# mask = inside & (~edge_region)

# ---- signed distances to the 3 triangle side faces ----
# Positive means inside.
# For this triangle, d_right is redundant because the two oblique faces meet at x=R.


corner_radius = 0.8e-6
side_distances = [d_left, d_low, d_high]

inside_xy = (
    (d_left >= 0) &
    (d_low >= 0) &
    (d_high >= 0)
)

inside_z = d_z >= 0

# ---- rounded XY corners by erode-then-dilate ----
r = float(corner_radius)

if r <= 0:
    rounded_xy = inside_xy.astype(np.float32)
else:
    # Erode triangle by radius r:
    # keep only points at least r away from all triangle side faces.
    eroded_xy = (
        (d_left >= r) &
        (d_low >= r) &
        (d_high >= r)
    )

    rounded_xy = np.zeros_like(inside_xy, dtype=np.float32)

    vx, vy = float(voxel_size[0]), float(voxel_size[1])
    aa = 1.0 * min(vx, vy)  # anti-alias transition width, about 1 simulation voxel

    # Apply per z-slice.
    # This is good for in-plane particles. If you later use strong x/y tilts,
    # this is still an approximation.
    for k in range(inside_xy.shape[2]):
        eroded2 = eroded_xy[:, :, k]
        inside2 = inside_xy[:, :, k]

        if not np.any(eroded2):
            rounded_xy[:, :, k] = inside2.astype(np.float32)
            continue

        # Distance from every pixel to the smaller eroded triangle.
        dist_to_eroded = distance_transform_edt(~eroded2, sampling=(vx, vy))

        # Hard rounded mask:
        # rounded2 = inside2 & (dist_to_eroded <= r)

        # Softer fractional boundary:
        rounded2 = np.clip((r - dist_to_eroded) / aa + 0.5, 0.0, 1.0)

        # Do not allow pixels outside the original sharp triangle.
        rounded2 = rounded2 * inside2.astype(np.float32)

        rounded_xy[:, :, k] = rounded2

# keep top/bottom flat for now
mask = rounded_xy * inside_z.astype(np.float32)

# mask = mask.astype(np.float32)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

ax.voxels(mask, facecolors='red', edgecolor='k')

ax.set(title='Voxel mask of the scatterer', xlabel='X', ylabel='Y', zlabel='Z')
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad=-10

plt.show()
# %%
#%%
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
side_length = 8e-6          # 8 microns
thickness   = 2e-6          # 2 microns
rotation    = (0, 0, 0)     # in degrees
voxel_size  = (0.32e-6, 0.32e-6, 0.32e-6)

# Convert rotation to radians
rotation = np.deg2rad(rotation)

# -----------------------------
# Regular pentagon geometry
# -----------------------------
# Circumradius of a regular pentagon from side length
R = side_length / (2 * np.sin(np.pi / 5))

# Determine grid size
R_ceil = np.ceil(R / np.min(voxel_size[:2]))
thickness_ceil = np.ceil((thickness / 2) / voxel_size[2])
ceil = int(max(R_ceil, thickness_ceil)) + 2   # small margin

# Create grid
x = np.arange(-ceil, ceil) * voxel_size[0]
y = np.arange(-ceil, ceil) * voxel_size[1]
z = np.arange(-ceil, ceil) * voxel_size[2]
Y, X, Z = np.meshgrid(x, y, z, indexing="ij")

# -----------------------------
# Rotate the grid
# -----------------------------
cx, cy, cz = np.cos(rotation)
sx, sy, sz = np.sin(rotation)

XR = (
    cx * cy * X
    + (cx * sy * sz - sx * cz) * Y
    + (cx * sy * cz + sx * sz) * Z
)
YR = (
    sx * cy * X
    + (sx * sy * sz + cx * cz) * Y
    + (sx * sy * cz - cx * sz) * Z
)
ZR = (
    -sy * X
    + cy * sz * Y
    + cy * cz * Z
)

# -----------------------------
# Pentagon vertices in XY plane
# -----------------------------
# Start with one vertex pointing upward
angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/2
vertices = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=1)

# -----------------------------
# Signed distances to pentagon side faces
# Positive inside
# -----------------------------
d_faces = []

for i in range(5):
    v1 = vertices[i]
    v2 = vertices[(i + 1) % 5]
    edge = v2 - v1

    # For a CCW polygon, the inside is on the left side of each edge
    # signed distance = cross(edge, point-v1) / |edge|
    dx = XR - v1[0]
    dy = YR - v1[1]
    cross = edge[0] * dy - edge[1] * dx
    dist = cross / np.linalg.norm(edge)
    d_faces.append(dist)

d_faces = np.stack(d_faces, axis=0)

# Distance to top/bottom
d_z = thickness / 2 - np.abs(ZR)

# Inside the pentagonal prism
inside_2d = np.all(d_faces >= 0, axis=0)
inside = inside_2d & (d_z >= 0)

# -----------------------------
# Optional: scrape only edges/corners
# -----------------------------
edge_scrape = 1.0 * min(voxel_size)   # one voxel bevel

close_to_face = np.concatenate([
    d_faces < edge_scrape,
    (d_z[None, ...] < edge_scrape)
], axis=0)

n_close = np.sum(close_to_face, axis=0)

# Remove only edge/corner voxels
edge_region = inside & (n_close >= 2)

# Final mask
mask = inside & (~edge_region)

# If you want a solid pentagon without scraped edges, use:
# mask = inside

mask = mask.astype(np.float32)

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.voxels(mask, facecolors='red', edgecolor='k')

ax.set_title('Voxel mask of a pentagonal prism')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = -10

plt.show()
#%%