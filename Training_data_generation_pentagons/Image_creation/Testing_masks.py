# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
import numpy as np
import matplotlib.pyplot as plt
from Training_data_generation_pentagons.Image_creation.Pentagonal_Scatterer_Class import Pentagonal
# -----------------------------
# Settings
# -----------------------------
side_length = 7e-6
thickness = 2.0e-6
rotation = (0, 0, np.deg2rad(60))   # 60 degrees around z
voxel_size = np.array((0.16e-6, 0.16e-6, 0.16e-6), dtype=float)

# -----------------------------
# Create pentagon scatterer
# -----------------------------
pent = Pentagonal(
    side_length=side_length,
    thickness=thickness,
    rotation=rotation,
)

props = pent.properties()

side_length = float(np.asarray(props["side_length"]))
thickness = float(np.asarray(props["thickness"]))
rotation = tuple(float(r) for r in np.asarray(props["rotation"]))

# -----------------------------
# Call your class get() directly
# -----------------------------
vol = pent.get(
    side_length=side_length,
    thickness=thickness,
    rotation=rotation,
    voxel_size=voxel_size,
)

vol = np.asarray(vol, dtype=np.float32)
mask = vol > 0.5

Nx, Ny, Nz = mask.shape
ix = Nx // 2
iy = Ny // 2
iz = Nz // 2

# -----------------------------
# Plot middle slices
# -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 4))

ax[0].imshow(vol[:, :, iz].T, origin="lower", cmap="gray", aspect="equal")
ax[0].set_title("XY slice, z middle")
ax[0].set_xlabel("X index")
ax[0].set_ylabel("Y index")

ax[1].imshow(vol[:, iy, :].T, origin="lower", cmap="gray", aspect="auto")
ax[1].set_title("XZ slice, y middle")
ax[1].set_xlabel("X index")
ax[1].set_ylabel("Z index")

ax[2].imshow(vol[ix, :, :].T, origin="lower", cmap="gray", aspect="auto")
ax[2].set_title("YZ slice, x middle")
ax[2].set_xlabel("Y index")
ax[2].set_ylabel("Z index")

plt.suptitle(
    f"Pentagonal prism\n"
    f"side={side_length*1e6:.2f} µm, "
    f"thickness={thickness*1e6:.2f} µm, "
    f"rotation={rotation}",
    y=1.05,
)

plt.tight_layout()
plt.show()

# -----------------------------
# 3D voxel plot
# -----------------------------
ds = max(1, int(max(mask.shape) / 70))
mask_ds = mask[::ds, ::ds, ::ds]

fig = plt.figure(figsize=(6, 6))
ax3d = fig.add_subplot(111, projection="3d")

ax3d.voxels(
    mask_ds,
    facecolors="red",
    edgecolor="k",
    linewidth=0.25,
)

ax3d.set_title(f"Pentagonal prism, 3D voxels, downsample {ds}x")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")

ax3d.set_xticklabels([])
ax3d.set_yticklabels([])
ax3d.set_zticklabels([])

plt.tight_layout()
plt.show()
# %%
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