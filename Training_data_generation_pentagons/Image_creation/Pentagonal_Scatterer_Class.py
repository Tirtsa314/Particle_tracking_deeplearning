
import deeptrack as dt
import numpy as np
from deeptrack import units as u
from deeptrack.backend.units import ConversionTable


class Pentagonal(dt.Scatterer):
    __conversion_table__ = ConversionTable(
        side_length=(u.meter, u.meter),
        thickness=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        side_length=7e-6,
        thickness=2.0e-6,
        rotation=(0, 0, 0),
        refractive_index=None,
        **kwargs,
    ):
        init_kwargs = dict(kwargs)
        init_kwargs["side_length"] = side_length
        init_kwargs["thickness"] = thickness
        init_kwargs["rotation"] = rotation

        if refractive_index is not None:
            init_kwargs["refractive_index"] = refractive_index

        super().__init__(**init_kwargs)

    def _process_properties(self, properties):
        properties = super()._process_properties(properties)

        # Ensure side_length is scalar
        properties["side_length"] = float(np.array(properties["side_length"]))

        # Ensure thickness is scalar
        properties["thickness"] = float(np.array(properties["thickness"]))

        # Ensure rotation has length 3
        rot = np.array(properties["rotation"], dtype=float)

        if rot.ndim == 0:
            rot = [rot, 0, 0]
        elif rot.size == 1:
            rot = [rot[0], 0, 0]
        elif rot.size == 2:
            rot = [rot[0], rot[1], 0]
        else:
            rot = rot[:3]

        properties["rotation"] = tuple(float(r) for r in rot)

        return properties

    def get(
        self,
        *ignore,
        side_length,
        thickness,
        rotation,
        voxel_size,
        **kwargs
    ):
        # -------------------------------------------------
        # 1. Geometry of a regular pentagon
        # -------------------------------------------------
        # side_length = polygon edge length
        # circumradius = center -> vertex distance
        R = side_length / (2 * np.sin(np.pi / 5))

        # apothem = center -> side distance
        apothem = side_length / (2 * np.tan(np.pi / 5))

        # Grid size
        R_ceil = np.ceil(R / np.min(voxel_size[:2]))
        thickness_ceil = np.ceil((thickness / 2) / voxel_size[2])
        ceil = int(max(R_ceil, thickness_ceil)) + 2

        # -------------------------------------------------
        # 2. Create voxel grid
        # -------------------------------------------------
        x = np.arange(-ceil, ceil) * voxel_size[0]
        y = np.arange(-ceil, ceil) * voxel_size[1]
        z = np.arange(-ceil, ceil) * voxel_size[2]

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # -------------------------------------------------
        # 3. Rotate grid into particle frame
        # -------------------------------------------------
        rx, ry, rz = rotation

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx,  cx]
        ], dtype=float)

        Ry = np.array([
            [ cy, 0, sy],
            [  0, 1,  0],
            [-sy, 0, cy]
        ], dtype=float)

        Rz = np.array([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [ 0,   0, 1]
        ], dtype=float)

        # World -> object frame
        Rmat = Rz @ Ry @ Rx
        Rt = Rmat.T

        XR = Rt[0, 0] * X + Rt[0, 1] * Y + Rt[0, 2] * Z
        YR = Rt[1, 0] * X + Rt[1, 1] * Y + Rt[1, 2] * Z
        ZR = Rt[2, 0] * X + Rt[2, 1] * Y + Rt[2, 2] * Z

        # -------------------------------------------------
        # 4. Regular pentagon in the XR-YR plane
        # -------------------------------------------------
        # One vertex points along +x direction
        angles = 2 * np.pi * np.arange(5) / 5.0
        verts = np.column_stack([
            R * np.cos(angles),
            R * np.sin(angles)
        ])

        # Make sure vertices are CCW
        # (these angles already are, but this keeps the logic explicit)
        # Signed distance to each polygon side:
        # inside means "left side" of every directed edge for a CCW polygon
        d_faces = []
        for i in range(5):
            v0 = verts[i]
            v1 = verts[(i + 1) % 5]
            ex = v1[0] - v0[0]
            ey = v1[1] - v0[1]
            edge_len = np.sqrt(ex**2 + ey**2)

            # signed distance to the infinite line of the edge
            d = (ex * (YR - v0[1]) - ey * (XR - v0[0])) / edge_len
            d_faces.append(d)

        d_faces = np.stack(d_faces, axis=0)

        # z faces
        d_z = thickness / 2 - np.abs(ZR)

        # Inside the prism
        inside_xy = np.all(d_faces >= 0, axis=0)
        inside = inside_xy & (d_z >= 0)

        # -------------------------------------------------
        # 5. Optional edge scraping / beveling
        # -------------------------------------------------
        edge_scrape = 1.0 * min(voxel_size)

        close_to_side_faces = d_faces < edge_scrape
        close_to_z_face = d_z < edge_scrape

        close_to_any_face = np.concatenate(
            [close_to_side_faces, close_to_z_face[None, ...]],
            axis=0
        )

        n_close = np.sum(close_to_any_face, axis=0)

        # Remove only true edges/corners, not flat face interiors
        edge_region = inside & (n_close >= 2)

        mask = inside & (~edge_region)

        return mask.astype(np.float32)


#%%











# import numpy as np
# import matplotlib.pyplot as plt

# # Optional: 3D plotting
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# # ---- Put your Triangular class definition ABOVE this, but keep ONLY ONE _process_properties ----
# # (Your get() is fine as-is.)

# def plot_triangular_mask(scatterer, voxel_size=(0.2e-6, 0.2e-6, 0.2e-6), title="Triangular prism"):
#     """
#     Calls scatterer.get(...) directly (bypassing optics) and visualizes the resulting 3D mask.
#     """
#     props = scatterer.properties()  # deeptrack scatterer stores current properties here
#     side_length = props["side_length"]
#     thickness = props["thickness"]
#     rotation = props["rotation"]

#     # Call get() directly
#     vol = scatterer.get(
#         side_length=side_length,
#         thickness=thickness,
#         rotation=rotation,
#         voxel_size=np.array(voxel_size, dtype=float),
#     )
#     vol = np.asarray(vol)
#     Nx, Ny, Nz = vol.shape   # vol[ix, iy, iz] corresponds to (x, y, z)

#     ix = Nx // 2
#     iy = Ny // 2
#     iz = Nz // 2

#     fig, ax = plt.subplots(1, 3, figsize=(13, 4))

#     # --- XY slice at fixed Z ---
#     # axes: y (rows) vs x (cols) when displayed
#     ax[0].imshow(vol[:, :, iz].T, origin="lower", cmap="gray", aspect="equal")
#     ax[0].set_title("XY slice (Z mid)")
#     ax[0].set_xlabel("X index")
#     ax[0].set_ylabel("Y index")

#     # --- XZ slice at fixed Y ---
#     # axes: z (rows) vs x (cols) when displayed
#     ax[1].imshow(vol[:, iy, :].T, origin="lower", cmap="gray", aspect="auto")
#     ax[1].set_title("XZ slice (Y mid)")
#     ax[1].set_xlabel("X index")
#     ax[1].set_ylabel("Z index")

#     # --- YZ slice at fixed X ---
#     # axes: z (rows) vs y (cols) when displayed
#     ax[2].imshow(vol[ix, :, :].T, origin="lower", cmap="gray", aspect="auto")
#     ax[2].set_title("YZ slice (X mid)")
#     ax[2].set_xlabel("Y index")
#     ax[2].set_ylabel("Z index")

#     plt.suptitle(
#         f"{title}\nside_length={side_length:.3e} m, thickness={thickness:.3e} m, rotation={rotation}",
#         y=1.05
#     )
#     plt.tight_layout()
#     plt.show()

#     # 3D voxel plot (downsample to keep it light)
#     # Convert to boolean for voxels
#     mask = vol > 0.5

#     # Downsample if large
#     ds = max(1, int(max(mask.shape) / 64))  # aim for <= ~64 in largest dimension
#     mask_ds = mask[::ds, ::ds, ::ds]

#     fig = plt.figure(figsize=(6, 6))
#     ax3d = fig.add_subplot(111, projection="3d")
#     ax3d.voxels(mask_ds, edgecolor="k")
#     ax3d.set_title(f"3D voxels (downsample {ds}x)")
#     ax3d.set_xlabel("Y index (downsampled)")
#     ax3d.set_ylabel("X index (downsampled)")
#     ax3d.set_zlabel("Z index (downsampled)")
#     plt.tight_layout()
#     plt.show()

#     return vol


# # ------------------ Example usage ------------------
# # Assuming you have deeptrack imported and ConversionTable/u set up in your class file:
# # import deeptrack as dt
# # from deeptrack import units as u
# # from deeptrack.backend.units import ConversionTable

# # Example rotations to test:
# # rotation = (alpha(z), beta(y), gamma(x)) based on the matrix we derived.
# tri0 = Triangular(side_length=10e-6, thickness=2.5e-6, rotation=(0, 0, 0))
# plot_triangular_mask(tri0, voxel_size=(0.2e-6, 0.2e-6, 0.2e-6), title="No rotation")

# tri_z = Triangular(side_length=10e-6, thickness=2.5e-6, rotation=(np.pi/6, 0, 0))
# plot_triangular_mask(tri_z, voxel_size=(0.2e-6, 0.2e-6, 0.2e-6), title="Rotate around Z (alpha)")

# tri_y = Triangular(side_length=10e-6, thickness=2.5e-6, rotation=(0, np.pi/6, 0))
# plot_triangular_mask(tri_y, voxel_size=(0.2e-6, 0.2e-6, 0.2e-6), title="Rotate around Y (beta)")

# tri_x = Triangular(side_length=10e-6, thickness=2.5e-6, rotation=(0, 0, np.pi/6))
# plot_triangular_mask(tri_x, voxel_size=(0.2e-6, 0.2e-6, 0.2e-6), title="Rotate around X (gamma)")

# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# import deeptrack as dt

# def simulate_one(triangle, optics=None, title=""):
#     if optics is None:
#         optics = dt.Fluorescence()

#     pip = optics(triangle)
#     pip.update()
#     img = pip.resolve()

#     plt.figure(figsize=(4,4))
#     plt.imshow(img, cmap="gray")
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

#     return img

# rotation = (np.pi+np.pi/6, 0, 0)
# # Example usage:
# rx, ry, rz = rotation
# tri = Triangular(
#     side_length=10e-6,
#     thickness=2.5e-6,
#     position=(64, 64),#(y,x)     # put it roughly in the center of a 128x128 image
#     rotation=rotation
# )
# simulate_one(tri, dt.Fluorescence(),f"Triangular prism — rotation (rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f})"
# )
# #%%




















#     def get(
#         self,
#         *ignore,
#         side_length,
#         thickness,
#         rotation,
#         voxel_size,
#         **kwargs
#     ):
#         R = side_length/np.sqrt(3)

#         # Determine grid size
#         R_ceil = np.ceil(np.max(R) / np.min(voxel_size[:2]))
#         thickness_ceil = np.ceil(thickness * 0.5 / voxel_size[2])
#         ceil = int(max(R_ceil, thickness_ceil))

#         # Create grid
#         x = np.arange(-ceil, ceil) * voxel_size[0]
#         y = np.arange(-ceil, ceil) * voxel_size[1]
#         z = np.arange(-ceil, ceil) * voxel_size[2]
#         Y, X, Z = np.meshgrid(y, x, z, sparse=True)

#         # # Rotate the grid
#         # cos = np.cos(rotation)
#         # sin = np.sin(rotation)
#         # XR = (
#         #     cos[0] * cos[1] * X
#         #     + (cos[0] * sin[1] * sin[2] - sin[0] * cos[2]) * Y
#         #     + (cos[0] * sin[1] * cos[2] + sin[0] * sin[2]) * Z
#         # )
#         # YR = (
#         #     sin[0] * cos[1] * X
#         #     + (sin[0] * sin[1] * sin[2] + cos[0] * cos[2]) * Y
#         #     + (sin[0] * sin[1] * cos[2] - cos[0] * sin[2]) * Z
#         # )
#         # ZR = (-sin[1] * X) + cos[1] * sin[2] * Y + cos[1] * cos[2] * Z
# # --- Rotate the grid (REPLACE your XR/YR/ZR formulas with this) ---
#         rx, ry, rz = rotation
#         cx, sx = np.cos(rx), np.sin(rx)
#         cy, sy = np.cos(ry), np.sin(ry)
#         cz, sz = np.cos(rz), np.sin(rz)

#         Rx = np.array([[1, 0, 0],
#                     [0, cx, -sx],
#                     [0, sx,  cx]], dtype=np.float32)
#         Ry = np.array([[ cy, 0, sy],
#                     [  0, 1,  0],
#                     [-sy, 0, cy]], dtype=np.float32)
#         Rz = np.array([[cz, -sz, 0],
#                     [sz,  cz, 0],
#                     [ 0,   0, 1]], dtype=np.float32)

#         R = Rz @ Ry @ Rx

#         # grid -> object frame uses inverse
#         Rt = R.T

#         # IMPORTANT: take SCALARS from Rt, then combine X,Y,Z (broadcasts fine)
#         r00, r01, r02 = float(Rt[0, 0]), float(Rt[0, 1]), float(Rt[0, 2])
#         r10, r11, r12 = float(Rt[1, 0]), float(Rt[1, 1]), float(Rt[1, 2])
#         r20, r21, r22 = float(Rt[2, 0]), float(Rt[2, 1]), float(Rt[2, 2])

#         XR = r00 * X + r01 * Y + r02 * Z
#         YR = r10 * X + r11 * Y + r12 * Z
#         ZR = r20 * X + r21 * Y + r22 * Z


#         # Triangle mask
#         mask = (
#         (-R/2 < XR) & (XR < R) &
#         ((np.sqrt(3)/3 * XR - np.sqrt(3)/3 * R) < YR) &
#         (YR < (-np.sqrt(3)/3 * XR + np.sqrt(3)/3 * R)) &
#         (np.abs(ZR) < thickness/2)
#     )
#         return mask.astype(np.float32)

# def get(self, *ignore, side_length, thickness, rotation, voxel_size, **kwargs):
#     Rtri = side_length / np.sqrt(3)  # triangle “radius” (scalar)

#     # Determine grid size
#     R_ceil = np.ceil(np.max(Rtri) / np.min(voxel_size[:2]))
#     thickness_ceil = np.ceil(thickness * 0.5 / voxel_size[2])
#     ceil = int(max(R_ceil, thickness_ceil))

#     # Create grid
#     x = np.arange(-ceil, ceil) * voxel_size[0]
#     y = np.arange(-ceil, ceil) * voxel_size[1]
#     z = np.arange(-ceil, ceil) * voxel_size[2]
#     Y, X, Z = np.meshgrid(y, x, z, sparse=True)

#     # Rotation matrix (same convention as your sampler)
#     rx, ry, rz = rotation
#     cx, sx = np.cos(rx), np.sin(rx)
#     cy, sy = np.cos(ry), np.sin(ry)
#     cz, sz = np.cos(rz), np.sin(rz)

#     Rx = np.array([[1, 0, 0],
#                    [0, cx, -sx],
#                    [0, sx,  cx]], dtype=np.float32)
#     Ry = np.array([[ cy, 0, sy],
#                    [  0, 1,  0],
#                    [-sy, 0, cy]], dtype=np.float32)
#     Rz = np.array([[cz, -sz, 0],
#                    [sz,  cz, 0],
#                    [ 0,   0, 1]], dtype=np.float32)

#     Rmat = Rz @ Ry @ Rx

#     # grid -> object frame uses inverse
#     Rt = Rmat.T

#     # scalar coefficients so broadcasting works
#     r00, r01, r02 = float(Rt[0, 0]), float(Rt[0, 1]), float(Rt[0, 2])
#     r10, r11, r12 = float(Rt[1, 0]), float(Rt[1, 1]), float(Rt[1, 2])
#     r20, r21, r22 = float(Rt[2, 0]), float(Rt[2, 1]), float(Rt[2, 2])

#     XR = r00 * X + r01 * Y + r02 * Z
#     YR = r10 * X + r11 * Y + r12 * Z
#     ZR = r20 * X + r21 * Y + r22 * Z

#     # Triangle mask (use Rtri, not the rotation matrix!)
#     mask = (
#         (-Rtri/2 < XR) & (XR < Rtri) &
#         ((np.sqrt(3)/3 * XR - np.sqrt(3)/3 * Rtri) < YR) &
#         (YR < (-np.sqrt(3)/3 * XR + np.sqrt(3)/3 * Rtri)) &
#         (np.abs(ZR) < thickness/2)
#     )

#     return mask.astype(np.float32)
# %%
