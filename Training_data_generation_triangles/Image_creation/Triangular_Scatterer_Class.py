# %% 
import deeptrack as dt
import numpy as np
from deeptrack import units as u
from deeptrack.backend.units import ConversionTable
from Training_data_generation_triangles.Rotations import rotation_matrix_xyz

class Triangular(dt.Scatterer):
    __conversion_table__ = ConversionTable(
        side_length=(u.meter, u.meter),
        thickness=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
            self,
            side_length=10e-6,
            thickness=2.5e-6,
            rotation=(0, 0, 0),
            refractive_index=None,
            **kwargs
        ):

        init_kwargs = dict(kwargs)
        init_kwargs["side_length"] = side_length
        init_kwargs["thickness"] = thickness
        init_kwargs["rotation"] = rotation

        if refractive_index is not None:
            init_kwargs["refractive_index"] = refractive_index

        super().__init__(**init_kwargs)
        
        # super().__init__(
        #     side_length=side_length,
        #     thickness=thickness,
        #     rotation=rotation,
        #     refractive_index=refractive_index,
        #     **kwargs,
        # )

    def _process_properties(self, properties):

        properties = super()._process_properties(properties)

    def _process_properties(self, properties):

        properties = super()._process_properties(properties)

        # Ensure side_length is scalar
        properties["side_length"] = float(np.array(properties["side_length"]))

        # Ensure thickness is scalar
        properties["thickness"] = float(np.array(properties["thickness"]))

        # Ensure rotation has length 3
        rot = np.array(properties["rotation"])

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

        R = side_length/np.sqrt(3) #R is centre to vertex distance, so furthest distance from centre

        # Determine grid size
        R_ceil = np.ceil(np.max(R) / np.min(voxel_size[:2])) #rounds to upper int needed to cover the object
        thickness_ceil = np.ceil(thickness * 0.5 / voxel_size[2])
        ceil = int(max(R_ceil, thickness_ceil))

        # Create grid in array-axis order:
        # axis 0 = y / image row
        # axis 1 = x / image column
        # axis 2 = z
        y = np.arange(-ceil, ceil) * voxel_size[0] 
        x = np.arange(-ceil, ceil) * voxel_size[1]
        z = np.arange(-ceil, ceil) * voxel_size[2]

        Y, X, Z = np.meshgrid(y, x, z, indexing="ij") #Need to be swapped since they are given to DT!

        # Rotate the grid
        rx, ry, rz = rotation

        Rmat = rotation_matrix_xyz(rx, ry, rz)

        # Grid/world -> object frame uses inverse rotation.
        Rt = Rmat.T

        XR = Rt[0, 0] * X + Rt[0, 1] * Y + Rt[0, 2] * Z
        YR = Rt[1, 0] * X + Rt[1, 1] * Y + Rt[1, 2] * Z
        ZR = Rt[2, 0] * X + Rt[2, 1] * Y + Rt[2, 2] * Z


        # ---- signed distances to the prism faces (positive inside) ----
        a = np.sqrt(3) / 3
        norm_oblique = np.sqrt(a**2 + 1)

        d_left  = XR + R / 2
        d_right = R - XR
        d_low   = (YR - (a * XR - a * R)) / norm_oblique
        d_high  = ((-a * XR + a * R) - YR) / norm_oblique
        d_z     = thickness / 2 - np.abs(ZR)

        # hard prism
        inside = (
            (d_left  >= 0) &
            (d_right >= 0) &
            (d_low   >= 0) &
            (d_high  >= 0) &
            (d_z     >= 0)
        )

        # ---- scrape only edges, not flat surfaces ----
        # thickness of bevel region
        edge_scrape = 1.0 * min(voxel_size)   # try 1 voxel first
        # edge_scrape = 2.0 * min(voxel_size) # stronger bevel

        close_to_face = np.stack([
            d_left  < edge_scrape,
            d_right < edge_scrape,
            d_low   < edge_scrape,
            d_high  < edge_scrape,
            d_z     < edge_scrape,
        ], axis=0)

        n_close = np.sum(close_to_face, axis=0)

        # near 2 or more faces => edge/corner region
        edge_region = inside & (n_close >= 2)

        # remove only that edge layer
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
