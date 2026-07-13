""" 
author: Tirtsa den Haan 
06-07-2026
"""
import deeptrack as dt
import numpy as np
from deeptrack import units as u
from deeptrack.backend.units import ConversionTable
from Training_data_generation_triangles_structured.Rotations import rotation_matrix_xyz

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
        

    def _process_properties(self, properties):

        properties = super()._process_properties(properties)
        properties["side_length"] = float(np.array(properties["side_length"]))
        properties["thickness"] = float(np.array(properties["thickness"]))

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
        # axis 0 = y  image row
        # axis 1 = x  image column
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

        edge_scrape = 1.0 * min(voxel_size)  
   

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


