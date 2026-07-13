""" 
author: Tirtsa den Haan 
06-07-2026
"""
print("Image_generator.py started")             
import numpy as np
import deeptrack as dt
from deeptrack import units as u
from Training_data_generation_triangles.Image_creation.Triangular_Scatterer_Class import Triangular
from deeptrack.aberrations import SphericalAberration
from deeptrack.aberrations import GaussianApodization

def generate_image(
    pos_sampler,
    rot_sampler,
    N=20,
    um_per_pixel=0.32,
    image_size=1024,
    side_length_um=10.0,
    thickness_um=2.5,
    noise_sigma=0.01, 
    scale=2.0
    ):


    triangle = Triangular(
        side_length=side_length_um * u.um,
        thickness=thickness_um * u.um,
        rotation=rot_sampler,
        refractive_index = 1.49 + np.random.uniform(0, 0.001) * 1j,
        position=pos_sampler,
        z= np.random.uniform(-0.5, 0.5) * u.um, #in micrometer
        class_id=1,
        is_triangle=1,
    )
    
    triangle.store_properties()
    triangle.voxel_size = (um_per_pixel * u.um,
                       um_per_pixel * u.um,
                       um_per_pixel * u.um)

    sample = triangle ^ N

    spectrum = np.linspace(400e-9, 700e-9, 10)

    g0 = 2e-5
    illumination_conditions = [
        ((0.0, 0.0), 1.00),
        ((+g0, 0.0), 1.00),
        ((-g0, 0.0), 1.00),
        ((0.0, +g0), 1.00),
        ((0.0, -g0), 1.00),
    ]

    renders = []
    scale = int(scale)

    for wavelength in spectrum:
        for (gx, gy), c in illumination_conditions:
            illum = dt.IlluminationGradient(
                gradient=(gx, gy),
                constant=c,
            )

            bf = dt.Brightfield(
                wavelength=wavelength,
                NA=0.75,
                resolution=um_per_pixel * u.um,
                magnification=scale,
                refractive_index_medium=1.33,
                illumination=illum,
                output_region=(0, 0, image_size, image_size),
                pupil = SphericalAberration(coefficient=0.2) >> GaussianApodization(sigma=1.8),
            )

            renders.append(bf(sample))

    
    incoherently_illuminated_sample = renders[0]
    for r in renders[1:]:
        incoherently_illuminated_sample = incoherently_illuminated_sample + r
    incoherently_illuminated_sample = incoherently_illuminated_sample / len(renders)

    noise = dt.Gaussian(sigma=noise_sigma)

    image_of_particles = (
        incoherently_illuminated_sample
        >> dt.AveragePooling((scale, scale, 1))
        >> noise
        >> dt.GaussianBlur(sigma=1.1)
    )

    image_of_particles.store_properties()
    
    return image_of_particles

