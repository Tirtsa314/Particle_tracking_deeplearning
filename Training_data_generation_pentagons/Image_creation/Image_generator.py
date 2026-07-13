
# %%   
print("Image_generator.py started")             
import numpy as np
import deeptrack as dt
from deeptrack import units as u
from Training_data_generation_pentagons.Image_creation.Pentagonal_Scatterer_Class import Pentagonal
from Training_data_generation_pentagons.Image_creation.particle_position.cluster_position import cluster_positions
from deeptrack.aberrations import Aberration
from deeptrack.aberrations import SphericalAberration
from deeptrack.aberrations import GaussianApodization

def generate_image(
    sampler,
    pos_sampler,
    rot_sampler,
    N=20,
    um_per_pixel=0.32,
    image_size=1024,
    side_length_um = 7.0,
    thickness_um = 2.0,
    noise_sigma=0.0048, return_parts=False,
    scale=2.0
    ):


    particle = Pentagonal(
        side_length=side_length_um * u.um,
        thickness=thickness_um * u.um,
        rotation=rot_sampler,
        refractive_index=1.49 + 0.001j,
        position=pos_sampler,
        z=0.0 * u.um,
        class_id=0,
        is_pentagon=1,
    )
    
    particle.store_properties()

    particle.voxel_size = (
        um_per_pixel * u.um,
        um_per_pixel * u.um,
        um_per_pixel * u.um,
    )

    sample = particle ^ N

    spectrum = np.linspace(400e-9, 700e-9, 10)

    g0 = 2e-5
    illumination_conditions = [
        ((0.0, 0.0), 1.00),
        ((+g0, 0.0), 1.00),
        ((-g0, 0.0), 1.00),
        ((0.0, +g0), 1.00),
        ((0.0, -g0), 1.00),
    ]

    combined_pupil = (
    dt.GaussianApodization(sigma=2.8)
    >> dt.Defocus(coefficient=-3.0)
    )

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
                upsample=1,
                output_region=(0, 0, image_size, image_size),
                # padding=(30, 30, 30, 30),
                pupil=combined_pupil
                # pupil=SphericalAberration(coefficient=0.3),
            )

            renders.append(bf(sample))

    
    incoherently_illuminated_sample = renders[0]
    for r in renders[1:]:
        incoherently_illuminated_sample = incoherently_illuminated_sample + r
    incoherently_illuminated_sample = incoherently_illuminated_sample / len(renders)

    # wavelengths: this part you already had
    # spectrum = np.linspace(400e-9, 700e-9, 3)

    # small set of illumination-field conditions
    # this is NOT true angular illumination,
    # but it can soften directional-looking contrast



    
    # pip = optics(sample)
    noise = dt.Gaussian(sigma=noise_sigma)
    # offset = dt.Add(value=lambda: np.random.uniform(0.0, 0.2))

    image_of_particles = (
        # pip
        incoherently_illuminated_sample
        # >> offset
        >> dt.AveragePooling((scale, scale, 1))
        >> noise
        >> dt.GaussianBlur(sigma=1.5)
    #    >> dt.NormalizeMinMax()
    )

    # IMPORTANT: store properties on the final pipeline
    image_of_particles.store_properties()
    
    return image_of_particles

# %%
