""" 
author: Tirtsa den Haan 
06-07-2026

Main analysis function
"""


#%%

import pickle
import pandas as pd
import webbrowser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images")

from radial_density import make_theoretical_cluster_by_df_shape, average_radial_profile_from_selected_particles_full_circle, average_radial_profile_from_selected_particles


#%%
triangle_side_um = 10.7
rin_um = triangle_side_um * np.sqrt(3) / 6
spacing_um = 2 * rin_um
#%%
""" 


RADIAL DENSITY


 """


hex_theory_df = make_theoretical_cluster_by_df_shape(
    real_df=particle_position_df,
    spacing_px=spacing_um / 0.65,
    image_shape=(2048, 2048),
    alpha=0.02,
    return_shape=True,
)


um_per_px=0.65
dr_um=0.5
r_max_um=400.0

avg_profile_df_real = average_radial_profile_from_selected_particles(
        real_df=particle_position_df,
        x_col="x",
        y_col="y",
        um_per_px=um_per_px,
        dr_um=dr_um,
        r_max_um=r_max_um,
    )
#%%

avg_profile_df_theo = average_radial_profile_from_selected_particles(
        hex_theory_df,
        x_col="x",
        y_col="y",
        um_per_px=um_per_px,
        dr_um=dr_um,
        r_max_um=r_max_um,
    )

#%%

avg_profile_df_real_full = average_radial_profile_from_selected_particles_full_circle(
        real_df=particle_position_df,
        x_col="x",
        y_col="y",
        um_per_px=um_per_px,
        dr_um=dr_um,
        r_max_um=r_max_um,
    )

#%%
avg_profile_df_theo_full = average_radial_profile_from_selected_particles_full_circle(
        hex_theory_df,
        x_col="x",
        y_col="y",
        um_per_px=um_per_px,
        dr_um=dr_um,
        r_max_um=r_max_um,
    )
#%%



VORONOI CONSTRUCTIONS



