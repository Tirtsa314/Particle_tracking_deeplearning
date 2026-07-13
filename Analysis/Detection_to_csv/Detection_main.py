
#%%

import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Detection_to_csv")
print(os.getcwd())


#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import nd2
from pprint import pprint
import pandas as pd
from matplotlib.patches import Circle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.optimize import minimize
from pathlib import Path as FilePath
from matplotlib.path import Path as MplPath
from Model_detection import Model_detection
from Centering_detections import refine_all_particles_to_df
from Manual_corrections import make_refined_review_html
from removing import remove_marked_particles
from Centering_manual_corrections import refine_all_manual_particles_to_df
from merging_to_final import merge_df
from select_region import make_particle_roi_selection_html
import webbrowser
from final_removing import make_remove_particles_html, remove_particles_from_roi_df
from Model_detection import reading_nd2

#%%

frame , arr8 = reading_nd2(r"c:\Users\Public\Hydrazine 003(good).nd2", frame_index=-1)




#%%
""" 
Detect

results in df with ["particle_id", "x", "y", "poly_json"]
"""


model = YOLO(r"C:\Users\DenHaan\Downloads\best_09-04_20epoch_ratio1.pt")
nd2_path = r"c:\Users\Public\Hydrazine 003(good).nd2"
file_name = FilePath(nd2_path).stem

um_per_px=0.32
side_um=10.0
scale_side = side_um/10.0

det_df, image_frame, arr8 = Model_detection(nd2_path,
                    model, 
                    frame_index=-1, 
                    mpp_real = um_per_px, 
                    side_real_um = side_um,
                    confidence_threshold=0.05,
)


#%%
"""
Centering detections

results in df: [
    "particle_id",
    "x_old",
    "y_old",
    "poly_json",
    "x_refined",
    "y_refined",
    "theta_rad",
    "theta_deg",
    "side_px",
    "side_um",
    "thickness_px",
    "loss",
    "dark_score",
    "bright_score",
    "mean_outline_distance",
    "yolo_outside_fraction",
    "outer_vertices_json",
    "mid_vertices_json",
    "inner_vertices_json",
    "success",
    "error",
]
"""
detected_df = pd.read_csv(r"c:\particle_csv_files\Hydrazine 010.csv")

df_det_fitted = refine_all_particles_to_df(
    det_df,
    arr8,
    um_per_px=um_per_px,
    side_um=side_um,
    grid_thickness_um=0.5*scale_side,
    grid_band_extra_um=0.5*scale_side,
    grid_max_center_shift_um=8.0*scale_side,
    grid_center_step_px=1.0,
    grid_theta_step_deg=10.0,
    grid_blur_sigma=0.5,
    grid_dark_top_fraction=1.0,
    grid_bright_top_fraction=1.0,
    grid_w_dark=1.0,
    grid_w_bright=0.2,
    grid_max_yolo_outside_fraction=0.05,
    w_bright_fraction=2.0,

    refine_thickness_um=0.5*scale_side,
    refine_center_window_px=5.0,
    refine_theta_window_deg=15.0,
    refine_side_window_px=2.0,
    refine_blur_sigma=0.5,
    refine_dark_top_fraction=1.0,
    refine_bright_top_fraction=0.2,
    refine_w_dark=1.0,
    refine_w_bright=1.0,
    refine_max_yolo_outside_fraction=0.05,
    refine_maxiter=120,
    pixel_scale=um_per_px,

    verbose=True,
)
#%%


"""
Clicking missed particles and removing remove

results in df: [
    "missing_x",
    "missing_y",
    "remove_particle_id",
    "remove_x",
    "remove_y",
]
"""

html_path = make_refined_review_html(
    image=arr8,
    df_det_fitted=df_det_fitted,
    output_html="review_refined_triangles.html",
    output_csv_name=f"{nd2_path}_refined_triangle_fitsmanual_triangle_edits.csv",
    success_only=True,
)

print(html_path)
webbrowser.open(FilePath(html_path).as_uri())


manual_edits_csv = f"Downloads{file_name}_refined_triangle_fitsmanual_triangle_edits.csv"
manual_corrections_df = remove_marked_particles(df_det_fitted, manual_edits_csv, id_col="particle_id")
#%%

"""
Centering the manual clicks

with df: [
    "x_manual_refined",
    "y_manual_refined",
    "theta_rad",
    "theta_deg",
    "side_px",
    "side_um",
    "thickness_px",
    "loss",
    "dark_score",
    "bright_score",
    "outer_vertices_json",
    "mid_vertices_json",
    "inner_vertices_json",
    "success",
    "error"
]

"""   

#CHANGE THIS WHITH FURTURE FILES:
manual_corrections_df = pd.read_csv(r"c:\particle_csv_files\Hydrazine 010_refined_triangle_fitsmanual_triangle_edits.csv")

manual_corrected_df = refine_all_manual_particles_to_df(
                manual_corrections_df,
                arr8,
                um_per_px=um_per_px,
                side_um=side_um,
                grid_thickness_um=0.5*scale_side,
                grid_band_extra_um=0.5*scale_side,
                grid_max_center_shift_um=4.0*scale_side,
                grid_center_step_px=1.0,
                grid_theta_step_deg=10.0,
                grid_blur_sigma=0.5,
                grid_dark_top_fraction=1.0,
                grid_bright_top_fraction=1.0,
                grid_w_dark=1.0,
                grid_w_bright=0.2,
                w_bright_fraction=2.0,

                pixel_scale=um_per_px,

                verbose=True,
        )
#%%
manual_corrected_df.to_csv(r"c:\particle_csv_files\Hydrazine 010_manual_corrected.csv", index=False)

#%%

df_det_fitted = pd.read_csv(r"c:\particle_csv_files\Hydrazine 010_refined_triangle_fits.csv")
"""
Merged code

results: [
    "particle_id",
    "poly_json",
    "theta_rad",
    "theta_deg",
    "side_px",
    "side_um",
    "thickness_px",
    "outer_vertices_json",
    "mid_vertices_json",
    "inner_vertices_json",
    "x",
    "y",
]


"""

final_df = merge_df(df_det_fitted, manual_corrected_df)

#%%

"""
Select a region
"""
output_dir = FilePath(r"C:\particle_csv_files")

roi_html = make_particle_roi_selection_html(
    image=arr8,
    det_df=final_df,
    output_html=output_dir / f"{nd2_path}_select_roi.html",
    output_csv_name=f"{nd2_path}_selected_particles.csv",
    output_roi_csv_name=f"{nd2_path}_roi_vertices.csv",
    x_col="x",
    y_col="y",
    um_per_px=um_per_px,
    point_radius=2.0,
)

"""
TODO
make reusable names in the saved files,

""" 
# %%
ROI_df = pd.read_csv(r"c:\Analysis_images\Hydrazine 010\Hydrazine_010_final_ROI(1).csv")

html_path = make_remove_particles_html(
    image=arr8,
    roi_df=ROI_df,
    output_html=r"C:\Analysis_images\Hydrazine 010\remove_particles.html",
    output_csv_name="Hydrazine_010_particles_to_remove.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    polygon_col="mid_vertices_json",   # or "outer_vertices_json", or None
    point_radius=4,
)

webbrowser.open(FilePath(html_path).resolve().as_uri())


#%%
# %%
"""
Apply manual removals
"""

final_roi_df = remove_particles_from_roi_df(
    roi_df_or_csv=("c:\Analysis_images\Hydrazine 010\Hydrazine_010_final_ROI(1).csv"),
    removal_csv=r"c:\Users\DenHaan\Downloads\Hydrazine_010_particles_to_remove (1).csv",
    id_col="particle_id",
    x_col="x",
    y_col="y",
    save_path=r"C:\Analysis_images\Hydrazine 010\Hydrazine_010_final_ROI(2).csv",
)
# %%
