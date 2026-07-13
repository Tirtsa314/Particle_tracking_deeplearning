
#%%

import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images")
print(os.getcwd())

#%%
from loading_image import reading_nd2
import pandas as pd
import webbrowser
from pathlib import Path

#%%
import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\add_remove")
print(os.getcwd())

#%%
from add_remove import make_add_particles_html, add_manual_particles, make_remove_particles_html, remove_marked_particles, make_edit_particles_html, apply_particle_edits
from add_remove import remove_too_close_particles

#%%
"""
Here I can add and remove missed particles by the model




"""

csv_path = r"c:\Analysis_images\Hydrazine 010\final_particles_voronoi.csv"
image_path =  r"c:\Users\Public\Hydrazine 010.nd2"

frame,arr8 = reading_nd2(image_path, frame_index=-1)

det_df = pd.read_csv(csv_path)
# df_removed = pd.read_csv(csv_path)

#%%

"""
ADDING AND REMOVING PARTICLES

"""
# df_initial = pd.read_csv(r"c:\Analysis_images\Hydrazine 010\Hydrazine_010_final_ROI(2).csv")
df_initial = pd.read_csv(csv_path)

html_path_remove = make_remove_particles_html(
    arr8,
    df_initial,
    output_html=r"C:\Analysis_images\Hydrazine 010\remove_particles.html",
    output_csv_name="particles_to_remove.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    polygon_col=None,
    point_radius=4,
)
webbrowser.open(Path(html_path_remove).resolve().as_uri())


#%%
manual_remove_csv = r"c:\Users\DenHaan\Downloads\particles_to_remove (3).csv"
df_removed =remove_marked_particles(df_initial, manual_remove_csv, id_col="particle_id")

#%%

html_path_add = make_add_particles_html(
    arr8,
    df_removed,
    output_html=r"C:\Analysis_images\Hydrazine 010\add_particles.html",
    output_csv_name="particles_to_add.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    point_radius=4,
)

webbrowser.open(Path(html_path_add).resolve().as_uri())

#%%

manual_add_csv = r"c:\Users\DenHaan\Downloads\particles_to_add (3).csv"
df_added, added_only_df = add_manual_particles(
    df_removed,
    manual_add_csv,
    id_col="particle_id",
)



#%%
df_clean, removed_df, close_pairs_df = remove_too_close_particles(df_added,
    min_dist_um=0.2,
    um_per_px=0.06,
    x_col="x",
    y_col="y",
    id_col="particle_id",
    prefer_original=True,
)


df_clean.to_csv(
    r"c:\Analysis_images\Hydrazine 010\final_particles_voronoi.csv",
    index=False
)



# %%

df_initial = pd.read_csv(csv_path)

html_path_edit = make_edit_particles_html(
    arr8,
    df_initial,
    output_html=r"C:\Analysis_images\Hydrazine 010\edit_particles.html",
    output_edits_csv_name="particle_edits.csv",
    x_col="x",
    y_col="y",
    id_col="particle_id",
    polygon_col=None,
    point_radius=4,
)

webbrowser.open(Path(html_path_edit).resolve().as_uri())

#%%
manual_edits_csv = r"c:\Users\DenHaan\Downloads\particle_edits.csv"

df_clean, added_df, duplicate_removed_df, close_pairs_df = apply_particle_edits(
    df_initial,
    manual_edits_csv,
    x_col="x",
    y_col="y",
    id_col="particle_id",
    min_dist_um=0.2,
    um_per_px=0.06,
    prefer_original=True,
    save_path=r"c:\Analysis_images\Hydrazine 010\final_particles_voronoi.csv",
)
# %%
