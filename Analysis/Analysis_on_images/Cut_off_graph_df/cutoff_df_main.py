#%%

import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images")
print(os.getcwd())


from loading_image import reading_nd2
#%%

import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Cut_off_graph_df")
print(os.getcwd())

#%%


import pandas as pd
import webbrowser
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cutoff_graph import compute_cutoff_graph_from_df, plot_cutoff_graph_cutout_um

#%%

csv_path = r"c:\Analysis_images\Hydrazine 010\final_particles_voronoi.csv"
image_path =  r"c:\Users\Public\Hydrazine 010.nd2"

frame,arr8 = reading_nd2(image_path, frame_index=-1)

det_df = pd.read_csv(csv_path)


cutoff_um = 9.0

#%%

""" 
Making a graph using cutoff distance

"""

G, graph_df = compute_cutoff_graph_from_df(
    det_df,
    cutoff_um,
    um_per_px=0.65,
    x_col="x",
    y_col="y",
    particle_id_col=None,
)

#%%



fig, ax = plot_cutoff_graph_cutout_um(
    image=arr8,
    graph_df=graph_df,
    G=G,
    x_col="x",
    y_col="y",
    value_col="cutoff_graph_coord_num",
    um_per_px=0.65,
    xlim_px=(185, 585),
    ylim_px=(375, 675),
    show_image=True,
    show_graph=True,
    show_centers=False,
    color_centers_by_value=False,
    graph_color="cyan",
    graph_line_width=1.0,
    graph_alpha=0.8,
    center_size=8,
    title=f"Neighbours graph based on cut off distancen with cut off distance ${cutoff_um} \\, \\mu m$",
)





# #%%

graph_df.to_csv(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Cut_off_graph_df\cut_off_dataframes\cutoff_{cutoff_um}_graph_hydrazine010_improved.csv", index=False)
# %%


import pickle
from pathlib import Path

graph_folder = Path(
    r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Cut_off_graph_df\cut_off_dataframes"
)

graph_folder.mkdir(parents=True, exist_ok=True)

with open(graph_folder / f"cutoff_{cutoff_um}_graph_hydrazine010_improved.pkl", "wb") as f:
    pickle.dump(G, f)

# %%
