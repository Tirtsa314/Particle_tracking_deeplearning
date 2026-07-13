#%%

import os

import cv2

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images")
print(os.getcwd())

from loading_image import reading_nd2
#%%

import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Voronoi_graph_df")
print(os.getcwd())

#%%
from voronoi_constructions import compute_voronoi_from_df, plot_voronoi_cells, make_interactive_voronoi_html, plot_voronoi_edge_histogram
from neighbour_graph_from_voronoi import make_voronoi_neighbor_graph, plot_voronoi_neighbor_graph, make_interactive_voronoi_graph_html, plot_voronoi_cutout_um
import pandas as pd
import webbrowser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

#%%

csv_path = r"c:\Analysis_images\Hydrazine 010\final_particles_voronoi.csv"
image_path =  r"c:\Users\Public\Hydrazine 010.nd2"
# image_path = r"c:\Users\Public\Hydrazine 003(good).nd2"


frame,arr8 = reading_nd2(image_path, frame_index=-1)

det_df = pd.read_csv(csv_path)


min_edge_length = 5.0

#%%

def save_nd2_crop_as_png(
    arr8,
    output_png,
    x0,
    y0,
    crop_w,
    crop_h,
):
  

    H, W = arr8.shape

    # Make sure crop stays inside image
    x0 = int(x0)
    y0 = int(y0)
    x1 = min(W, x0 + int(crop_w))
    y1 = min(H, y0 + int(crop_h))

    x0 = max(0, x0)
    y0 = max(0, y0)

    crop8 = arr8[y0:y1, x0:x1]

    ok = cv2.imwrite(str(output_png), crop8)


    print("Saved:", output_png)
    print("crop shape:", crop8.shape)
    print("dtype:", crop8.dtype)
    print("min/max:", crop8.min(), crop8.max())

    return crop8

crop = save_nd2_crop_as_png(
    arr8=arr8,
    output_png=r"C:\Analysis_images\crop_test.png",
    x0=976,
    y0=793,
    crop_w=106,
    crop_h=106,

)

#%%
"""
In this code I make voronoi edges and graph of the neighbours from voronoi
"""


vor_df, vor, regions, vertices = compute_voronoi_from_df(
    det_df,
    x_col="x",
    y_col="y",
    image_shape=arr8.shape,
    um_per_px=0.65,
    min_edge_length_um=min_edge_length
)

# plot_voronoi_cells(
#     arr8,
#     vor_df,
#     x_col="x",
#     y_col="y",
#     value_col="voronoi_num_edges",
# )
# %%
html_path = make_interactive_voronoi_html(
    image=arr8,
    vor_df=vor_df,
    output_html=r"C:\Analysis_images\Hydrazine 010\interactive_voronoi.html",
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    value_col="voronoi_num_edges",
    robust_percentiles=(2, 98),
    point_radius=2.5,
    alpha=0.35,
    um_per_px=0.65,
    scale_bar_um=50,
    scale_bar_corner="bottom-left",
)

webbrowser.open(Path(html_path).resolve().as_uri())


# %%
# """ 
# Histogram of number of edges per Voronoi cell
# """
# counts, fig, ax = plot_voronoi_edge_histogram(
#     vor_df,
#     edge_col="voronoi_num_edges",
#     image_shape=arr8.shape,
#     exclude_border_cells=True,
#     border_margin_px=20,
# )
# # %%

"""
Making graph of edges
"""
# %%
G, graph_df = make_voronoi_neighbor_graph(
    vor_df,
    vor,
    x_col="x",
    y_col="y",
    exclude_border_cells=False,
    min_edge_length_um=min_edge_length,
    um_per_px=0.65,
)

# print(G)
# print(graph_df[["x", "y", "voronoi_num_edges", "voronoi_graph_coord_num"]].head())


# # %%
# output_dir = Path(r"C:\Analysis_images\Hydrazine 010")

# html_path_graph = make_interactive_voronoi_graph_html(
#     image=arr8,
#     graph_df=graph_df,
#     G=G,
#     output_html=output_dir / "Hydrazine_010_interactive_voronoi_graph.html",
#     x_col="x",
#     y_col="y",
#     poly_col="voronoi_poly",
#     value_col="voronoi_num_edges",
#     cmap_name="tab20",
#     alpha=0.35,
#     point_radius=2.5,
#     edge_line_width=0.8,
#     edge_alpha=1.0,
# )

# webbrowser.open(Path(html_path_graph).resolve().as_uri())
# # %%
graph_df.to_csv(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Voronoi_graph_df\voronoi_dataframes\voronoi_graph_hydrazine010_improved.csv", index=False)
# %%


import pickle
from pathlib import Path

graph_folder = Path(
    r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Voronoi_graph_df\voronoi_dataframes"
)

graph_folder.mkdir(parents=True, exist_ok=True)

with open(graph_folder / "voronoi_graph_hydrazine010_improved.pkl", "wb") as f:
    pickle.dump(G, f)
# %%

x_shift = -3
y_shift = 0

fig, ax = plot_voronoi_cutout_um(
    image=arr8,
    graph_df=graph_df,
    G=G,
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    value_col="voronoi_num_edges",
    um_per_px=0.32,
    # xlim_px=(222+x_shift, 222+x_shift+25),
    # ylim_px=(506+y_shift, 506+y_shift+25),
    # xlim_px=(185, 585),
    # ylim_px=(375, 675),
    xlim_px = (0, 1600),
    ylim_px = (280, 1600),
    show_image=True,
    show_voronoi=True,
    show_graph=False,
    show_centers=False,
    voronoi_alpha=0.35,
    voronoi_edge_color="white",
    voronoi_edge_width=0.15,
    graph_color="black",
    graph_line_width=3.0,
    graph_alpha=1,
    center_size=6,
    figsize=(30, 30),
    # title=f"",
    # title=f"Neighbour graph with Voronoi construction: minimum edge length of ${min_edge_length} \\, \\mu m$",
    show_legend=False,
    legend_loc="center left",
    legend_bbox_to_anchor=(1.02, 0.5),
    cell_alpha=0.30,
    legend_alpha=0,
    scale_bar_fontsize=30,
    scale_bar_height_um=1,
    axis_label_fontsize=35,
    tick_fontsize=30,
    title_fontsize=35,
)


# %%
