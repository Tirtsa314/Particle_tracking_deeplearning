


#%%


import os

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images")
print(os.getcwd())

#%%
from loading_image import reading_nd2, save_image
# from neighbour_graph_from_voronoi import make_voronoi_neighbor_graph, plot_voronoi_neighbor_graph
from order_parameter import compute_order_parameter_graph,plot_coordination_vs_average_order_parameter, plot_coordination_number_histogram, plot_order_parameter_graph_map, plot_order_parameter_graph_map_on_image, plot_order_parameter_histogram, make_interactive_voronoi_graph_order_html, plot_order_parameter_cutout_um,plot_valid_order_parameter_values_line, plot_valid_order_parameter_histogram_colored
from voronoi_orderparameter_overlay import make_interactive_voronoi_order_alpha_html
from vor_par_with_centers import make_interactive_voronoi_order_centers_html
from radial_density import plot_radial_density, make_perfect_hexagonal_df, average_radial_profile_from_selected_particles_full_circle, average_radial_profile_from_selected_particles, make_hexagonal_image_selected_by_alpha_shape
import pickle
import pandas as pd
import webbrowser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

"""
- Load selected particles
- Voronoi constructions
- histogram of number of edges per Voronoi cell
- Making graph of edges
- coordination number
- order parameter
- radial density
"""

#%%

os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Voronoi_graph_df\voronoi_dataframes")
print(os.getcwd())

voronoi_graph_df = pd.read_csv(r"voronoi_graph_hydrazine010_improved.csv")



with open("voronoi_graph_hydrazine010_improved.pkl", "rb") as f:
    G_voronoi = pickle.load(f)

#%%
# cutoff_um = 5.0


os.chdir(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images\Cut_off_graph_df\cut_off_dataframes")
print(os.getcwd())

cutoff_graph_df = pd.read_csv(r"cutoff_{cutoff_um}_graph_hydrazine010_improved.csv")

with open(f"cutoff_9.0_graph_hydrazine010_improved.pkl", "rb") as f:
    G_cutoff = pickle.load(f)

#%%
"""
Loading selected particles
"""
csv_path = r"c:\Analysis_images\Hydrazine 010\final_particles_voronoi.csv"
image_path = r"c:\Users\Public\Hydrazine 010.nd2"
det_df = pd.read_csv(csv_path)
frame, arr8 = reading_nd2(image_path, frame_index=-1)

#%%
save_image(arr8, r"C:\Analysis_images\image_hydrazine010_lastframe.png")

#%%
triangle_side_um = 10.7
rin_um = triangle_side_um * np.sqrt(3) / 6
spacing_um = 2 * rin_um
#%%

hex_theory_df, alpha_shape = make_hexagonal_image_selected_by_alpha_shape(
    real_df=det_df,
    spacing_px=spacing_um / 0.65,          
    image_shape=(2048, 2048),
    alpha=0.02,
    return_shape=True,
)

#%%

import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.scatter(hex_theory_df["x"], hex_theory_df["y"], s=0.5, color="red")

plt.xlim(700, 1400)
plt.ylim(1400, 700)
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.title("Theoretical particle centres")
plt.show()

#%%
plt.figure(figsize=(5, 5))
#
# show original image
# plt.imshow(arr8, cmap="gray")

# overlay real particle centres
plt.scatter(det_df["x"], det_df["y"], s=0.5, color="red")

# same crop as your theoretical plot
plt.xlim(700, 1400)
plt.ylim(1400, 700)
plt.title("Real particle centres")

plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.show()



#%%
""" 
- radial density
- load the graph_df's 
- coordination number
- order parameter

  """
um_per_px=0.65
dr_um=0.5
r_max_um=400.0

avg_profile_df_real = average_radial_profile_from_selected_particles(
        det_df,
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
        det_df,
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
from pathlib import Path

output_dir = Path(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images")
output_dir.mkdir(parents=True, exist_ok=True)

avg_profile_df_real.to_csv(output_dir / "avg_profile_real_cluster_area.csv", index=False)
avg_profile_df_theo.to_csv(output_dir / "avg_profile_theo_cluster_area.csv", index=False)

avg_profile_df_real_full.to_csv(output_dir / "avg_profile_real_full_circle.csv", index=False)
avg_profile_df_theo_full.to_csv(output_dir / "avg_profile_theo_full_circle.csv", index=False)

#%%
output_dir = Path(r"C:\Particle_Tracking_Deeplearning_Tirtsa\Analysis\Analysis_on_images")

avg_profile_df_real = pd.read_csv(output_dir / "avg_profile_real_cluster_area.csv")
avg_profile_df_theo = pd.read_csv(output_dir / "avg_profile_theo_cluster_area.csv")
avg_profile_df_real_full = pd.read_csv(output_dir / "avg_profile_real_full_circle.csv")
avg_profile_df_theo_full = pd.read_csv(output_dir / "avg_profile_theo_full_circle.csv")


def add_moving_average(df, y_col="g_r", window=30):
    df = df.copy()
    df[f"{y_col}_smooth"] = (
        df[y_col]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    return df

# x-axis used in your plot: r / 2a
x = avg_profile_df_real["r_um"].to_numpy() / spacing_um

y_real = avg_profile_df_real["g_r"].to_numpy()
y_theo = avg_profile_df_theo["g_r"].to_numpy()


# find local peaks
peaks, props = find_peaks(
    y_real,
    prominence=0.1,   # increase if it detects too many small wiggles
    distance=2        # minimum number of bins between peaks
)

# sort peaks by peak height, highest first
top2_peaks = peaks[np.argsort(y_real[peaks])[-2:]][::-1]

print("Two highest peaks:")
for i, peak_idx in enumerate(top2_peaks, start=1):
    print(
        f"Peak {i}: "
        f"x = {x[peak_idx]:.3f} in r/2a, "
        f"r = {avg_profile_df_real['r_um'].iloc[peak_idx]:.3f} µm, "
        f"g(r) = {y_real[peak_idx]:.3f}"
    )

# find local peaks
peaks, props = find_peaks(
    y_theo,
    prominence=0.1,   # increase if it detects too many small wiggles
    distance=2        # minimum number of bins between peaks
)

# sort peaks by peak height, highest first
top2_peaks = peaks[np.argsort(y_theo[peaks])[-2:]][::-1]

print("Two highest peaks:")
for i, peak_idx in enumerate(top2_peaks, start=1):
    print(
        f"Peak {i}: "
        f"x = {x[peak_idx]:.3f} in r/2a, "
        f"r = {avg_profile_df_theo['r_um'].iloc[peak_idx]:.3f} µm, "
        f"g(r) = {y_theo[peak_idx]:.3f}"
    )

r_max_um = 400
fig, ax = plt.subplots(figsize=(14, 14))

# ax.plot((avg_profile_df_theo["r_um"])/spacing_um, avg_profile_df_theo["g_r"], color = "red", label="theoretical hexagonal packing")
avg_profile_df_theo_smooth = add_moving_average(avg_profile_df_theo, window=10)

# raw theoretical curve, faint
plt.plot(
    avg_profile_df_theo["r_um"]/spacing_um,
    avg_profile_df_theo["g_r"],
    color="green",
    alpha=0.8,
    linewidth=0.5,
    label="theoretical hexagonal packing, ideal"
)

# smoothed theoretical curve
plt.plot(
    avg_profile_df_theo["r_um"]/spacing_um,
    avg_profile_df_theo_smooth["g_r_smooth"],
    color="red",
    linewidth=1.5,
    label="theoretical hexagonal packing, smoothed"
)
ax.plot((avg_profile_df_real["r_um"])/spacing_um, avg_profile_df_real["g_r"], color = "blue", label="real particles")


ax.set_xlabel("r/2a", fontsize=20)
ax.set_ylabel("g(r)", fontsize=20)
# ax.set_title(f"Average g(r) with full circles", fontsize=20)
ax.set_title(f"Average g(r) with circles only in the cluster area", fontsize=20)
ax.grid(True, alpha=0.6)
ax.set_xlim(0, r_max_um/(spacing_um))
ax.set_ylim(0, 4)
ax.tick_params(axis="both", labelsize=16)
ax.legend(fontsize=16)
plt.show()











#%%
plot_radial_density(         
    um_per_px=0.65,
    dr_um=0.5,
    r_max_um=400.0,
    selected_df = det_df,
    output_dir = Path(r"c:\Analysis_images\Hydrazine 010")
    )


#%%



hex_df = make_perfect_hexagonal_df(
    n_rows=50,
    n_cols=50,
    spacing_um=spacing_um,
    um_per_px=0.65,
)

#%%

hex_profile_df = average_radial_profile_from_selected_particles(
    selected_df=hex_df,
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    dr_um=0.25,
    r_max_um=200.0,
)


plt.figure(figsize=(6, 4))

plt.plot(
    hex_profile_df["r_um"] / spacing_um,
    hex_profile_df["g_r"],
    label="perfect hexagonal packing"
)

plt.xlabel("r / 2a")
plt.ylabel("g(r)")
plt.title("Theoretical perfect hexagonal packing")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()







#%%
order_n = 3


order_df = compute_order_parameter_graph(
    voronoi_graph_df,
    G,
    order_n=order_n,
    x_col="x",
    y_col="y",
    particle_id_col=None
)


#%%


plot_order_parameter_histogram(
    order_df,
    order_n=order_n,
    bins=40,
    output_path=r"C:\Analysis_images\Hydrazine 010\psi_3_histogram_nowimproved_voronoi_edges.png",
)




#%%



plot_order_parameter_graph_map(
    order_df,
    order_n=order_n,
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    s=4,
    n_bins=10,
    image_shape=arr8.shape,
    output_path=r"C:\Analysis_images\Hydrazine 010\psi{order_n}_graph_map_improved_voronoi.png",
)






#%%

html_path = make_interactive_voronoi_graph_order_html(
    image=arr8,
    graph_df=order_df,
    G=G,
    output_html=r"C:\Analysis_images\Hydrazine 010\hydrazine010_voronoi_graph_psi3.html",
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    value_col="voronoi_num_edges",
    psi_col="psi_3",
    point_radius=3.5,
)

print(html_path)

webbrowser.open(Path(html_path).resolve().as_uri())





# %%

order_n = 3


order_df = compute_order_parameter_graph(
    voronoi_graph_df,
    G,
    order_n=order_n,
    x_col="x",
    y_col="y",
    particle_id_col=None
)

#%%




































#%%




"""
PLOTTING ORDER PARAMETER ON IMAGE

"""
order_n = 3

order_df = compute_order_parameter_graph(
    df=vor_df,
    G=G,
    order_n=order_n,
    x_col="x",
    y_col="y",
    particle_id_col=None,   
)

plot_order_parameter_graph_map_on_image(
    image=arr8,
    order_df=order_df,
    order_n=order_n,
    x_col="x",
    y_col="y",
    s=10,
    n_bins=10,
    image_shape=arr8.shape,
    x_frac=(0, 1),
    y_frac=(0, 1),
    output_path=output_dir / f"Hydrazine_010_psi{order_n}_on_image.png",
)
# %%


html_voronoi_orderpar = make_interactive_voronoi_order_alpha_html(
    arr8,
    vor_df=graph_order_df,
    output_html="interactive_voronoi_order_alpha.html",
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    psi_col=f"psi_{order_n}",
    color_value_col="voronoi_num_edges",
    cmap_name="tab20",
    min_alpha=0.0,
    max_alpha=1.0,
    point_radius=2.5,
)
# %%
webbrowser.open(Path(html_voronoi_orderpar).resolve().as_uri())
# %%



html_voronoi_order_centers = make_interactive_voronoi_order_centers_html(
    image=arr8,
    vor_df=graph_order_df,
    output_html=r"C:\Analysis_images\Hydrazine 010\interactive_voronoi_order_centers.html",
    x_col="x",
    y_col="y",
    poly_col="voronoi_poly",
    psi_col=f"psi_{order_n}",
    cell_value_col="voronoi_num_edges",
    cell_cmap_name="tab20",   # palette for cells
    center_cmap_name="YlOrRd",  # different palette for centers
    cell_alpha=0.30,
    point_radius=3.5,
    n_center_bins=10,
)

webbrowser.open(Path(html_voronoi_order_centers).resolve().as_uri())
# %%
order_n = 3



"""
    G_voronoi , voronoi_graph_df

    G_cutoff , cutoff_graph_df


"""
setting = "cutoff"  # or "cutoff"
required_coord_num = 3


G = G_voronoi if setting == "voronoi" else G_cutoff
df = voronoi_graph_df if setting == "voronoi" else cutoff_graph_df


order_df = compute_order_parameter_graph(
    df,
    G,
    order_n=order_n,
    x_col="x",
    y_col="y",
    particle_id_col=None,
    required_coord_num=required_coord_num,
)
# %%


fig, ax = plot_order_parameter_cutout_um(
    image=arr8,
    graph_df=order_df,
    G=G,
    x_col="x",
    y_col="y",
    psi_col=f"psi_{order_n}",
    um_per_px=0.65,
    xlim_px=(185, 585),
    ylim_px=(375, 675),
    # xlim_px = (0, 2048),
    # ylim_px = (2048, 0),
    show_image=True,
    show_graph=False,
    show_centers=True,
    graph_color="black",
    graph_line_width=3,
    graph_alpha=1,
    center_size=200,
    center_cmap="plasma",
    center_edge_color="black",
    center_edge_width=1,
    figsize=(20, 20),
    # title=rf"Order parameter $\psi_{order_n}$ with {setting} neighbours",
    title=rf"$\psi_{order_n}$ with {setting}, only {required_coord_num} neighbours",
    # title=rf"Neighbour graph with cut-off distance 9.0$\mu m$",
    # title="Microscopy image of colloidal cluster",
    invalid_color="lightblue",
    labelfontsize=35,
    tickfontsize=30,
    titlefontsize=35,
    fontsizecolorbar = 30,
    fontsizetickcbar = 30
)


# %%
fig, ax = plot_valid_order_parameter_histogram_colored(
    df=order_df,
    order_n=order_n,
    psi_col=f"psi_{order_n}",
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    # xlim_px=(185, 585),
    # ylim_px=(375, 675),
    bins=100,
    center_cmap="plasma",
    center_vmin=0.0,
    center_vmax=1.0,
    figsize=(15, 20),
    show_colorbar=False,
    # title=rf"Histogram $\psi_{{{order_n}}}$ with {setting} neighbours",
    title=rf"Histogram $\psi_{{{order_n}}}$ with {setting} neighbours, only 3 neighbours",
    titlefontsize=35,
    labelfontsize=30,
    tickfontsize=30,
    legendfontsize=30,
    borderpad=0.5,
    labelspacing=0.5,
    handlelength=2.0,
    normalize=True
)


#%%



# fig, ax = plot_valid_order_parameter_values_line(
#     df=order_df,
#     order_n=order_n,
#     psi_col=f"psi_{order_n}",
#     x_col="x",
#     y_col="y",
#     um_per_px=0.65,
#     xlim_px=(185, 585),
#     ylim_px=(375, 675),
#     center_cmap="plasma",
#     sort_values=True,
#     title=rf"Valid $|\psi_{{{order_n}}}|$ values with {setting} neighbours"
# )
# %%
""" Coord histogram"""


fig, ax = plot_coordination_number_histogram(
    df=order_df,
    coord_col="coord_num",
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    xlim_px=(185, 585),
    ylim_px=(375, 675),
    normalize=True,
    title=f"Coordination number distribution with {setting} neighbours"
)

fig, ax = plot_coordination_vs_average_order_parameter(
    df=order_df,
    order_n=order_n,
    psi_col=f"psi_{order_n}",
    x_col="x",
    y_col="y",
    um_per_px=0.65,
    xlim_px=(185, 585),
    ylim_px=(375, 675),
    title=rf"Average $\psi_{{{order_n}}}$ per coordination number with {setting} neighbours"
)
# %%
