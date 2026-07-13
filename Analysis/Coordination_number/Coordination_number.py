#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import networkx as nx

# ----------------------------
# Normal (unweighted) cluster size with persistence filter
# ----------------------------
def compute_coord_num(particles_csv, cutoff, min_persist=2):
    particles = pd.read_csv(particles_csv)

    particles_xy = particles[['x', 'y']].values

    if len(particles_xy) == 0:
        return np.nan, np.array([])

    particles_tree = cKDTree(particles_xy)

    coord_num = particles_tree.query_ball_point(particles_xy, r=cutoff, return_length=True) - 1
    mean_coordination = np.mean(coord_num)

    return mean_coordination, coord_num

csv_path = r"C:\Users\DenHaan\Downloads\tiled_detections.csv"
cutoff = 46.8

mean_coordination, coord_num = compute_coord_num(csv_path,cutoff)

#%%
from matplotlib.patches import Circle

coord_num_nonzero = coord_num[coord_num > 0]

plt.figure(figsize=(8, 5))
bins = np.arange(coord_num_nonzero.min(), coord_num_nonzero.max() + 2) - 0.5
plt.hist(coord_num_nonzero, bins=bins, edgecolor="black")
plt.xticks(np.arange(coord_num_nonzero.min(), coord_num_nonzero.max() + 1))
plt.xlabel("Coordination number")
plt.ylabel("Count")
plt.title("coordination numbers")
plt.show()
# %%
