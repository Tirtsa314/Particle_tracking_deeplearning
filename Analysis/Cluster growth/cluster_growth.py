import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import networkx as nx

# ----------------------------
# Normal (unweighted) cluster size with persistence filter
# ----------------------------
def compute_mean_cluster_size(particles_csv, cutoff, min_persist=2):
    particles = pd.read_csv(particles_csv)

    # # Standardize column names
    # for df in [particles]:
    #     df.columns = [col.upper() for col in df.columns]
    # particles.rename(columns={'POSITION_X': 'X', 'POSITION_Y': 'Y'}, inplace=True)
    frame_df = particles.loc[particles['FRAME'] == frame, ['X', 'Y', 'PARTICLE']].copy()

    particles_xy = frame_df[['X', 'Y']].values
    particle_ids = frame_df['PARTICLE'].values

    frames = sorted(set(particles['FRAME'].unique()))
    mean_cluster_sizes = []
    cluster_memory = []  # memory for persistence

    for frame in frames:
        particles_xy = particles.loc[particles['FRAME'] == frame, ['X', 'Y']].values #extracting values in that frame

        if len(particles_xy) == 0:
            mean_cluster_sizes.append(np.nan)
            # cluster_memory = []
            continue

        # Build bipartite graph red<->green
        particles_tree = cKDTree(particles_xy)
    
        pairs = particles_tree.query_pairs(r=cutoff)

        G = nx.Graph()
        G.add_nodes_from([f"p{pid}" for pid in particle_ids])
        for i, j in pairs:
            G.add_edge(f"p{particle_ids[i]}", f"p{particle_ids[j]}")

        clusters = [set(c) for c in nx.connected_components(G) if len(c) > 1]

        # persistence filter
        persisted_clusters = []
        for cl in clusters:
            matched = False
            for old_cl, age in cluster_memory:
                if len(cl & old_cl) > 0:
                    persisted_clusters.append((cl, age + 1))
                    matched = True
                    break
            if not matched:
                persisted_clusters.append((cl, 1))

        valid_clusters = [cl for cl, age in persisted_clusters if age >= min_persist]
        cluster_memory = persisted_clusters

        if valid_clusters:
            sizes = [len(c) for c in valid_clusters]
            mean_size = np.mean(sizes)   # <-- normal mean
        else:
            mean_size = np.nan

        mean_cluster_sizes.append(mean_size)

    # frame → time (10 s per frame)
    time = np.array([f * 10 for f in frames])
    return time, np.array(mean_cluster_sizes)



# ----------------------------
# New method: rolling mean + derivative
# ----------------------------
def rolling_average_and_growth(time, values, window, frame_interval=10, window_in_minutes=True):
    if window_in_minutes:
        window_frames = int(round(window * 60 / frame_interval))
    else:
        window_frames = int(window)
    if window_frames < 1:
        window_frames = 1

    df = pd.DataFrame({"time": time, "value": values})

    # rolling mean
    df["value_smooth"] = df["value"].rolling(
        window=window_frames,
        center=True,
        min_periods=1
    ).mean()

    # derivative of smoothed curve
    dv = np.diff(df["value_smooth"].values)
    dt = np.diff(df["time"].values) / 60.0  # sec → min
    rate = dv / dt
    t_mid = ((df["time"].values[:-1] + df["time"].values[1:]) / 2) / 60.0

    return df["time"].values/60.0, df["value_smooth"].values, t_mid, rate


# # ----------------------------
# # Run analysis
# # ----------------------------
# cutoff_1 = 14, 16
# particle_path = 
# time1, sizes1 = compute_mean_cluster_size(r"C:\Users\Caipa\Desktop\PhD LION\Year3&4\BSc_Clusters\code\cluster size-yogesh\cluster size\2um+2um_0-1h_red_coords_average_cluster_size_and_growth_rate.csv", r"C:\Users\Caipa\Desktop\PhD LION\Year3&4\BSc_Clusters\code\cluster size-yogesh\cluster size\2um+2um_0-1h_green_coords_average_cluster_size_and_growth_rate.csv", cutoff_1)




# # New method
# window_minutes = 2
# t1_roll, s1_roll, t1_mid_roll, rate1_roll = rolling_average_and_growth(time1, sizes1, window_minutes)



# plt.figure(figsize=(14,5))

# # ----------------------------
# # Cluster sizes (raw, no rolling average)
# # ----------------------------
# ax1 = plt.subplot(1,2,1)

# plt.plot(time2/60.0, sizes2, 'o', alpha=1, label="Equal Sized Spheres", markeredgecolor='black', markersize=10, linewidth=0.2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         keredgecolor='black', markersize=10, linewidth=0.2)
# plt.xlabel('T (min)', fontsize=20)
# plt.ylabel(r'$\langle N_c \rangle$', fontsize=20)
# plt.legend(fontsize=11)
# plt.tick_params(axis='both', which='major', labelsize=20)

# # Add (a) in top-left
# ax1.text(-0.15, 1.05, "(a)", transform=ax1.transAxes,
#          fontsize=22, va='top', ha='left')
# # ----------------------------
# # Growth rates (rolling-averaged)
# # ----------------------------
# ax2 = plt.subplot(1,2,2)
# plt.plot(t2_mid_roll, rate2_roll, 'o', label="Equal Sized Spheres",markeredgecolor='black', markersize=10, linewidth=0.2)
# plt.plot(t1_mid_roll, rate1_roll, 's', label="Unequal Sized Spheres", markeredgecolor='black', markersize=10, linewidth=0.2)
# plt.xlabel('T (min)', fontsize=20)
# plt.ylabel(r'$d\langle N_c \rangle / dt$', fontsize=20, labelpad=-12)
# plt.legend(fontsize=11, loc='upper left')
# plt.tick_params(axis='both', which='major', labelsize=20)
# ax2.text(-0.1, 1.05, "(b)", transform=ax2.transAxes,
#          fontsize=22, va='top', ha='left')

# # fontweight='bold'
# plt.tight_layout()
# # plt.savefig("Average_clauster_size_and_growth_rate_.pdf", dpi=600)