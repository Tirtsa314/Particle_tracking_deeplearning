#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\DenHaan\Downloads\tiled_detections.csv")

# Extract coordinates
coords = df[['x', 'y']].values

# Define cutoff distance in pixel, typically 1.1time partciles diameter
cutoff_distance = 46.8

# Calculate all pairwise distances using broadcasting
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
distances = np.linalg.norm(diff, axis=2)

# Mask to exclude self distances and those above cutoff
mask = (distances < cutoff_distance) & (distances != 0)

# Calculate angles using broadcasting and masked distances
dy, dx = diff[:, :, 1], diff[:, :, 0]
angles = np.arctan2(dy, dx)
masked_angles = np.where(mask, angles, np.nan)

# Calculate psi values
exp_angles = np.exp(6j * masked_angles)
psi_values = np.nanmean(exp_angles, axis=1)
psi_values = np.abs(np.nan_to_num(psi_values))  # Replace NaN with 0 and take absolute value

# Ensure all arrays have the same length
if len(psi_values) != len(coords):
    psi_values = psi_values[:len(coords)]

# Extract x and y coordinates
x_coords, y_coords = coords[:, 0], coords[:, 1]

# Set up the figure and axis with equal aspect ratio
fig, ax1 = plt.subplots()
ax1.set_aspect('equal')

# Scatter plot for the data using psi values as color
sc1 = ax1.scatter(x_coords, y_coords, c=psi_values, s=1, cmap=plt.cm.OrRd)

# Custom limits for x and y axis
# plt.xlim(150, 1150)
# plt.ylim(-2000, -1000)

# # Custom x and y ticks
# plt.xticks([150, 650, 1150], [0, 50, 100], fontsize=14)
# plt.yticks([-2000, -1500, -1000], [0, 50, 100], fontsize=14)

# Adjust tick parameters
plt.tick_params(axis='both', direction='in', length=10, width=2)

# Add a colorbar
cb = plt.colorbar(sc1, pad=0.02)
cb.outline.set_linewidth(2)
cb.ax.tick_params(length=8)
cb.ax.yaxis.set_ticks_position('right')
cb.ax.tick_params(direction='in')
cb.set_label(label=r'$\psi_{4}$', size=16)
cb.ax.tick_params(labelsize=16)
cb.ax.yaxis.set_label_coords(3.5, 0.5)

# Set axis labels and padding
plt.xlabel('X ($\mu$m)', size=14)
plt.ylabel('Y ($\mu$m)', size=14)
plt.gca().yaxis.labelpad = -10
plt.gca().xaxis.labelpad = -2
plt.gca().invert_yaxis()
# Set spine thickness
spine_thickness = 2  
for spine in plt.gca().spines.values():
    spine.set_linewidth(spine_thickness)

# Save the figure with high DPI and tight bounding box
#plt.savefig('psi_6_GR_all_nn_crop_1.pdf', dpi=600, bbox_inches='tight')

# Display the plot
plt.show()
# %%
