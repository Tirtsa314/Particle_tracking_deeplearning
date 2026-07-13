
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(arr8, cmap="gray")

# plot all detected centers
ax.scatter(
    det_df["x"],
    det_df["y"],
    s=2,
    c=np.arange(len(det_df)),
    cmap="hsv",
    marker="o"
)

cutoff = 46.8

# choose a few random particles
n_circles = 40
np.random.seed(0)  # optional
indices_to_circle = np.random.choice(len(det_df), size=min(n_circles, len(det_df)), replace=False)

for i in indices_to_circle:
    circ = Circle(
        (det_df["x"].iloc[i], det_df["y"].iloc[i]),
        cutoff,
        edgecolor="red",
        facecolor="none",
        linewidth=1.5
    )
    ax.add_patch(circ)

ax.set_axis_off()
ax.set_title("Detected centers with cutoff circles")
plt.show()