# # ==================================================
# # Simulation: Name
# # Author: Tirtsa
# # Date: 2026-02-24

# #In this code I simulate images for training the model
# # ==================================================.
# # %% 
# import numpy as np
# #from deeptrack import units as u
# #from deeptrack.backend.units import ConversionTable
# import matplotlib.pyplot as plt
# import trackpy as tp

# # if __name__ == "__main__":
# #     stream_export_yolo_parallel(
# #         out_dir="C:/Generation_rand_triangles",
# #         split="train",
# #         num_workers=1
# #     )

# # %%
# from multiprocessing import Queue
# from Training_data_generation.Yolo_export import worker_loop

# q = Queue(maxsize=1)
# worker_loop(q, n=1)          # generate exactly 1 sample
# img_u8, lines = q.get()

# print("image shape:", img_u8.shape)
# print("num label lines:", len(lines))
# import numpy as np
# import matplotlib.pyplot as plt
# #%%
# def show_overlay(img_u8, outlines_px, title="overlay"):
#     plt.figure(figsize=(6,6))
#     plt.imshow(img_u8, cmap="gray")

#     for poly in outlines_px:
#         if poly is None or len(poly) < 3:
#             continue
#         poly = np.asarray(poly)

#         # close the polygon
#         x = np.r_[poly[:,0], poly[0,0]]
#         y = np.r_[poly[:,1], poly[0,1]]

#         plt.plot(x, y, linewidth=1)

#     plt.title(title)
#     plt.axis("off")
#     plt.show()
#     # suppose you already have one sample
# # img_u8 = ...  shape (H,W)
# # outlines_px = sampler.outlines_px

# #%%
# plt.figure(figsize=(6,6))
# plt.imshow(img_u8, cmap="gray")
# #%%
# from Training_data_generation.Image_creation.Image_generator import generate_image
# from Training_data_generation.Image_creation.Triangle_Sampler_Class import TrianglePrismSampler
# # Create sampler manually
# sampler = TrianglePrismSampler(
#     H=2048,
#     W=2048,
#     margin=50,
#     side_length_px=30,
#     thickness_px=8,
# )

# image_of_particles = generate_image(
#     sampler=sampler,
#     pos_sampler=sampler.pos_sampler,
#     rot_sampler=sampler.rot_sampler,
# )
# #%%

# sampler.reset()
# p1 = sampler.pos_sampler()
# r1 = sampler.rot_sampler()
# p2 = sampler.pos_sampler()
# r2 = sampler.rot_sampler()
# print("p1,r1:", p1, r1)
# print("p2,r2:", p2, r2)
# #%%
# sampler.reset()
# for i in range(3):
#     p = sampler.pos_sampler()
#     r = sampler.rot_sampler()
#     print(i, p, r)


# #%%
# img = image_of_particles.update(verbose=True).resolve()
# img_u8 = (img * 255).astype(np.uint8)
# #%%

# # outlines_px = sampler.outlines_px
# # show_overlay(img_u8, outlines_px)
# #%%

# # %%

# plt.figure()
# plt.imshow(img_u8, cmap="gray")
# plt.axis("off")
# plt.show()
# # %%
# #In this code I simulate images for training the model
# # ==================================================.
# # %% 
# import numpy as np
# #from deeptrack import units as u
# #from deeptrack.backend.units import ConversionTable
# import matplotlib.pyplot as plt
# import trackpy as tp
# from Training_data_generation.Image_creation.Image_generator import generate_image
# from Training_data_generation.Yolo_export import stream_export_yolo_parallel

# # if __name__ == "__main__":
# #     stream_export_yolo_parallel(
# #         out_dir="C:/Generation_rand_triangles",
# #         split="train",
# #         num_workers=1
# #     )

# # %%
# from multiprocessing import Queue
# from Training_data_generation.Yolo_export import worker_loop

# q = Queue(maxsize=1)
# worker_loop(q, n=1)          # generate exactly 1 sample
# img_u8, lines = q.get()

# print("image shape:", img_u8.shape)
# print("num label lines:", len(lines))
# import numpy as np
# import matplotlib.pyplot as plt

# def show_overlay(img_u8, outlines_px, title="overlay"):
#     plt.figure(figsize=(6,6))
#     plt.imshow(img_u8, cmap="gray")

#     for poly in outlines_px:
#         if poly is None or len(poly) < 3:
#             continue
#         poly = np.asarray(poly)

#         # close the polygon
#         x = np.r_[poly[:,0], poly[0,0]]
#         y = np.r_[poly[:,1], poly[0,1]]

#         plt.plot(x, y, linewidth=1)

#     plt.title(title)
#     plt.axis("off")
#     plt.show()
#     # suppose you already have one sample
# # img_u8 = ...  shape (H,W)
# # outlines_px = sampler.outlines_px

# show_overlay(img_u8, sampler.outlines_px)

# # %%
# from Training_data_generation.Image_creation.Triangle_Sampler_Class import TrianglePrismSampler
# from Training_data_generation.Image_creation.Image_generator import generate_image
# from Training_data_generation.Image_creation.particle_position.cluster_position import cluster_positions

# # Create sampler manually
# rot_sampler_fn, pos_sampler_fn, N = cluster_positions()   # one cluster only

# sampler = TrianglePrismSampler(
#     H=2048, W=2048, margin=50, side_length_px=30, thickness_px=8,
#     rot_sampler_fn=rot_sampler_fn,
#     pos_sampler_fn=pos_sampler_fn,
# )

# image_of_particles = generate_image(
#     sampler=sampler,
#     pos_sampler=sampler.pos_sampler,
#     rot_sampler=sampler.rot_sampler, N=N
# )

# sampler.reset()

# img = image_of_particles.update().resolve()
# img_u8 = (img * 255).astype(np.uint8)

# # outlines_px = sampler.outlines_px
# %%
# from pathlib import Path

# out_dir = Path(r"C:\Generation_rand_triangles")
# split = "train"

# img_dir = out_dir / f"images/{split}"
# lab_dir = out_dir / f"labels/{split}"
# img_dir.mkdir(parents=True, exist_ok=True)
# lab_dir.mkdir(parents=True, exist_ok=True)

# stem = "000000"
# img_path = img_dir / f"{stem}.png"
# lab_path = lab_dir / f"{stem}.txt"

# import cv2
# cv2.imwrite(str(img_path), img_u8)

# lab_path.write_text("\n".join(lines) + ("\n" if lines else ""))
# print("wrote:", img_path, lab_path)
# # %%
from multiprocessing import Queue
from Training_data_generation.Yolo_export import worker_loop

out_q = Queue(maxsize=1)

# generate exactly one sample into the queue
worker_loop(out_q, n=1)

# pretend to be the writer: read one item and write it
img_u8, lines = out_q.get()

out_dir = Path(r"C:\Generation_rand_triangles")
split = "train"
img_dir = out_dir / f"images/{split}"
lab_dir = out_dir / f"labels/{split}"
img_dir.mkdir(parents=True, exist_ok=True)
lab_dir.mkdir(parents=True, exist_ok=True)

import cv2
stem = "000001"
cv2.imwrite(str(img_dir / f"{stem}.png"), img_u8)
(lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

print("queue→disk OK")
# %%
from multiprocessing import Queue
from Training_data_generation.Yolo_export import worker_loop
import cv2
from pathlib import Path

out = Path(r"C:\Generation_rand_triangles")
(out / "images/train").mkdir(parents=True, exist_ok=True)
(out / "labels/train").mkdir(parents=True, exist_ok=True)

q = Queue(maxsize=1)

for i in range(3):
    worker_loop(q, n=1)              # generate 1
    img_u8, lines = q.get()

    stem = f"{i:06d}"
    cv2.imwrite(str(out / "images/train" / f"{stem}.png"), img_u8)
    (out / "labels/train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    print("wrote", stem)
# %%
