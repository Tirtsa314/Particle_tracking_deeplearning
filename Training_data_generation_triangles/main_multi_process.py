
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
from Training_data_generation_triangles.Image_creation.Image_generator import generate_image
from Training_data_generation_triangles.Yolo_export import stream_export_yolo_parallel

if __name__ == "__main__":
    stream_export_yolo_parallel(
        out_dir="c:\Train_sets\Thesis_set_11-06",
        split="train",
        num_workers=1,
        scale = 2
    )

