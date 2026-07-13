
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
from Training_data_generation_pentagons.Image_creation.Image_generator import generate_image
from Training_data_generation_pentagons.Yolo_export import stream_export_yolo_parallel

if __name__ == "__main__":
    stream_export_yolo_parallel(
        out_dir=r"C:\Train_sets\Pentagons\pentagons_16-03",
        split="train",
        num_workers=1,
        scale = 2
    )