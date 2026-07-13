""" 
author: Tirtsa den Haan 
06-07-2026
"""
from Training_data_generation_triangles_structured.Yolo_export import stream_export_yolo_parallel

if __name__ == "__main__":
    stream_export_yolo_parallel(
        out_dir="c:\Train_sets\Thesis_set_11-06",
        split="train",
        num_workers=1,
        queue_size=64,
        scale = 1
    )

