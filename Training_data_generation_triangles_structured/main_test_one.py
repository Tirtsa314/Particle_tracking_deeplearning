""" 
author: Tirtsa den Haan 
06-07-2026
"""
from Training_data_generation_triangles_structured.Yolo_export import export_one_yolo_test

if __name__ == "__main__":
    export_one_yolo_test(
        out_dir=r"C:\Train_sets\Test_set",
        split="train",
        stem="000010",
        scale=1,
        show_overlay=True,
    )