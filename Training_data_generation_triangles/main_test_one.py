
from Training_data_generation_triangles.Yolo_export import export_one_yolo_test

if __name__ == "__main__":
    export_one_yolo_test(
        out_dir=r"C:\Train_sets\Test_set",
        split="train",
        stem="000008",
        scale=2,
        show_overlay=True,
    )