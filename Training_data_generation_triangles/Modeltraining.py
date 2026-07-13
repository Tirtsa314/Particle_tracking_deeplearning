# #%%
# from ultralytics import YOLO


# particle_finder2000_model = YOLO("yolo11m-seg.pt")


# train_results = particle_finder2000_model.train(
#     data=".yaml",  # Path to dataset configuration file
#     epochs=100,  # Number of training epochs
#     imgsz=2048,  # Image size for training
#     device="gpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
# )

# # Evaluate the model's performance on the validation set
# metrics = particle_finder2000_model.val()

# # Export the model to ONNX format for deployment
# path = particle_finder2000_model.export(format="onnx")  # Returns the path to the exported model

# print("Exported model saved to:", path)

#%%
from ultralytics import YOLO
import time
from pathlib import Path

if __name__ == "__main__":
    model = YOLO("yolo11m-seg.pt")

    epochs = 1
    imgsz = 1024
    batch = 1

    t0 = time.time()

    results = model.train(
        data="data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0
    )

    elapsed = time.time() - t0
    save_dir = Path(results.save_dir)
    summary_file = save_dir / "training_summary.txt"

    summary_text = (
        f"Model: yolo11m-seg.pt\n"
        f"Data: data.yaml\n"
        f"Epochs: {epochs}\n"
        f"Image size: {imgsz}\n"
        f"Batch size: {batch}\n"
        f"Training time: {elapsed/60:.2f} minutes\n"
        f"Saved to: {save_dir}\n"
        f"Estimated 100 epochs: {100*elapsed/3600:.2f} hours\n"
    )

    print(summary_text)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Summary written to: {summary_file}")