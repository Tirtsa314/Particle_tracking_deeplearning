""" 
author: Tirtsa den Haan 
06-07-2026
"""
from ultralytics import YOLO
import torch
import time
import gc
from pathlib import Path

GB = 1024**3 #Number of bytes in one gigabyte


def print_cuda_mem(prefix=""): #prints current GPU memory usage in GB
    if not torch.cuda.is_available():
        return
    print(
        f"{prefix}"
        f"alloc={torch.cuda.memory_allocated()/GB:.2f} GB, "
        f"reserved={torch.cuda.memory_reserved()/GB:.2f} GB, "
        f"peak_alloc={torch.cuda.max_memory_allocated()/GB:.2f} GB, "
        f"peak_reserved={torch.cuda.max_memory_reserved()/GB:.2f} GB"
    )


def on_train_epoch_start(trainer):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def on_train_epoch_end(trainer):
    if torch.cuda.is_available():
        epoch_num = trainer.epoch + 1
        print_cuda_mem(prefix=f"[epoch {epoch_num}] ")


def attach_callbacks(model):
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)


def cleanup_cuda(): #cleans out the memory of the GPU
    gc.collect() #collect garbage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect() #collects interprocess memory
        except Exception:
            pass #ignore if no interprocess memory to collect
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize() #makes sure all processes are finished


def is_oom_error(e: Exception) -> bool: #checks if the exception is a CUDA out-of-memory error
    msg = str(e).lower() #converts the exception message to lowercase for case-insensitive comparison
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg


if __name__ == "__main__":
    init_weights = Path("/home/s2329255/ondemand/projects/default/24/runs/segment/train/weights/best.pt") #weights from previously trained model
    data_path = "/home/s2329255/ondemand/projects/Tracking/merged_dataset_09-04/data.yaml"

    epochs = 100
    imgsz = 512
    batch = 1

    project = "/home/s2329255/ondemand/projects/Tracking/runs/segment"
    name = "Model_nieuweyolo_15-04_ratio1_100epochs"

    run_dir = Path(project) / name
    last_ckpt = run_dir / "weights" / "last.pt" #last weights are stored here

    max_retries = 20 #max retries from OOM errors
    retry = 0 #retry count
    t0 = time.time()

    results = None

    while True:
        model = None
        try:
            print_cuda_mem(prefix="[before train call] ")

            if last_ckpt.exists():
                print(f"\nFound checkpoint: {last_ckpt}")
                print("Resuming training from last completed epoch...")
                model = YOLO(str(last_ckpt))
                attach_callbacks(model)

                # Resume uses the checkpoint's saved state/args
                results = model.train(resume=True)

            else:
                print("\nNo checkpoint found, starting new run...")
                model = YOLO(str(init_weights))
                attach_callbacks(model)

                results = model.train(
                    data=data_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    device=0,
                    workers=0,
                    cache=False,
                    amp=True,
                    mosaic=0.0,
                    close_mosaic=0,
                    mixup=0.0,
                    cutmix=0.0,
                    copy_paste=0.0,
                    mask_ratio=1,
                    overlap_mask=False,
                    val=False,
                    plots=False,
                    project=project,
                    name=name,
                    exist_ok=True,
                )

            print_cuda_mem(prefix="[after train call] ")
            print("\nTraining finished successfully.")
            break

        except Exception as e:
            if not is_oom_error(e):
                raise

            retry += 1
            print(f"\nCUDA OOM caught (retry {retry}/{max_retries}): {e}")

            # Free references first
            del model
            cleanup_cuda()
            print_cuda_mem(prefix="[after cleanup] ")

            if not last_ckpt.exists():
                raise RuntimeError(
                    "OOM happened before the first epoch checkpoint was written, "
                    "so there is nothing to resume from."
                ) from e

            if retry >= max_retries:
                raise RuntimeError(
                    f"Still failing after {max_retries} OOM retries."
                ) from e

            print("Retrying from last.pt in 10 seconds...\n")
            time.sleep(10)

    elapsed = time.time() - t0
    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else run_dir
    summary_file = save_dir / "training_summary.txt"
    
    summary_text = (
        f"Initial weights: {init_weights}\n"
        f"Data: {data_path}\n"
        f"Requested epochs variable: {epochs}\n"
        f"Image size: {imgsz}\n"
        f"Batch size: {batch}\n"
        f"Training time: {elapsed/60:.2f} minutes\n"
        f"Training time: {elapsed/3600:.2f} hours\n"
        f"Saved to: {save_dir}\n"
    )

    print(summary_text)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Summary written to: {summary_file}")