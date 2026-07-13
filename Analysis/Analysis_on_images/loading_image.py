import numpy as np
import nd2
from pprint import pprint
from pathlib import Path
import imageio.v3 as iio


def to_uint8(img, mode="minmax", p_low=0, p_high=100):

    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img.copy()

    img = img.astype(np.float32)

    if mode == "divide256":
        img8 = np.clip(img / 256.0, 0, 255).astype(np.uint8)

    elif mode == "minmax":
        mn, mx = img.min(), img.max()
        if mx <= mn:
            img8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img8 = ((img - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    elif mode == "percentile":
        lo, hi = np.percentile(img, [p_low, p_high])
        if hi <= lo:
            img8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img8 = ((img - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

    else:
        raise ValueError("mode must be 'minmax', 'percentile', or 'divide256'")

    return img8

def reading_nd2(file_path, frame_index=-1):
    
    FRAME_INDEX = frame_index   # -1 = last frame, 0 = first frame, 10 = frame 10, etc.

    # Optional: inspect ND2 metadata
    with nd2.ND2File(file_path) as f:
        print("shape:", f.shape)
        print("sizes:", f.sizes)
        print("voxel size:", f.voxel_size())

        print("\n=== text_info ===")
        pprint(f.text_info)

        print("\n=== experiment ===")
        pprint(f.experiment)

        print("\n=== first frame metadata ===")
        pprint(f.frame_metadata(0))


    # Read ND2 as lazy dask array
    framesarr = nd2.imread(file_path, dask=True)

    print("framesarr shape:", framesarr.shape)
    print("framesarr ndim :", framesarr.ndim)

    # Select frame safely
    if framesarr.ndim == 2:
        # ND2 contains only one 2D image
        raw_frame = framesarr

    elif framesarr.ndim == 3:
        # ND2 contains multiple frames: shape probably (T, Y, X)
        raw_frame = framesarr[FRAME_INDEX]

    elif framesarr.ndim == 4:
        # ND2 may have time + channel or z
        # common shape: (T, C, Y, X) or (T, Z, Y, X)
        raw_frame = framesarr[FRAME_INDEX, 0]

    else:
        raise ValueError(f"Unexpected ND2 shape: {framesarr.shape}")

    # Convert from dask array to numpy array
    try:
        img16 = raw_frame.compute()
    except AttributeError:
        img16 = np.asarray(raw_frame)

    img16 = np.squeeze(img16)

    print("selected img16 shape:", img16.shape)

    if img16.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img16.shape}")

    arr8 = to_uint8(img16, mode="percentile", p_low=0, p_high=100)
    frame = np.stack([arr8, arr8, arr8], axis=-1)

    return frame,arr8





def save_image(image, save_path):


    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(image)

    # remove single-channel dimension if present, e.g. (512, 512, 1)
    img = np.squeeze(img)

    # convert float image to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.float32)

        img_min = np.min(img)
        img_max = np.max(img)

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        img = (255 * img).astype(np.uint8)

    iio.imwrite(save_path, img)

    print(f"Saved image to: {save_path}")