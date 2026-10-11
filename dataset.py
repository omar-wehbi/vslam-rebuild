from pathlib import Path

import cv2
import numpy as np

from frame import Frame


# inferring the intrinsic matrix from an image
def infer_camera_matrix(image):
    h, w = image.shape[:2]
    f = 0.9 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def load_frames(folder, method="ORB", ext=".png", match_type="KNN", K=None, detector_config=None,
                 distortion=None):
    """
    Load images and build list of Frame. If K is None, intrinsics are inferred
    from the first image; otherwise the provided 3x3 matrix is used for all frames.
    """
    folder = Path(folder)
    image_paths = sorted(folder.glob(f"*{ext}"))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {folder}")

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Failed to load image {image_paths[0]}")
    if K is None:
        K = infer_camera_matrix(first)
    else:
        K = np.asarray(K, dtype=np.float64)

    n_total = len(image_paths)
    frames = []
    ''' loop through the images and build the frames'''
    for i, p in enumerate(image_paths):
        # Use grayscale for stable feature extraction (ORB/SIFT are intensity-based).
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image {p}")
        frame = Frame(img, K, method, match_type, detector_config=detector_config, distortion=distortion)
        frames.append(frame)
        # Overwrite same line: image x/n loaded
        print(f"\rimage {i + 1}/{n_total} loaded", end="", flush=True)
    if n_total > 0:
        print()  # newline after progress so next output is on a new line

    return frames


def iter_frames(folder, method="ORB", ext=".png", match_type="KNN", K=None, detector_config=None,
                 distortion=None):
    """
    Iterator version of load_frames: yields Frame objects sequentially without keeping
    all images resident in memory at once.

    Returns
    -------
    (K, n_total, iterator)
        K is the 3x3 intrinsics used for all frames; n_total is total frames;
        iterator yields Frame objects in sequence order.
    """
    folder = Path(folder)
    image_paths = sorted(folder.glob(f"*{ext}"))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {folder}")

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Failed to load image {image_paths[0]}")
    if K is None:
        K_use = infer_camera_matrix(first)
    else:
        K_use = np.asarray(K, dtype=np.float64)

    n_total = len(image_paths)

    def _gen():
        for i, p in enumerate(image_paths):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to load image {p}")
            yield Frame(img, K_use, method, match_type, detector_config=detector_config, distortion=distortion)

    return K_use, n_total, _gen()


def load_timestamps_for_images(folder, ext=".png", rgb_file=None):
    """
    Load timestamps from rgb.txt for images in the same order as load_frames.
    Used only when saving estimated trajectory to file (TUM format).
    folder : path to directory containing images (order = sorted glob).
    rgb_file : optional path to rgb.txt; if None, look in folder or folder.parent.
    Returns list of (timestamp_float, path_str) in sorted path order, or None if no file found.
    """
    folder = Path(folder) #convert the folder to a Path object
    rgb_txt = Path(rgb_file) if rgb_file else None #convert the rgb_file to a Path object in case it is given.
    if rgb_txt is None or not rgb_txt.is_file(): #if the rgb_txt is not given or not a file, look for it in the folder or the folder's parent.
        rgb_txt = folder / "rgb.txt" #look for it in the folder.
    if not rgb_txt.is_file(): #if the rgb_txt is not a file, look for it in the folder's parent.
        rgb_txt = folder.parent / "rgb.txt" #look for it in the folder's parent.
    if not rgb_txt.is_file(): #if the rgb_txt is not a file, return None.
        return None
    # Build dict: basename -> timestamp (images may be in rgb/ subfolder in file)
    name_to_ts = {} #build a dictionary to store the timestamps.
    ''' loop through the rgb_txt file and store the timestamps in the dictionary'''
    with open(rgb_txt) as f: #open the rgb_txt file.
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                continue
            # path can be "rgb/1311877977.077520.png" or just filename
            path_str = parts[1]
            basename = Path(path_str).name
            name_to_ts[basename] = ts
    # Order by sorted image paths (same as load_frames)
    image_paths = sorted(folder.glob(f"*{ext}"))
    ''' loop through the image paths and store the timestamps in the result list'''
    result = []
    for p in image_paths:
        ts = name_to_ts.get(p.name) #get the timestamp for the image.
        if ts is not None: #if the timestamp is found, add it to the result list.
            result.append((ts, str(p))) #add the timestamp and the image path to the result list.
    return result if result else None #return the result list if it is not empty, otherwise return None.
