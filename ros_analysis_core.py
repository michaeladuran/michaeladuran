"""Image loading, pairing, segmentation and ROS signal calculations."""
from __future__ import annotations
import re
from pathlib import Path
from typing import BinaryIO
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

def pair_key(relative_path: str) -> tuple[str, str]:
    p = Path(relative_path.replace("\\", "/"))
    stem = re.sub(r"(?i)(?:[_ .-]?C00[12])", "", p.stem)
    return p.parent.as_posix().casefold(), re.sub(r"[_ .-]+", "_", stem).strip("_").casefold()

def to_2d(array: np.ndarray, plane: int = 0) -> np.ndarray:
    arr = np.squeeze(np.asarray(array))
    if arr.ndim == 2: return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float64)
        return rgb @ np.array([0.2126, 0.7152, 0.0722])
    if arr.ndim > 2:
        arr = arr.reshape((-1, *arr.shape[-2:]))
        return arr[min(max(int(plane), 0), arr.shape[0]-1)]
    raise ValueError(f"Unsupported TIFF shape: {np.asarray(array).shape}")

def read_tiff(source: str | Path | BinaryIO, plane: int = 0) -> np.ndarray:
    with tifffile.TiffFile(source) as tif:
        if not tif.series: raise ValueError("TIFF contains no image series")
        return to_2d(tif.series[0].asarray(), plane)

def segment_nuclei(image: np.ndarray, gaussian_sigma: float = 1.0, threshold_multiplier: float = 1.0,
                   min_area: int = 50, max_area: int = 0, split_touching: bool = True):
    work = np.asarray(image, dtype=np.float32)
    smooth = ndi.gaussian_filter(work, sigma=gaussian_sigma) if gaussian_sigma > 0 else work
    finite = smooth[np.isfinite(smooth)]
    if finite.size == 0 or np.max(finite) <= np.min(finite):
        return np.zeros(work.shape, np.int32), 0, float("nan")
    threshold = float(filters.threshold_otsu(finite)) * threshold_multiplier
    mask = morphology.remove_small_objects(smooth > threshold, min_size=max(1, int(min_area)))
    mask = ndi.binary_fill_holes(mask)
    if max_area > 0:
        lab = measure.label(mask)
        for region in measure.regionprops(lab):
            if region.area > max_area: mask[lab == region.label] = False
    if not np.any(mask): return np.zeros(work.shape, np.int32), 0, threshold
    if split_touching:
        dist = ndi.distance_transform_edt(mask)
        markers = measure.label(morphology.local_maxima(dist) & mask)
        labels = segmentation.watershed(-dist, markers, mask=mask) if markers.max() > 1 else measure.label(mask)
    else: labels = measure.label(mask)
    labels = morphology.remove_small_objects(labels, min_size=max(1, int(min_area)))
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32), int(labels.max()), threshold

def integrated_signal(image: np.ndarray, background_percentile: float = 0.0, correction: bool = False):
    finite = np.asarray(image, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0: raise ValueError("Image contains no finite pixel values")
    background = float(np.percentile(finite, np.clip(background_percentile, 0, 99.9)))
    total = float(finite.sum(dtype=np.float64))
    if correction: total = max(0.0, total - background * finite.size)
    return total, background

def analyze_pair(ros, dapi, *, background_percentile=0, background_correction=False,
                 gaussian_sigma=1, threshold_multiplier=1, min_area=50, max_area=0, split_touching=True):
    if ros.shape != dapi.shape:
        raise ValueError(f"Channel dimensions differ (ROS {ros.shape}, DAPI {dapi.shape}); register/crop first")
    labels, count, threshold = segment_nuclei(dapi, gaussian_sigma, threshold_multiplier, min_area, max_area, split_touching)
    signal, background = integrated_signal(ros, background_percentile, background_correction)
    return {"ros_integrated": signal, "ros_background": background, "nuclei_count": count,
            "nuclei_labels": labels, "dapi_threshold": threshold,
            "ros_per_nucleus": signal / count if count else np.nan}

def csv_bytes(records) -> bytes:
    columns = ["folder_name", "image_stem", "ros_file", "dapi_file", "ros_integrated_intensity",
        "estimated_ros_background_per_pixel", "dapi_positive_object_count_cell_estimate", "ros_integrated_intensity_per_estimated_cell", "qc_notes"]
    rows = [{"folder_name": r.folder, "image_stem": r.stem, "ros_file": r.ros_name, "dapi_file": r.dapi_name,
        "ros_integrated_intensity": r.ros_integrated, "estimated_ros_background_per_pixel": r.ros_background,
        "dapi_positive_object_count_cell_estimate": r.nuclei_count, "ros_integrated_intensity_per_estimated_cell": r.per_cell,
        "qc_notes": "; ".join(r.warnings)} for r in records]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8-sig")
