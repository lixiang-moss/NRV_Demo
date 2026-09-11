from __future__ import annotations

import cv2
import numpy as np
import torch


def as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def event_polarity_rgb(voxels: torch.Tensor | np.ndarray) -> np.ndarray:
    array = as_numpy(voxels)
    if array.ndim != 5:
        raise ValueError("Expected event voxels [B,T,C,H,W]")
    positive = np.clip(array, 0, None).sum(axis=2)
    negative = np.clip(-array, 0, None).sum(axis=2)
    strength = np.maximum(positive, negative)
    scales = np.ones(array.shape[0], dtype=np.float32)
    for batch in range(array.shape[0]):
        active = strength[batch][strength[batch] > 0]
        if active.size:
            scales[batch] = max(float(np.percentile(active, 99)), 1e-6)
    positive = np.clip(positive / scales[:, None, None, None], 0, 1)
    negative = np.clip(negative / scales[:, None, None, None], 0, 1)
    rgb = np.full((*positive.shape, 3), 255.0, dtype=np.float32)
    rgb[..., 0] *= 1 - negative
    rgb[..., 1] *= 1 - np.maximum(positive, negative)
    rgb[..., 2] *= 1 - positive
    return rgb.round().astype(np.uint8)


def flow_hsv_rgb(flows: torch.Tensor | np.ndarray, max_magnitude: float) -> np.ndarray:
    array = np.nan_to_num(as_numpy(flows))
    if array.ndim != 5 or array.shape[2] != 2 or max_magnitude <= 0:
        raise ValueError("Expected flow [B,T,2,H,W] and a positive scale")
    u, v = array[:, :, 0], array[:, :, 1]
    magnitude = np.hypot(u, v)
    rgb = np.empty((*magnitude.shape, 3), dtype=np.uint8)
    for batch in range(array.shape[0]):
        for step in range(array.shape[1]):
            hsv = np.empty((*magnitude.shape[-2:], 3), dtype=np.uint8)
            hsv[..., 0] = ((np.arctan2(v[batch, step], u[batch, step]) + np.pi) * 90 / np.pi).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = (255 * np.clip(magnitude[batch, step] / max_magnitude, 0, 1)).round().astype(np.uint8)
            rgb[batch, step] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
