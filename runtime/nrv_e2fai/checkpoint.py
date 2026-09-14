from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch

from .model import FrozenFlowRecurrentImageE2FAI


OFFICIAL_E2FAI_SHA256 = "ab2ce83b29d081ea1a6de88dd75e4cbc3fe01d00fce4c38b680b7aa1d0cb1eef"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recurrent_model(
    adapter_checkpoint: str | Path,
    *,
    backbone_checkpoint: str | Path,
    device: str | torch.device,
    sensor_height: int,
    sensor_width: int,
    supported_resolutions: tuple[tuple[int, int], ...] | None = None,
):
    adapter_path, backbone_path = Path(adapter_checkpoint), Path(backbone_checkpoint)
    for path in (adapter_path, backbone_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    backbone_hash = sha256(backbone_path)
    if backbone_hash != OFFICIAL_E2FAI_SHA256:
        raise ValueError(f"Unexpected E2FAI backbone SHA256: {backbone_hash}")

    payload = torch.load(adapter_path, map_location="cpu")
    required = {"image_adapter", "pretrained_sha256", "resolved_args"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Image checkpoint must contain {sorted(required)}")
    adapter_state = payload["image_adapter"]
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ValueError("image_adapter must be a non-empty state dict")
    if payload["pretrained_sha256"] != backbone_hash:
        raise ValueError("Image checkpoint and E2FAI backbone do not match")
    projection = adapter_state.get("projection.weight")
    if not isinstance(projection, torch.Tensor) or projection.ndim != 4:
        raise ValueError("Invalid image residual projection.weight")

    model = FrozenFlowRecurrentImageE2FAI(
        backbone_path, sensor_height, sensor_width, int(projection.shape[0]),
        supported_resolutions=supported_resolutions,
    )
    model.image_adapter.load_state_dict(adapter_state, strict=True)
    model.to(device).eval()
    return model, {
        "adapter_checkpoint": str(adapter_path.resolve()),
        "adapter_sha256": sha256(adapter_path),
        "backbone_checkpoint": str(backbone_path.resolve()),
        "backbone_sha256": backbone_hash,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "objective": payload["resolved_args"].get("objective"),
    }
