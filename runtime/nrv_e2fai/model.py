from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.double_conv(value)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(value)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        dy, dx = skip.size(2) - value.size(2), skip.size(3) - value.size(3)
        value = F.pad(value, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.conv(torch.cat((skip, value), dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class UNet(nn.Module):
    def __init__(self, input_channels: int = 15, output_channels: int = 3):
        super().__init__()
        self.n_channels = input_channels
        self.n_classes = output_channels
        self.bilinear = True
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, output_channels)


class ConvGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.input_size, self.hidden_size = input_size, hidden_size
        channels = input_size + hidden_size
        self.reset_gate = nn.Conv2d(channels, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(channels, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(channels, hidden_size, kernel_size, padding=padding)
        for layer in (self.reset_gate, self.update_gate, self.out_gate):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if state is None:
            state = value.new_zeros(value.shape[0], self.hidden_size, *value.shape[2:])
        stacked = torch.cat((value, state), dim=1)
        update = torch.sigmoid(self.update_gate(stacked))
        reset = torch.sigmoid(self.reset_gate(stacked))
        candidate = torch.tanh(self.out_gate(torch.cat((value, state * reset), dim=1)))
        return state * (1 - update) + candidate * update


class DenseFlowInterpolation(nn.Module):
    """Exact cached form of the E2FAI tile-flow interpolation."""

    def __init__(self, input_height: int, input_width: int, output_height: int, output_width: int):
        super().__init__()
        self.input_height, self.input_width = input_height, input_width
        self.output_height, self.output_width = output_height, output_width
        tile_width, tile_height = output_width / input_width, output_height / input_height
        x_range = np.arange(tile_width, output_width + tile_width)
        y_range = np.arange(tile_height, output_height + tile_height)
        pixels = torch.from_numpy(np.array([[x, y] for y in y_range for x in x_range])).float()
        if pixels.shape[0] != output_height * output_width:
            raise ValueError("Unsupported tile/output dimensions")
        tile_x = (pixels[:, 0] - tile_width / 2) / tile_width
        tile_y = (pixels[:, 1] - tile_height / 2) / tile_height
        x0, y0 = torch.floor(tile_x).int(), torch.floor(tile_y).int()
        dx, dy = tile_x - x0, tile_y - y0
        padded_width = input_width + 2
        self.register_buffer("indices", torch.stack((
            y0.long() * padded_width + x0.long(),
            (y0 + 1).long() * padded_width + x0.long(),
            y0.long() * padded_width + x0.long() + 1,
            (y0 + 1).long() * padded_width + x0.long() + 1,
        )), persistent=False)
        self.register_buffer("weights", torch.stack((
            (1 - dx) * (1 - dy), (1 - dx) * dy, dx * (1 - dy), dx * dy,
        )), persistent=False)

    def forward(self, flow_tiles: torch.Tensor) -> torch.Tensor:
        flat = F.pad(flow_tiles, (1, 1, 1, 1), mode="replicate").flatten(2)
        values = [
            torch.gather(flat, 2, index.expand(flow_tiles.shape[0], 2, -1))
            for index in self.indices
        ]
        dense = sum(value * weight for value, weight in zip(values, self.weights))
        return dense.reshape(flow_tiles.shape[0], 2, self.output_height, self.output_width)


class RecurrentImageResidual(nn.Module):
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.projection = nn.Conv2d(128, hidden_channels, 1)
        self.recurrent = ConvGRU(hidden_channels, hidden_channels)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

    def forward(self, feature: torch.Tensor, state: torch.Tensor | None):
        state = self.recurrent(F.relu(self.projection(feature), inplace=False), state)
        return self.decoder(state), state


class FrozenFlowRecurrentImageE2FAI(nn.Module):
    def __init__(
        self,
        backbone_checkpoint: str | Path,
        sensor_height: int,
        sensor_width: int,
        recurrent_channels: int = 32,
        num_bins: int = 15,
        flow_tile_size: int = 16,
        supported_resolutions: tuple[tuple[int, int], ...] | None = None,
    ):
        super().__init__()
        resolutions = supported_resolutions or ((sensor_width, sensor_height),)
        resolutions = tuple(dict.fromkeys((int(width), int(height)) for width, height in resolutions))
        if (sensor_width, sensor_height) not in resolutions:
            resolutions = ((sensor_width, sensor_height),) + resolutions
        if any(min(width, height) <= 0 or width % flow_tile_size or height % flow_tile_size
               for width, height in resolutions):
            raise ValueError("All sensor dimensions must be positive and divisible by 16")
        self.sensor_height, self.sensor_width, self.num_bins = sensor_height, sensor_width, num_bins
        self.supported_resolutions = frozenset(resolutions)
        self.backbone = UNet(num_bins, 3)
        checkpoint = torch.load(Path(backbone_checkpoint), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        if state and all(key.startswith("model.") for key in state):
            state = {key[6:]: value for key, value in state.items()}
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False).eval()
        self.image_adapter = RecurrentImageResidual(recurrent_channels)
        self.flow_pool = nn.AvgPool2d(flow_tile_size)
        self.flow_interpolations = nn.ModuleDict({
            '{}x{}'.format(width, height): DenseFlowInterpolation(
                height // flow_tile_size, width // flow_tile_size, height, width)
            for width, height in resolutions
        })

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_step(self, voxel: torch.Tensor, state: torch.Tensor | None = None):
        if voxel.ndim != 4 or voxel.shape[1] != self.num_bins:
            raise ValueError(f"Expected voxel [B,{self.num_bins},H,W], got {tuple(voxel.shape)}")
        height, width = int(voxel.shape[2]), int(voxel.shape[3])
        if (width, height) not in self.supported_resolutions:
            raise ValueError(f"Unsupported model resolution: {width}x{height}")
        with torch.no_grad():
            x1 = self.backbone.inc(voxel)
            x2 = self.backbone.down1(x1)
            x3 = self.backbone.down2(x2)
            x4 = self.backbone.down3(x3)
            x5 = self.backbone.down4(x4)
            decoded = self.backbone.up1(x5, x4)
            feature = self.backbone.up2(decoded, x3)
            decoded = self.backbone.up3(feature, x2)
            decoded = self.backbone.up4(decoded, x1)
            raw = self.backbone.outc(decoded)
        flow = self.flow_interpolations['{}x{}'.format(width, height)](self.flow_pool(raw[:, :2]))
        raw_residual, state = self.image_adapter(feature, state)
        residual = raw_residual - raw_residual.mean(dim=(-2, -1), keepdim=True)
        log_image = raw[:, 2:3] + residual
        return {"flow": flow, "log_image": log_image, "image": log_image.exp()}, state
