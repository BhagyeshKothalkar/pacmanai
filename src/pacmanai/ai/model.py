from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pacmanai.dataset.data_utils import ModelBatch


@dataclass(frozen=True)
class PacmanFiLMConfig:
    # Input
    image_channels: int = 3
    state_channels: int = 6
    global_dim: int = 4
    action_dim: int = 2

    # CNN
    stem: int = 8
    hidden: int = 16
    bottleneck: int = 32
    blocks: int = 2

    # FiLM
    condition_hidden: int = 16
    condition_dim: int = 32

    # Misc
    groups: int = 8
    residual: bool = True


def _norm(c: int, groups: int) -> nn.GroupNorm:
    groups = min(groups, c)
    while c % groups:
        groups -= 1
    return nn.GroupNorm(groups, c)


class DSBlock(nn.Module):
    def __init__(self, c: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            _norm(c, groups),
            nn.SiLU(),
            nn.Conv2d(c, c, 1, bias=False),
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            _norm(c, groups),
            nn.SiLU(),
            nn.Conv2d(c, c, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        gamma, beta = self.proj(z).chunk(2, dim=-1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]


class PacmanFiLM(nn.Module):
    def __init__(self, cfg: PacmanFiLMConfig):
        super().__init__()
        self.cfg = cfg

        spatial_in = cfg.image_channels + 2 * cfg.state_channels

        self.stem = nn.Sequential(
            nn.Conv2d(spatial_in, cfg.stem, 3, padding=1, bias=False),
            _norm(cfg.stem, cfg.groups),
            nn.SiLU(),
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(cfg.stem, cfg.hidden, 3, stride=2, padding=1, bias=False),
            _norm(cfg.hidden, cfg.groups),
            nn.SiLU(),
            *[DSBlock(cfg.hidden, cfg.groups) for _ in range(cfg.blocks)],
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(cfg.hidden, cfg.bottleneck, 3, stride=2, padding=1, bias=False),
            _norm(cfg.bottleneck, cfg.groups),
            nn.SiLU(),
            *[DSBlock(cfg.bottleneck, cfg.groups) for _ in range(cfg.blocks)],
        )

        self.condition = nn.Sequential(
            nn.Linear(2 * cfg.global_dim + cfg.action_dim, cfg.condition_hidden),
            nn.SiLU(),
            nn.Linear(cfg.condition_hidden, cfg.condition_dim),
            nn.SiLU(),
        )

        self.film = FiLM(cfg.condition_dim, cfg.bottleneck)

        self.decoder = nn.Sequential(
            *[DSBlock(cfg.bottleneck, cfg.groups) for _ in range(cfg.blocks)],

            nn.Conv2d(
                cfg.bottleneck,
                cfg.hidden,
                3,
                padding=1,
                bias=False,
            ),
            _norm(cfg.hidden, cfg.groups),
            nn.SiLU(),

            nn.Conv2d(
                cfg.hidden,
                cfg.stem,
                3,
                padding=1,
                bias=False,
            ),
            _norm(cfg.stem, cfg.groups),
            nn.SiLU(),

            nn.Conv2d(cfg.stem, cfg.image_channels, 3, padding=1),
        )

    def forward(
        self,
        *,
        image: Tensor,
        state_map: Tensor,
        state_global: Tensor,
        state_to_map: Tensor,
        state_to_global: Tensor,
        action: Tensor,
    ) -> Tensor:
        size = image.shape[-2:]

        state_map = F.interpolate(state_map, size=size, mode="nearest")
        state_to_map = F.interpolate(state_to_map, size=size, mode="nearest")

        x = self.stem(torch.cat(
            (image, state_map, state_to_map),
            dim=1,
        ))

        x = self.enc1(x)
        x = self.enc2(x)

        z = self.condition(torch.cat(
            (state_global, state_to_global, action),
            dim=-1,
        ))

        x = self.film(x, z)
        x = self.decoder(
            F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        )
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        output = image + x if self.cfg.residual else x

        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
