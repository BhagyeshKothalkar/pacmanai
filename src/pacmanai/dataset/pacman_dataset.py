from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class PacmanDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        transform: v2.Transform | None = None,
        cache_images: bool = False,
    ) -> None:
        self.root = Path(root)

        self.metadata = json.loads((self.root / "metadata.json").read_text())

        self.snapshots = {
            snapshot["id"]: snapshot
            for snapshot in map(
                json.loads,
                (self.root / "snapshots.jsonl").read_text().splitlines(),
            )
        }

        self.transitions = [
            json.loads(line)
            for line in (self.root / "transitions.jsonl").read_text().splitlines()
        ]

        self.transform = transform or v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(
                    torch.float32,
                    scale=True,
                ),
            ]
        )

        self.cache_images = cache_images

        # IMPORTANT:
        #
        # With num_workers > 0, each DataLoader worker gets its own
        # Dataset instance, so each worker gets its own cache.
        #
        # This is still useful, but it does NOT mean there is one
        # shared cache across workers.
        self._image_cache: dict[str, torch.Tensor] | None = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.transitions)

    def _image(
        self,
        snapshot_id: str,
    ) -> torch.Tensor:
        if self._image_cache is not None and snapshot_id in self._image_cache:
            return self._image_cache[snapshot_id]

        image_path = self.root / self.snapshots[snapshot_id]["frame"]

        # The context manager is important so the underlying
        # file handle is closed promptly.
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        if self._image_cache is not None:
            self._image_cache[snapshot_id] = tensor

        return tensor

    def _state(
        self,
        snapshot_id: str,
    ) -> dict[str, torch.Tensor]:
        state = self.snapshots[snapshot_id]["state"]

        rows = self.metadata["rows"]
        cols = self.metadata["cols"]

        pellets = torch.zeros(
            (rows, cols),
            dtype=torch.bool,
        )

        for x, y in state["pellets"]:
            pellets[y, x] = True

        return {
            "player": torch.tensor(
                state["player"],
                dtype=torch.long,
            ),
            "ghosts": torch.tensor(
                state["ghosts"],
                dtype=torch.long,
            ),
            "pellets": pellets,
            "score": torch.tensor(
                state["score"],
                dtype=torch.long,
            ),
            "lives": torch.tensor(
                state["lives"],
                dtype=torch.long,
            ),
            "running": torch.tensor(
                state["running"],
                dtype=torch.bool,
            ),
            "game_over": torch.tensor(
                state["game_over"],
                dtype=torch.bool,
            ),
        }

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        transition = self.transitions[index]

        before_id = transition["before"]
        after_id = transition["after"]

        event = transition["event"]

        return {
            "image": self._image(before_id),
            "state": self._state(before_id),
            "state_to": self._state(after_id),
            "event": {
                "type": event["type"],
                "key": event["key"],
                "action": torch.tensor(
                    event["action"],
                    dtype=torch.long,
                ),
            },
            "target_image": self._image(after_id),
            "episode": transition["episode"],
            "t": transition["t"],
            "before_id": before_id,
            "after_id": after_id,
        }
