from typing import TypedDict

import torch


class ModelBatch(TypedDict):
    image: torch.Tensor
    state_map: torch.Tensor
    state_global: torch.Tensor
    state_to_map: torch.Tensor
    state_to_global: torch.Tensor
    action: torch.Tensor
    target_image: torch.Tensor


def _states_to_spatial(
    state: dict[str, torch.Tensor],
    rows: int,
    cols: int,
) -> torch.Tensor:
    """
    Convert batched structured state into spatial feature planes.

    Returns:
        Tensor of shape [B, 1 + G + 1, rows, cols].

        Channels:
            0       : player
            1..G    : ghosts
            G + 1   : pellets
    """

    player: torch.Tensor = state["player"]  # [B, 2]
    ghosts: torch.Tensor = state["ghosts"]  # [B, G, 2]
    pellets: torch.Tensor = state["pellets"]  # [B, R, C]

    batch_size: int = player.shape[0]
    num_ghosts: int = ghosts.shape[1]

    spatial = torch.zeros(
        (batch_size, 1 + num_ghosts + 1, rows, cols),
        dtype=torch.float32,
        device=player.device,
    )

    batch_idx = torch.arange(
        batch_size,
        device=player.device,
    )

    # Player
    px: torch.Tensor = player[:, 0]
    py: torch.Tensor = player[:, 1]

    spatial[batch_idx, 0, py, px] = 1.0

    # Ghosts
    for ghost_idx in range(num_ghosts):
        gx: torch.Tensor = ghosts[:, ghost_idx, 0]
        gy: torch.Tensor = ghosts[:, ghost_idx, 1]

        spatial[
            batch_idx,
            1 + ghost_idx,
            gy,
            gx,
        ] = 1.0

    # Pellets
    spatial[:, -1] = pellets.float()

    return spatial


def _global_state(
    state: dict[str, torch.Tensor],
    *,
    score_scale: float,
    max_lives: float,
) -> torch.Tensor:
    """
    Convert global state variables into normalized features.

    Returns:
        Tensor of shape [B, 4]:
            [score, lives, running, game_over]
    """

    score: torch.Tensor = state["score"].float() / score_scale

    lives: torch.Tensor = state["lives"].float() / max_lives

    running: torch.Tensor = state["running"].float()
    game_over: torch.Tensor = state["game_over"].float()

    return torch.stack(
        [
            score,
            lives,
            running,
            game_over,
        ],
        dim=-1,
    )


def preprocess_batch(
    batch: dict[str, object],
    *,
    device: torch.device,
    rows: int,
    cols: int,
    score_scale: float,
    max_lives: float = 3.0,
) -> ModelBatch:
    """
    Convert a DataLoader batch into model-ready tensors.

    Args:
        batch:
            Batch produced by DataLoader using the default collate_fn.

        device:
            Device on which model inputs should reside.

        rows:
            Number of rows in the Pac-Man game grid.

        cols:
            Number of columns in the Pac-Man game grid.

        score_scale:
            Value used to normalize the score.

        max_lives:
            Maximum number of lives, used to normalize lives.

    Returns:
        ModelBatch containing:

            image:
                [B, 3, H, W]

            state_map:
                [B, 1 + G + 1, R, C]

            state_global:
                [B, 4]

            state_to_map:
                [B, 1 + G + 1, R, C]

            state_to_global:
                [B, 4]

            action:
                [B, 2]

            target_image:
                [B, 3, H, W]
    """

    # These casts are necessary because the DataLoader batch is
    # heterogeneous. A TypedDict for the raw dataset batch would be
    # even cleaner if you want to make this fully statically typed.

    image: torch.Tensor = batch["image"]  # type: ignore[assignment]
    target_image: torch.Tensor = batch["target_image"]  # type: ignore[assignment]

    raw_state: dict[str, torch.Tensor] = batch["state"]  # type: ignore[assignment]

    raw_state_to: dict[str, torch.Tensor] = batch["state_to"]  # type: ignore[assignment]

    state = {
        key: value.to(device, non_blocking=True) for key, value in raw_state.items()
    }

    state_to = {
        key: value.to(device, non_blocking=True) for key, value in raw_state_to.items()
    }

    image = image.to(device, non_blocking=True)
    target_image = target_image.to(device, non_blocking=True)

    # Spatial representations
    state_map = _states_to_spatial(
        state,
        rows=rows,
        cols=cols,
    )

    state_to_map = _states_to_spatial(
        state_to,
        rows=rows,
        cols=cols,
    )

    # Global representations
    state_global = _global_state(
        state,
        score_scale=score_scale,
        max_lives=max_lives,
    )

    state_to_global = _global_state(
        state_to,
        score_scale=score_scale,
        max_lives=max_lives,
    )

    # Action
    event: dict[str, object] = batch["event"]  # type: ignore[assignment]
    action: torch.Tensor = event["action"]  # type: ignore[assignment]
    action = action.to(device, non_blocking=True)

    return {
        "image": image,
        "state_map": state_map,
        "state_global": state_global,
        "state_to_map": state_to_map,
        "state_to_global": state_to_global,
        "action": action,
        "target_image": target_image,
    }
