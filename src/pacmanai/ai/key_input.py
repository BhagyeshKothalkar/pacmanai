from __future__ import annotations

import numpy as np
import pygame
import torch

from pacmanai.ai.model import PacmanFiLM, PacmanFiLMConfig
from pacmanai.game.pacmangame import GameSnapshot, GameState

TILE = 24
FPS = 60

CHECKPOINT_PATH = "checkpoints/pacman_film/best.pt"

SCORE_SCALE = 10_000.0

KEY_DIRECTIONS = {
    pygame.K_RIGHT: (1, 0),
    pygame.K_LEFT: (-1, 0),
    pygame.K_DOWN: (0, 1),
    pygame.K_UP: (0, -1),
}


# -------------------------------------------------------------------
# Device
# -------------------------------------------------------------------


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# -------------------------------------------------------------------
# Initial renderer
#
# This is ONLY used for the first frame.
# -------------------------------------------------------------------


def draw_initial(
    game: GameState,
    screen: pygame.Surface,
) -> None:
    screen.fill("black")

    grid = game.get_grid()

    # Maze.
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "#":
                pygame.draw.rect(
                    screen,
                    "blue",
                    (
                        x * TILE,
                        y * TILE,
                        TILE,
                        TILE,
                    ),
                )

    # Pellets.
    for x, y in game.get_pellets():
        pygame.draw.circle(
            screen,
            "white",
            (
                x * TILE + TILE // 2,
                y * TILE + TILE // 2,
            ),
            3,
        )

    # Player.
    x, y = game.get_player_position()

    pygame.draw.circle(
        screen,
        "yellow",
        (
            x * TILE + TILE // 2,
            y * TILE + TILE // 2,
        ),
        TILE // 2 - 2,
    )

    # Ghosts.
    for x, y in game.get_ghost_positions():
        pygame.draw.circle(
            screen,
            "red",
            (
                x * TILE + TILE // 2,
                y * TILE + TILE // 2,
            ),
            TILE // 2 - 2,
        )

    pygame.display.flip()


# -------------------------------------------------------------------
# Pygame surface -> model image
# -------------------------------------------------------------------


def surface_to_tensor(
    surface: pygame.Surface,
) -> torch.Tensor:
    """
    Convert a Pygame RGB surface into:

        [1, 3, H, W]

    with values in [0, 1].
    """

    array = pygame.surfarray.array3d(surface)

    # Pygame returns [W, H, C].
    array = np.transpose(
        array,
        (1, 0, 2),
    )

    tensor = torch.from_numpy(array.copy()).float() / 255.0

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    return tensor.unsqueeze(0)


# -------------------------------------------------------------------
# Model image -> Pygame surface
# -------------------------------------------------------------------


def tensor_to_surface(
    image: torch.Tensor,
) -> pygame.Surface:
    """
    Convert:

        [1, 3, H, W]

    or

        [3, H, W]

    in [0, 1] into a Pygame surface.
    """

    if image.ndim == 4:
        image = image[0]

    image = image.detach().float().clamp(0, 1).cpu()

    array = image.permute(1, 2, 0).mul(255.0).round().byte().numpy()

    # Pygame expects [W, H, C].
    array = np.transpose(
        array,
        (1, 0, 2),
    )

    return pygame.surfarray.make_surface(array)


# -------------------------------------------------------------------
# GameSnapshot -> model state
# -------------------------------------------------------------------


def snapshot_to_state(
    snapshot: GameSnapshot,
) -> dict[str, torch.Tensor]:
    """
    Convert GameSnapshot into the same structured state
    representation consumed by data_utils.py.

    State format:

        player: [B, 2]
        ghosts: [B, 4, 2]
        pellets: [B, rows, cols]
        score: [B]
        lives: [B]
        running: [B]
        game_over: [B]
    """

    rows = len(snapshot.grid)
    cols = len(snapshot.grid[0])

    player = torch.tensor(
        [snapshot.player],
        dtype=torch.long,
    )

    ghosts = torch.tensor(
        [snapshot.ghosts],
        dtype=torch.long,
    )

    pellets = torch.zeros(
        (1, rows, cols),
        dtype=torch.float32,
    )

    for x, y in snapshot.pellets:
        pellets[0, y, x] = 1.0

    score = torch.tensor(
        [snapshot.score],
        dtype=torch.float32,
    )

    lives = torch.tensor(
        [snapshot.lives],
        dtype=torch.float32,
    )

    running = torch.tensor(
        [snapshot.running],
        dtype=torch.float32,
    )

    game_over = torch.tensor(
        [snapshot.game_over],
        dtype=torch.float32,
    )

    return {
        "player": player,
        "ghosts": ghosts,
        "pellets": pellets,
        "score": score,
        "lives": lives,
        "running": running,
        "game_over": game_over,
    }


def state_to_spatial(
    state: dict[str, torch.Tensor],
    rows: int,
    cols: int,
) -> torch.Tensor:
    """
    Same spatial representation as data_utils._states_to_spatial().
    """

    player = state["player"]
    ghosts = state["ghosts"]
    pellets = state["pellets"]

    spatial = torch.zeros(
        (
            1,
            1 + ghosts.shape[1] + 1,
            rows,
            cols,
        ),
        dtype=torch.float32,
    )

    spatial[
        0,
        0,
        player[0, 1],
        player[0, 0],
    ] = 1.0

    for ghost_idx in range(ghosts.shape[1]):
        gx = ghosts[
            0,
            ghost_idx,
            0,
        ]

        gy = ghosts[
            0,
            ghost_idx,
            1,
        ]

        spatial[
            0,
            1 + ghost_idx,
            gy,
            gx,
        ] = 1.0

    spatial[:, -1] = pellets.float()

    return spatial


def state_to_global(
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Same global representation as data_utils._global_state().
    """

    score = state["score"] / SCORE_SCALE

    lives = state["lives"] / 3.0

    return torch.stack(
        [
            score,
            lives,
            state["running"],
            state["game_over"],
        ],
        dim=-1,
    )


# -------------------------------------------------------------------
# Model transition
# -------------------------------------------------------------------


@torch.no_grad()
def predict_next_frame(
    model: PacmanFiLM,
    current_frame: torch.Tensor,
    current_snapshot: GameSnapshot,
    next_snapshot: GameSnapshot,
    action: tuple[int, int],
    *,
    device: torch.device,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """
    Predict the next rendered frame from:

        current rendered frame
        current symbolic state
        next symbolic state
        action

    The predicted frame is constructed exactly as during training.
    """

    current_state = snapshot_to_state(current_snapshot)

    next_state = snapshot_to_state(next_snapshot)

    state_map = state_to_spatial(
        current_state,
        rows,
        cols,
    )

    state_to_map = state_to_spatial(
        next_state,
        rows,
        cols,
    )

    state_global = state_to_global(current_state)

    state_to_global_tensor = state_to_global(next_state)

    action_tensor = torch.tensor(
        [action],
        dtype=torch.float32,
    )

    current_frame = current_frame.to(device)

    state_map = state_map.to(device)
    state_to_map = state_to_map.to(device)
    state_global = state_global.to(device)
    state_to_global_tensor = state_to_global_tensor.to(device)
    action_tensor = action_tensor.to(device)

    raw_prediction, predicted_mask = model(
        image=current_frame,
        state_map=state_map,
        state_global=state_global,
        state_to_map=state_to_map,
        state_to_global=state_to_global_tensor,
        action=action_tensor,
    )

    # EXACT same construction as training.
    prediction = current_frame + predicted_mask * (raw_prediction - current_frame)

    return prediction.clamp(
        0.0,
        1.0,
    )


# -------------------------------------------------------------------
# Checkpoint
# -------------------------------------------------------------------


def load_model(
    device: torch.device,
) -> PacmanFiLM:
    config = PacmanFiLMConfig()

    model = PacmanFiLM(config).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "loss" in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']}")

    return model


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    device = get_device()

    print(f"Using device: {device}")

    pygame.init()

    game = GameState()

    rows, cols = game.get_dimensions()

    screen = pygame.display.set_mode(
        (
            cols * TILE,
            rows * TILE,
        )
    )

    pygame.display.set_caption("Pac-Man — Neural Renderer")

    clock = pygame.time.Clock()

    # ---------------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------------

    game.start()

    initial_snapshot = game.get_state()

    # ---------------------------------------------------------------
    # FIRST AND ONLY DIRECTLY GENERATED FRAME.
    # ---------------------------------------------------------------

    draw_initial(
        game,
        screen,
    )

    current_frame = surface_to_tensor(screen).to(device)

    current_snapshot = initial_snapshot

    print(f"Initial frame: {tuple(current_frame.shape)}")

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = load_model(device)

    running = True

    while running:
        action = None

        # -----------------------------------------------------------
        # Keyboard
        # -----------------------------------------------------------

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                running = False
                continue

            if event.key in KEY_DIRECTIONS and game.is_running():
                action = KEY_DIRECTIONS[event.key]

        if not running:
            break

        # -----------------------------------------------------------
        # No action this frame.
        # -----------------------------------------------------------

        if action is None:
            clock.tick(FPS)
            continue

        # -----------------------------------------------------------
        # Advance symbolic game state.
        # -----------------------------------------------------------

        next_snapshot = game.step(action)

        # -----------------------------------------------------------
        # Predict next image.
        # -----------------------------------------------------------

        next_frame = predict_next_frame(
            model,
            current_frame,
            current_snapshot,
            next_snapshot,
            action,
            device=device,
            rows=rows,
            cols=cols,
        )

        # -----------------------------------------------------------
        # Display model prediction.
        # -----------------------------------------------------------

        predicted_surface = tensor_to_surface(next_frame)

        screen.blit(
            predicted_surface,
            (0, 0),
        )

        pygame.display.flip()

        # -----------------------------------------------------------
        # Model prediction becomes next input frame.
        # -----------------------------------------------------------

        current_frame = next_frame

        current_snapshot = next_snapshot

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
