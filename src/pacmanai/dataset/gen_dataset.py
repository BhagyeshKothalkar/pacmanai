import json
import random
from pathlib import Path

import pygame

from pacmanai.game.pacmangame import GameState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TILE = 24
FPS = 60

ROWS = 19
COLS = 21

NUM_EPISODES = 1250
OUTPUT_DIR = Path("pacman_dataset_new")

EPISODES_PER_SEED = 1
INITIAL_SEED = 41024
MAX_STEPS_PER_EPISODE = 1

KEY_DIRECTIONS = {
    "RIGHT": (1, 0),
    "LEFT": (-1, 0),
    "DOWN": (0, 1),
    "UP": (0, -1),
}

ACTION_TO_KEY = {
    value: key
    for key, value in KEY_DIRECTIONS.items()
}




def next_seed():
    """
    Generate a fresh random maze seed.

    SystemRandom is intentionally used here so that generating a new maze
    seed is independent of the random policy used to select actions.
    """
    return random.SystemRandom().randrange(0, 2**32)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_state(state):
    """
    Serialize the dynamic portion of the complete game state.

    The maze/grid is generated from the episode seed and is stored separately
    in the dataset metadata for each maze seed.
    """

    return {
        "player": list(state.player),
        "ghosts": [list(pos) for pos in state.ghosts],
        "pellets": [list(pos) for pos in sorted(state.pellets)],
        "score": state.score,
        "lives": state.lives,
        "running": state.running,
        "game_over": state.game_over,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw(game, screen, font):
    """
    Render the current GameState to the pygame surface.

    This is deliberately independent of the interactive event loop.
    """

    screen.fill("black")

    grid = game.get_grid()

    # Maze
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "#":
                pygame.draw.rect(
                    screen,
                    "blue",
                    (x * TILE, y * TILE, TILE, TILE),
                )

    # Pellets
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

    # Player
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

    # Ghosts
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

    # # HUD
    # hud = font.render(
    #     f"Score: {game.get_score()}   Lives: {game.get_lives()}",
    #     True,
    #     "white",
    # )

    # screen.blit(
    #     hud,
    #     (8, len(grid) * TILE + 5),
    # )

    pygame.display.flip()


def save_frame(screen, path):
    """
    Save the exact currently rendered frame.
    """

    pygame.image.save(screen, path)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    snapshots_path = OUTPUT_DIR / "snapshots.jsonl"
    transitions_path = OUTPUT_DIR / "transitions.jsonl"

    # -----------------------------------------------------------------------
    # Pygame setup
    # -----------------------------------------------------------------------

    pygame.init()

    # Determine the first maze seed.
    current_seed = (
        INITIAL_SEED
        if INITIAL_SEED is not None
        else next_seed()
    )

    # Generate the first maze.
    game = GameState(
        seed=current_seed,
        width=COLS,
        height=ROWS,
    )

    rows, cols = (ROWS, COLS)

    screen = pygame.display.set_mode(
        (cols * TILE, rows * TILE)
    )
    pygame.display.set_caption("Pac-Man")

    font = pygame.font.Font(None, 24)

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    metadata = {
        "format_version": 2,
        "description": (
            "Pac-Man state-transition dataset. "
            "Each transition references two snapshots: "
            "the state/frame before and after a keypress. "
            "A fixed maze seed is used for multiple episodes before "
            "a new random seed generates the next maze."
        ),
        "rows": ROWS,
        "cols": COLS,
        "tile_size": TILE,
        "directions": {
            key: list(action)
            for key, action in KEY_DIRECTIONS.items()
        },
        "num_episodes": NUM_EPISODES,
        "EPISODES_per_seed": EPISODES_PER_SEED,
        "initial_seed": INITIAL_SEED,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "snapshot_schema": {
            "id": "integer",
            "episode": "integer",
            "t": "integer",
            "seed": "integer",
            "state": {
                "player": "[x, y]",
                "ghosts": "[[x, y], ...]",
                "pellets": "[[x, y], ...]",
                "score": "integer",
                "lives": "integer",
                "running": "boolean",
                "game_over": "boolean",
            },
            "frame": "relative path to PNG",
        },
        "transition_schema": {
            "episode": "integer",
            "t": "integer",
            "seed": "integer",
            "event": {
                "type": "keypress",
                "key": "RIGHT | LEFT | DOWN | UP",
                "action": "[dx, dy]",
            },
            "before": "snapshot id",
            "after": "snapshot id",
        },
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    snapshot_id = 0

    # -----------------------------------------------------------------------
    # JSONL files
    # -----------------------------------------------------------------------

    with (
        open(snapshots_path, "w", encoding="utf-8") as snapshots_file,
        open(transitions_path, "w", encoding="utf-8") as transitions_file,
    ):
        for episode in range(NUM_EPISODES):

            # ---------------------------------------------------------------
            # Choose maze seed.
            #
            # Episodes:
            #
            #   0 ... EPISODES_PER_SEED - 1
            #       -> first seed
            #
            #   EPISODES_PER_SEED ... 2*EPISODES_PER_SEED - 1
            #       -> second seed
            #
            #   etc.
            # ---------------------------------------------------------------

            if episode > 0 and episode % EPISODES_PER_SEED == 0:
                current_seed = next_seed()

                print(
                    f"New maze seed for episode {episode}: "
                    f"{current_seed}"
                )

            # ---------------------------------------------------------------
            # New episode.
            #
            # When the seed changes, create a new GameState so that
            # generate_maze() runs and creates the new maze.
            #
            # Otherwise reset the existing GameState so that the same
            # generated maze is reused.
            # ---------------------------------------------------------------

            if episode == 0 or episode % EPISODES_PER_SEED == 0:
                game = GameState(
                    seed=current_seed,
                    width=COLS,
                    height=ROWS,
                )
            else:
                game.reset()

            # Randomize the dynamic state while keeping the generated maze.
            game.randomize()

            step = 0

            # ---------------------------------------------------------------
            # Initial snapshot
            #
            # This is snapshot 0 for episode 0, and is the state BEFORE
            # the first keypress.
            # ---------------------------------------------------------------

            state = game.get_state()

            draw(game, screen, font)

            frame_path = frames_dir / f"{snapshot_id:08d}.png"
            save_frame(screen, frame_path)

            snapshots_file.write(
                json.dumps(
                    {
                        "id": snapshot_id,
                        "episode": episode,
                        "t": 0,
                        "seed": current_seed,
                        "state": serialize_state(state),
                        "frame": str(
                            frame_path.relative_to(OUTPUT_DIR)
                        ),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

            previous_snapshot_id = snapshot_id
            snapshot_id += 1

            # ---------------------------------------------------------------
            # Generate transitions
            # ---------------------------------------------------------------

            while (
                game.is_running()
                and not game.is_game_over()
                and step < MAX_STEPS_PER_EPISODE
            ):
                valid_actions = game.get_valid_actions()

                if not valid_actions:
                    break

                # -----------------------------------------------------------
                # Autonomous player.
                #
                # For now this is a uniformly random policy.
                # Replace this with whatever policy you eventually want.
                # -----------------------------------------------------------

                action = random.choice(valid_actions)
                key = ACTION_TO_KEY[action]

                # -----------------------------------------------------------
                # Apply the "keypress".
                # -----------------------------------------------------------

                game.step(action)

                step += 1

                # -----------------------------------------------------------
                # Capture resulting state and frame.
                # -----------------------------------------------------------

                state = game.get_state()

                draw(game, screen, font)

                frame_path = frames_dir / f"{snapshot_id:08d}.png"
                save_frame(screen, frame_path)

                snapshots_file.write(
                    json.dumps(
                        {
                            "id": snapshot_id,
                            "episode": episode,
                            "t": step,
                            "seed": current_seed,
                            "state": serialize_state(state),
                            "frame": str(
                                frame_path.relative_to(OUTPUT_DIR)
                            ),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                current_snapshot_id = snapshot_id
                snapshot_id += 1

                # -----------------------------------------------------------
                # Transition
                # -----------------------------------------------------------

                transitions_file.write(
                    json.dumps(
                        {
                            "episode": episode,
                            "t": step - 1,
                            "seed": current_seed,
                            "event": {
                                "type": "keypress",
                                "key": key,
                                "action": list(action),
                            },
                            "before": previous_snapshot_id,
                            "after": current_snapshot_id,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                previous_snapshot_id = current_snapshot_id

            print(
                f"Episode {episode + 1}/{NUM_EPISODES}: "
                f"{step} transitions "
                f"(seed={current_seed})"
            )

    pygame.quit()

    print()
    print(f"Dataset written to: {OUTPUT_DIR}")
    print(f"Snapshots:   {snapshots_path}")
    print(f"Transitions: {transitions_path}")
    print(f"Frames:      {frames_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    generate_dataset()
