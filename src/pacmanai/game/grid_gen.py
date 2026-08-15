#!/usr/bin/env python3

"""
Random Pac-Man-style ASCII maze generator.

The generator deliberately randomizes both the ordinary maze and the ghost
house position.  The ghost house may be placed anywhere that fits inside the
outer wall, and a random entrance side is connected to the ordinary maze.

Output:
    #   wall
    .   ordinary walkable / pellet cell
        special empty space (ghost house / tunnels)

The seed controls every random decision, so the same seed + configuration is
reproducible while different seeds normally produce different layouts.
"""

import argparse
import random
from collections import deque

WALL = "#"
PELLET = "."
EMPTY = " "

DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

DEFAULT_WIDTH = 21
DEFAULT_HEIGHT = 19
DEFAULT_GHOST_WIDTH = 7
DEFAULT_GHOST_HEIGHT = 5
DEFAULT_TUNNEL_LENGTH = 4
DEFAULT_LOOP_PROBABILITY = 0.12
DEFAULT_ATTEMPTS = 100


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def ghost_house(width, height, ghost_width, ghost_height, top=None, left=None):
    """Return the coordinates occupied by the ghost house.

    If top/left are omitted, retain the old centered position for backwards
    compatibility.  The generator itself always supplies a randomized
    position.
    """
    if top is None:
        top = (height - ghost_height) // 2
    if left is None:
        left = (width - ghost_width) // 2

    return {
        (r, c)
        for r in range(top, top + ghost_height)
        for c in range(left, left + ghost_width)
    }


def ghost_house_entrance(
    width,
    height,
    ghost_width,
    ghost_height,
    top=None,
    left=None,
    side="top",
):
    """Return the cell immediately outside the centered doorway on *side*."""
    if top is None:
        top = (height - ghost_height) // 2
    if left is None:
        left = (width - ghost_width) // 2

    mid_r = top + ghost_height // 2
    mid_c = left + ghost_width // 2

    if side == "top":
        return top - 1, mid_c
    if side == "bottom":
        return top + ghost_height, mid_c
    if side == "left":
        return mid_r, left - 1
    if side == "right":
        return mid_r, left + ghost_width

    raise ValueError("ghost-house entrance side must be top, bottom, left, or right")


def ghost_house_door(
    width,
    height,
    ghost_width,
    ghost_height,
    top=None,
    left=None,
    side="top",
):
    """Return the cell inside the ghost house at the doorway."""
    outside = ghost_house_entrance(
        width, height, ghost_width, ghost_height, top, left, side
    )
    dr, dc = DIRECTIONS[0]
    if side == "top":
        dr, dc = 1, 0
    elif side == "bottom":
        dr, dc = -1, 0
    elif side == "left":
        dr, dc = 0, 1
    elif side == "right":
        dr, dc = 0, -1
    return outside[0] + dr, outside[1] + dc


def random_ghost_house_position(
    width, height, ghost_width, ghost_height, rng
):
    """Pick a legal, non-central-biased house position and entrance side."""
    # Leave a one-cell wall around the house.  Every legal top/left is fair
    # game geometry; there is no special central position anymore.
    top_choices = range(2, height - ghost_height - 1)
    left_choices = range(2, width - ghost_width - 1)

    if not top_choices or not left_choices:
        raise ValueError("ghost house does not fit inside the maze")

    top = rng.choice(list(top_choices))
    left = rng.choice(list(left_choices))
    side = rng.choice(("top", "bottom", "left", "right"))
    return top, left, side


def lattice_nodes(width, height):
    """Return the coarse maze lattice at odd row/odd column coordinates."""
    return [
        (r, c)
        for r in range(1, height - 1, 2)
        for c in range(1, width - 1, 2)
    ]


# ---------------------------------------------------------------------------
# Random maze construction
# ---------------------------------------------------------------------------

def random_spanning_tree(width, height, rng):
    """Build a connected randomized depth-first corridor network."""
    nodes = lattice_nodes(width, height)
    if not nodes:
        raise RuntimeError("No usable maze nodes.")

    node_set = set(nodes)
    start = rng.choice(nodes)
    visited = {start}
    carved = {start}
    stack = [start]

    while stack:
        r, c = stack[-1]
        candidates = []

        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nxt = (r + dr, c + dc)
            if nxt in node_set and nxt not in visited:
                candidates.append(nxt)

        if not candidates:
            stack.pop()
            continue

        nxt = rng.choice(candidates)
        nr, nc = nxt

        visited.add(nxt)
        carved.add(nxt)
        carved.add(((r + nr) // 2, (c + nc) // 2))
        stack.append(nxt)

    return carved


def add_loops(carved, width, height, probability, rng):
    """Add randomized extra connections without breaking connectivity."""
    nodes = set(lattice_nodes(width, height))

    for r, c in nodes:
        if c + 2 < width - 1:
            a, b, wall = (r, c), (r, c + 2), (r, c + 1)
            if a in carved and b in carved and wall not in carved:
                if rng.random() < probability:
                    carved.add(wall)

        if r + 2 < height - 1:
            a, b, wall = (r, c), (r + 2, c), (r + 1, c)
            if a in carved and b in carved and wall not in carved:
                if rng.random() < probability:
                    carved.add(wall)


def connect_ghost_house(
    carved,
    width,
    height,
    ghost_width,
    ghost_height,
    top,
    left,
    side,
):
    """Connect the house doorway to the existing maze.

    A shortest grid path is opened from the outside of the doorway to any
    existing ordinary corridor.  The path never enters the house itself.
    Validation rejects layouts where this connector creates forbidden
    ordinary 2x2 geometry.
    """
    house = ghost_house(
        width, height, ghost_width, ghost_height, top, left
    )
    entrance = ghost_house_entrance(
        width, height, ghost_width, ghost_height, top, left, side
    )

    # The doorway must be inside the grid.  Legal house positions guarantee
    # this, but keeping the check here makes the helper safe on its own.
    if not (0 <= entrance[0] < height and 0 <= entrance[1] < width):
        raise RuntimeError("ghost-house entrance is outside the maze")

    # Search through walls for the nearest existing ordinary maze cell.
    # The search excludes the house, boundaries, and tunnel mouths.
    start = entrance
    queue = deque([start])
    parent = {start: None}
    target = None

    while queue:
        cell = queue.popleft()
        if cell in carved and cell not in house:
            target = cell
            break

        r, c = cell
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            nxt = (rr, cc)

            if not (1 <= rr < height - 1 and 1 <= cc < width - 1):
                continue
            if nxt in house or nxt in parent:
                continue

            parent[nxt] = cell
            queue.append(nxt)

    if target is None:
        raise RuntimeError("could not connect ghost house to maze")

    # Open the connector path.
    cell = target
    while cell is not None:
        carved.add(cell)
        cell = parent[cell]

    # Fill the special house itself.
    carved.update(house)


def add_side_tunnels(carved, width, height, tunnel_length):
    """Add the two side tunnels."""
    middle = height // 2

    for c in range(tunnel_length):
        carved.add((middle, c))

    for c in range(width - tunnel_length, width):
        carved.add((middle, c))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(
    carved,
    width,
    height,
    ghost_width,
    ghost_height,
    tunnel_length,
    ghost_top=None,
    ghost_left=None,
):
    """Convert internal geometry to the requested ASCII format."""
    grid = [[WALL for _ in range(width)] for _ in range(height)]

    house = ghost_house(
        width, height, ghost_width, ghost_height, ghost_top, ghost_left
    )
    middle = height // 2

    for r, c in carved:
        if not (0 <= r < height and 0 <= c < width):
            continue
        if r == 0 or r == height - 1:
            continue
        if c == 0 or c == width - 1:
            if r != middle:
                continue

        if (r, c) in house:
            grid[r][c] = EMPTY
        elif r == middle and (
            c < tunnel_length or c >= width - tunnel_length
        ):
            grid[r][c] = EMPTY
        else:
            grid[r][c] = PELLET

    # Explicit tunnel mouths.
    for c in range(tunnel_length):
        grid[middle][c] = EMPTY
        grid[middle][width - 1 - c] = EMPTY

    # Solid top/bottom boundary.
    for c in range(width):
        grid[0][c] = WALL
        grid[height - 1][c] = WALL

    # Solid left/right boundaries except tunnel row.
    for r in range(height):
        if r != middle:
            grid[r][0] = WALL
            grid[r][width - 1] = WALL

    return ["".join(row) for row in grid]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def flood_fill(grid, start):
    """Return all reachable non-wall cells."""
    height = len(grid)
    width = len(grid[0])

    if grid[start[0]][start[1]] == WALL:
        return set()

    seen = {start}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if grid[rr][cc] == WALL:
                continue
            nxt = (rr, cc)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    return seen


def validate_dimensions(grid, width, height):
    errors = []
    if len(grid) != height:
        errors.append("incorrect height")
    if any(len(row) != width for row in grid):
        errors.append("incorrect width")
    return errors


def validate_characters(grid):
    allowed = {WALL, PELLET, EMPTY}
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch not in allowed:
                return [f"invalid character at ({r}, {c})"]
    return []


def validate_boundary(grid, width, height, tunnel_length):
    errors = []

    if any(ch != WALL for ch in grid[0]):
        errors.append("top boundary is not solid")
    if any(ch != WALL for ch in grid[-1]):
        errors.append("bottom boundary is not solid")

    middle = height // 2
    for r in range(height):
        if r == middle:
            continue
        if grid[r][0] != WALL:
            errors.append("unexpected left boundary opening")
        if grid[r][width - 1] != WALL:
            errors.append("unexpected right boundary opening")

    for c in range(tunnel_length):
        if grid[middle][c] == WALL:
            errors.append("left tunnel is blocked")
        if grid[middle][width - 1 - c] == WALL:
            errors.append("right tunnel is blocked")

    return errors


def validate_connectivity(grid):
    """Every non-wall cell must belong to one connected movement region."""
    walkable = {
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] != WALL
    }

    if not walkable:
        return ["maze contains no walkable cells"]

    start = next(iter(walkable))
    reachable = flood_fill(grid, start)
    unreachable = walkable - reachable

    if unreachable:
        return [f"{len(unreachable)} walkable cells are unreachable"]

    return []


def find_ghost_house(grid, ghost_width, ghost_height):
    """Find the special empty rectangular house in a rendered maze."""
    height = len(grid)
    width = len(grid[0])
    candidates = []

    for top in range(1, height - ghost_height):
        for left in range(1, width - ghost_width):
            house_cells = [
                (r, c)
                for r in range(top, top + ghost_height)
                for c in range(left, left + ghost_width)
            ]
            if all(grid[r][c] == EMPTY for r, c in house_cells):
                candidates.append((top, left))

    if len(candidates) != 1:
        return None
    return candidates[0]


def validate_ghost_house(
    grid,
    width,
    height,
    ghost_width,
    ghost_height,
    ghost_top=None,
    ghost_left=None,
    entrance_side=None,
):
    errors = []

    if ghost_top is None or ghost_left is None:
        found = find_ghost_house(grid, ghost_width, ghost_height)
        if found is None:
            return ["could not uniquely locate ghost house"]
        ghost_top, ghost_left = found

    house = ghost_house(
        width, height, ghost_width, ghost_height, ghost_top, ghost_left
    )

    for r, c in house:
        if grid[r][c] != EMPTY:
            errors.append("ghost house contains non-empty cells")
            break

    if entrance_side is not None:
        entrance = ghost_house_entrance(
            width,
            height,
            ghost_width,
            ghost_height,
            ghost_top,
            ghost_left,
            entrance_side,
        )
        door = ghost_house_door(
            width,
            height,
            ghost_width,
            ghost_height,
            ghost_top,
            ghost_left,
            entrance_side,
        )

        if grid[entrance[0]][entrance[1]] == WALL:
            errors.append("ghost-house entrance is blocked")
        if grid[door[0]][door[1]] == WALL:
            errors.append("ghost-house door is blocked")

        # The doorway must actually lead to another non-house cell.
        if (
            grid[entrance[0]][entrance[1]] == WALL
            or grid[door[0]][door[1]] == WALL
        ):
            errors.append("ghost house is not connected through its entrance")

    return errors


def validate_no_ordinary_2x2(
    grid,
    width,
    height,
    ghost_width,
    ghost_height,
    ghost_top=None,
    ghost_left=None,
):
    """Ordinary maze geometry cannot contain an open 2x2 block."""
    if ghost_top is None or ghost_left is None:
        found = find_ghost_house(grid, ghost_width, ghost_height)
        if found is None:
            return ["could not locate ghost house for 2x2 validation"]
        ghost_top, ghost_left = found

    house = ghost_house(
        width, height, ghost_width, ghost_height, ghost_top, ghost_left
    )
    middle = height // 2

    for r in range(height - 1):
        for c in range(width - 1):
            cells = {
                (r, c),
                (r + 1, c),
                (r, c + 1),
                (r + 1, c + 1),
            }

            if cells & house:
                continue
            if any(rr == middle for rr, cc in cells):
                continue

            if all(grid[rr][cc] != WALL for rr, cc in cells):
                return [f"ordinary open 2x2 at ({r}, {c})"]

    return []


def validate(
    grid,
    width,
    height,
    ghost_width,
    ghost_height,
    tunnel_length,
    ghost_top=None,
    ghost_left=None,
    entrance_side=None,
):
    errors = validate_dimensions(grid, width, height)
    if errors:
        return errors

    errors += validate_characters(grid)
    errors += validate_boundary(grid, width, height, tunnel_length)
    errors += validate_connectivity(grid)
    errors += validate_ghost_house(
        grid,
        width,
        height,
        ghost_width,
        ghost_height,
        ghost_top,
        ghost_left,
        entrance_side,
    )
    errors += validate_no_ordinary_2x2(
        grid,
        width,
        height,
        ghost_width,
        ghost_height,
        ghost_top,
        ghost_left,
    )
    return errors


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_maze(
    seed,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    ghost_width=DEFAULT_GHOST_WIDTH,
    ghost_height=DEFAULT_GHOST_HEIGHT,
    tunnel_length=DEFAULT_TUNNEL_LENGTH,
    loop_probability=DEFAULT_LOOP_PROBABILITY,
    attempts=DEFAULT_ATTEMPTS,
):
    """
    Generate one deterministic random Pac-Man maze.

    Returns:
        list[str]
    """
    if width < 11 or height < 11:
        raise ValueError("width and height must both be >= 11")
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("width and height must be odd")
    if ghost_width % 2 == 0 or ghost_height % 2 == 0:
        raise ValueError("ghost dimensions must be odd")
    if ghost_width > width - 4:
        raise ValueError("ghost house is too wide")
    if ghost_height > height - 4:
        raise ValueError("ghost house is too tall")
    if tunnel_length < 1:
        raise ValueError("tunnel_length must be positive")
    if 2 * tunnel_length > width:
        raise ValueError("tunnels are too long")
    if not 0 <= loop_probability <= 1:
        raise ValueError("loop_probability must be between 0 and 1")

    rng = random.Random(seed)

    for _ in range(attempts):
        # 1. Generate a completely connected ordinary maze first.
        carved = random_spanning_tree(width, height, rng)

        # 2. Randomly introduce loops.
        add_loops(carved, width, height, loop_probability, rng)

        # 3. Randomize the ghost house position and entrance.
        top, left, side = random_ghost_house_position(
            width, height, ghost_width, ghost_height, rng
        )

        # 4. Connect the house to the ordinary maze, then add the tunnels.
        connect_ghost_house(
            carved,
            width,
            height,
            ghost_width,
            ghost_height,
            top,
            left,
            side,
        )
        add_side_tunnels(carved, width, height, tunnel_length)

        # 5. Render and reject layouts that violate the formal constraints.
        grid = render(
            carved,
            width,
            height,
            ghost_width,
            ghost_height,
            tunnel_length,
            top,
            left,
        )

        errors = validate(
            grid,
            width,
            height,
            ghost_width,
            ghost_height,
            tunnel_length,
            top,
            left,
            side,
        )

        if not errors:
            return grid

    raise RuntimeError(
        f"seed {seed} failed to produce a valid maze after {attempts} attempts"
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a random Pac-Man-style ASCII maze."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--ghost-width", type=int, default=DEFAULT_GHOST_WIDTH)
    parser.add_argument("--ghost-height", type=int, default=DEFAULT_GHOST_HEIGHT)
    parser.add_argument("--tunnel-length", type=int, default=DEFAULT_TUNNEL_LENGTH)
    parser.add_argument("--loop-probability", type=float, default=DEFAULT_LOOP_PROBABILITY)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    maze = generate_maze(
        seed=args.seed,
        width=args.width,
        height=args.height,
        ghost_width=args.ghost_width,
        ghost_height=args.ghost_height,
        tunnel_length=args.tunnel_length,
        loop_probability=args.loop_probability,
        attempts=args.attempts,
    )

    print("\n".join(maze))

    if args.validate:
        found = find_ghost_house(maze, args.ghost_width, args.ghost_height)
        top, left = found if found else (None, None)

        errors = validate(
            maze,
            args.width,
            args.height,
            args.ghost_width,
            args.ghost_height,
            args.tunnel_length,
            top,
            left,
        )

        print()
        if errors:
            print("INVALID")
            for error in errors:
                print(" -", error)
        else:
            print("VALID")


if __name__ == "__main__":
    main()
