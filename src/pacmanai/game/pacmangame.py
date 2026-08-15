import random
from dataclasses import dataclass

from .grid_gen import (
    generate_maze,
    find_ghost_house,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_GHOST_WIDTH,
    DEFAULT_GHOST_HEIGHT,
    DEFAULT_TUNNEL_LENGTH,
    DEFAULT_LOOP_PROBABILITY,
    DEFAULT_ATTEMPTS,
)

WALL, EMPTY, PELLET = "#", " ", "."

DIRECTIONS = (
    (1, 0),   # right
    (-1, 0),  # left
    (0, 1),   # down
    (0, -1),  # up
)


@dataclass(frozen=True)
class GameSnapshot:
    grid: tuple[str, ...]
    player: tuple[int, int]
    ghosts: tuple[tuple[int, int], ...]
    pellets: frozenset[tuple[int, int]]
    score: int
    lives: int
    running: bool
    game_over: bool


class GameState:
    def __init__(
        self,
        seed=None,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        ghost_width=DEFAULT_GHOST_WIDTH,
        ghost_height=DEFAULT_GHOST_HEIGHT,
        tunnel_length=DEFAULT_TUNNEL_LENGTH,
        loop_probability=DEFAULT_LOOP_PROBABILITY,
        attempts=DEFAULT_ATTEMPTS,
    ):
        # Generate a fresh maze when a new game is created.
        self._grid = tuple(
            generate_maze(
                seed=seed,
                width=width,
                height=height,
                ghost_width=ghost_width,
                ghost_height=ghost_height,
                tunnel_length=tunnel_length,
                loop_probability=loop_probability,
                attempts=attempts,
            )
        )

        self._rows = len(self._grid)
        self._cols = len(self._grid[0])

        self._ghost_width = ghost_width
        self._ghost_height = ghost_height

        # Find the generated maze's ghost house.
        ghost_house = find_ghost_house(
            self._grid,
            ghost_width,
            ghost_height,
        )

        if ghost_house is None:
            raise RuntimeError("Generated maze does not contain a valid ghost house")

        self._ghost_house = ghost_house

        self._player = self._find_player_start()
        self._ghosts = self._find_ghost_starts()

        self._pellets = self._initial_pellets()
        self._score = 0
        self._lives = 3
        self._running = False
        self._game_over = False

        self._validate_state()

    def start(self):
        if not self._game_over:
            self._running = True

    def stop(self):
        self._running = False

    def reset(self):
        # Reset the dynamic state, but keep the same generated maze.
        self._player = self._find_player_start()
        self._ghosts = self._find_ghost_starts()
        self._pellets = self._initial_pellets()
        self._score = 0
        self._lives = 3
        self._running = False
        self._game_over = False

        self._validate_state()

    def get_grid(self):
        return tuple(self._grid)

    def get_player_position(self):
        return tuple(self._player)

    def get_ghost_positions(self):
        return tuple(tuple(position) for position in self._ghosts)

    def get_pellets(self):
        return frozenset(self._pellets)

    def get_score(self):
        return self._score

    def get_lives(self):
        return self._lives

    def is_running(self):
        return self._running

    def is_game_over(self):
        return self._game_over

    def get_dimensions(self):
        return self._rows, self._cols

    def get_state(self):
        return GameSnapshot(
            grid=self._grid,
            player=self._player,
            ghosts=tuple(self._ghosts),
            pellets=frozenset(self._pellets),
            score=self._score,
            lives=self._lives,
            running=self._running,
            game_over=self._game_over,
        )

    def is_wall(self, x, y):
        return (
            y < 0
            or y >= self._rows
            or x < 0
            or x >= self._cols
            or self._grid[y][x] == WALL
        )

    def possible_moves(self, x, y):
        return [
            direction
            for direction in DIRECTIONS
            if not self.is_wall(
                x + direction[0],
                y + direction[1],
            )
        ]

    def get_valid_actions(self):
        return tuple(self.possible_moves(*self._player))

    def step(self, action):
        """
        Advance exactly one timestep.

        action:
            (1, 0)   right
            (-1, 0)  left
            (0, 1)   down
            (0, -1)  up
        """
        if not self._running or self._game_over:
            return self.get_state()

        if action not in DIRECTIONS:
            raise ValueError(f"Invalid action: {action}")

        self._move_player(action)
        self._move_ghosts()
        self._handle_collisions()

        self._validate_state()

        return self.get_state()

    def _initial_pellets(self):
        return {
            (x, y)
            for y, row in enumerate(self._grid)
            for x, cell in enumerate(row)
            if cell == PELLET
        }

    def _find_player_start(self):
        """
        Find a walkable starting position outside the ghost house.

        Prefer the first walkable cell in the maze that is not part
        of the ghost house.
        """
        house_top, house_left = self._ghost_house

        house_cells = {
            (x, y)
            for y in range(
                house_top,
                house_top + self._ghost_height,
            )
            for x in range(
                house_left,
                house_left + self._ghost_width,
            )
        }

        for y in range(self._rows):
            for x in range(self._cols):
                if not self.is_wall(x, y) and (x, y) not in house_cells:
                    return (x, y)

        raise RuntimeError("Generated maze has no valid player starting position")

    def _find_ghost_starts(self):
        """
        Place ghosts inside the generated ghost house.

        The house coordinates returned by find_ghost_house are
        (row, column), while game positions are (x, y).
        """
        top, left = self._ghost_house

        house_cells = [
            (left + x, top + y)
            for y in range(self._ghost_height)
            for x in range(self._ghost_width)
        ]

        # Make sure there are enough cells for the two ghosts.
        if len(house_cells) < 4:
            raise RuntimeError("Ghost house is too small for two ghosts")

        return house_cells[:4]

    def _move_player(self, action):
        x, y = self._player
        dx, dy = action

        nx, ny = x + dx, y + dy

        if not self.is_wall(nx, ny):
            self._player = (nx, ny)

        if self._player in self._pellets:
            self._pellets.remove(self._player)
            self._score += 10

    def _move_ghosts(self):
        new_positions = []

        for x, y in self._ghosts:
            moves = self.possible_moves(x, y)

            if moves:
                dx, dy = random.choice(moves)
                x += dx
                y += dy

            new_positions.append((x, y))

        self._ghosts = new_positions

    def _handle_collisions(self):
        if self._player in self._ghosts:
            self._lives -= 1

            if self._lives <= 0:
                self._game_over = True
                self._running = False
                return

            self._player = self._find_player_start()

        if not self._pellets:
            self._game_over = True
            self._running = False

    def _validate_state(self):
        assert len(self._grid) == self._rows
        assert all(len(row) == self._cols for row in self._grid)

        assert not self.is_wall(*self._player)

        for ghost in self._ghosts:
            assert not self.is_wall(*ghost)

        assert self._lives >= 0
        assert self._score >= 0

    def randomize(self):
        """
        Put the game into a random valid gameplay state.

        The generated maze remains unchanged. Dynamic state is randomized.
        """

        walkable = [
            (x, y)
            for y in range(self._rows)
            for x in range(self._cols)
            if not self.is_wall(x, y)
        ]

        # We need one player + two ghosts.
        if len(walkable) < 3:
            raise RuntimeError("Maze does not contain enough walkable cells")

        positions = random.sample(walkable, 5)

        self._player = positions[0]
        self._ghosts = positions[1:5]

        initial_pellets = self._initial_pellets()

        self._pellets = {
            pellet
            for pellet in initial_pellets
            if random.random() < 0.5
        }

        consumed = len(initial_pellets) - len(self._pellets)
        self._score = consumed * 10

        self._lives = random.randint(1, 3)

        self._running = True
        self._game_over = False

        if not self._pellets:
            self._pellets.add(random.choice(tuple(initial_pellets)))

            consumed = len(initial_pellets) - len(self._pellets)
            self._score = consumed * 10

        self._validate_state()
