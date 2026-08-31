# Pac-Man Driver

import pygame

from .pacmangame import GameState

TILE = 24
FPS = 60

KEY_DIRECTIONS = {
    pygame.K_RIGHT: (1, 0),
    pygame.K_LEFT: (-1, 0),
    pygame.K_DOWN: (0, 1),
    pygame.K_UP: (0, -1),
}


def draw(game, screen, font):
    screen.fill("black")

    grid = game.get_grid()

    # Maze.
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "#":
                pygame.draw.rect(
                    screen,
                    "blue",
                    (x * TILE, y * TILE, TILE, TILE),
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

    # HUD.
    hud = font.render(
        f"Score: {game.get_score()}   Lives: {game.get_lives()}",
        True,
        "white",
    )

    screen.blit(
        hud,
        (8, len(grid) * TILE + 5),
    )

    # Game over.
    if game.is_game_over():
        text = font.render(
            "GAME OVER - Press R to restart",
            True,
            "yellow",
        )

        screen.blit(
            text,
            (
                len(grid[0]) * TILE // 2 - text.get_width() // 2,
                len(grid) * TILE // 2,
            ),
        )

    pygame.display.flip()


def main():
    pygame.init()
    game = GameState()
    rows, cols = game.get_dimensions()
    screen = pygame.display.set_mode((cols * TILE, rows * TILE + 30))
    pygame.display.set_caption("Pac-Man")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    game.start()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                game.stop()
                continue

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_r and game.is_game_over():
                game.reset()
                game.start()
                continue

            if event.key in KEY_DIRECTIONS and game.is_running():
                game.step(KEY_DIRECTIONS[event.key])

        draw(game, screen, font)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
