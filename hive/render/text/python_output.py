from hive.game.game import Game


def game_as_text(game: Game, border=0, show_positions=False) -> str:
    """Convert a game to a text representation.

    Args:
        game (Game): The game object to convert.
        border (int): The number of empty spaces to add around the grid.
        show_positions (bool): Whether to show the positions of the pieces.

    Returns:
        str: The text representation of the game.
    """

    # find the extreme x value and y value - min is top left corner
    min_q = min([loc[0] for loc in game.grid.keys()])
    min_r = min([loc[1] for loc in game.grid.keys()])
    max_q = max([loc[0] for loc in game.grid.keys()])
    max_r = max([loc[1] for loc in game.grid.keys()])

    # loop over y values - these are the rows
    rows = []
    for r in range(min_r - border, max_r + 1 + border):
        row = [f'']

        # loop over x values - these are the columns
        for q in range(min_q-border-1, max_q + 1 + border, 2):

            # if q+r is odd, then add 1 to q
            if (q + r) % 2 == 1:
                q += 1

            piece = game.grid.get((q, r))
            if piece is not None:
                row.append(f"({piece.as_text_colour():^8})")
            else:
                if show_positions == True:
                    row.append(f"({q:^3}, {r:^3})")
                else:
                    row.append(f"{'':^8}")

        row_text = "".join(row)
        if r % 2 == 1:  # every other row is indented by 2 spaces
            row_text = "    " + row_text

        rows.append(row_text)

    return "\n".join(rows)


if __name__ == "__main__":
    from hive.game.pieces.ant import Ant
    from hive.game.types_and_errors import WHITE, BLACK

    grid = {(6, 2): Ant(BLACK),
            (5, 1): Ant(WHITE),
            (7, 1): Ant(WHITE),
            (8, 2): Ant(WHITE),
            (7, 3): Ant(WHITE),
            (5, 3): Ant(WHITE),
            (4, 2): Ant(WHITE), }

    game = Game(grid=grid)
    print(game_as_text(game))