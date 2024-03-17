from hive.game import Game
from hive.types import Piece, WHITE, PieceName, BLACK

red = "\033[31m"
blue = "\033[34m"
white = "\033[37m"
black = "\033[30m"
darkblue = "\033[36m"
yellow = "\033[33m"
reset = "\033[0m"


def piece_as_test(piece: Piece, highlight=False):
    piece_letter = piece.name.name[0]
    if highlight:
        return f"{red}{piece_letter}{piece.number}{reset}"
    elif piece.colour == WHITE:
        return f"{yellow}{piece_letter}{piece.number}{reset}"
    else:
        return f"{darkblue}{piece_letter}{piece.number}{reset}"


def game_to_text(game, highlight_piece_at=None, show_moves=None):
    min_r = min([loc[1] for loc in game.grid.keys()])
    max_r = max([loc[1] for loc in game.grid.keys()])
    max_q = max([loc[0] for loc in game.grid.keys()])
    min_q = min([loc[0] for loc in game.grid.keys()])


    if min_q % 2 == 1:
        min_q -= 1

    game_text = ""
    for r in range(min_r, max_r+1):
        row = f""
        if r % 2 != 0:
            row += "  "
        for q in range(min_q, max_q+2, 2):
            if (q + r) % 2 == 1:
                q += 1
            stack = game.grid.get((q, r), [])
            if stack != []:
                piece = stack[-1]
                if (q, r) == highlight_piece_at:
                    row += f"{piece_as_test(piece, highlight=True)}"
                else:
                    row += f"{piece_as_test(piece)}"
            else:
                row += "--"
            row += "  "
        game_text += row + "\n"

    return game_text



if __name__ == "__main__":
    grid = {(6,2): [Piece(BLACK, PieceName.ANT, 1)],
            (5,1): [Piece(WHITE, PieceName.ANT, 2)],
            (7,1): [Piece(BLACK, PieceName.ANT, 3)],
            (8,2): [Piece(WHITE, PieceName.ANT, 4)],
            (7,3): [Piece(BLACK, PieceName.ANT, 5)],
            (5,3): [Piece(WHITE, PieceName.ANT, 6)],
            (4,2): [Piece(BLACK, PieceName.ANT, 6)]}

    game = Game(grid=grid)
    print(game_to_text(game, highlight_piece_at=(6,2)))


