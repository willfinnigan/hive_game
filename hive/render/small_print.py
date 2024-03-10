from hive.game.game import Game
from hive.game.pieces.ant import Ant
from hive.game.pieces.queen import Queen
from hive.game.types_and_errors import WHITE, BLACK

red = "\033[31m"
blue = "\033[34m"
white = "\033[37m"
black = "\033[30m"
darkblue = "\033[36m"

def game_to_text(game):
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
            row += " "
        for q in range(min_q, max_q+2, 2):
            if (q + r) % 2 == 1:
                q += 1
            piece = game.grid.get((q, r))
            if piece is not None:
                row += f"{piece.as_text()}"
            else:
                row += "-"
            row += " "
        game_text += row + "\n"

    return game_text



if __name__ == "__main__":
    grid = {(6,2): Ant(BLACK),
            (5,1): Ant(WHITE),
            (7,1): Ant(WHITE),
            (8,2): Ant(WHITE),
            (7,3): Ant(WHITE),
            (5,3): Ant(WHITE),
            (4,2): Ant(WHITE),}

    game = Game(grid=grid)
    print(game_to_text(game))


