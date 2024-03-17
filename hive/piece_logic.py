from hive.grid_functions import one_move_away, is_position_connected, beetle_one_move_away, can_remove_piece
from hive.types import Piece, Grid, PieceName, Location


def get_ant_moves(grid: Grid, loc: Location):
    def _clockwise_moves(grid, loc):
        locations = [loc]
        possible_moves = one_move_away(grid, locations[-1])
        unexplored = [move for move in possible_moves if move not in locations]
        while len(unexplored) > 0:
            locations.append(unexplored[0])
            possible_moves = one_move_away(grid, locations[-1], positions_to_ignore=[loc])
            unexplored = [move for move in possible_moves if move not in locations]

        locations.remove(loc)
        return locations

    def _anticlockwise_moves(grid, loc):
        locations = [loc]
        possible_moves = one_move_away(grid, locations[0], positions_to_ignore=[loc])
        unexplored = [move for move in possible_moves if move not in locations]
        while len(unexplored) > 0:
            locations.append(unexplored[0])
            possible_moves = one_move_away(grid, locations[0])
            unexplored = [move for move in possible_moves if move not in locations]

        locations.remove(loc)
        return locations

    if can_remove_piece(grid, loc) == False:
        return []

    # move 1 piece at a time clockwise and anticlockwise round the board
    clockwise = _anticlockwise_moves(grid, loc)
    anticlockwise = _clockwise_moves(grid, loc)

    moves = clockwise + anticlockwise
    return list(set(moves))

def get_beetle_moves(grid: Grid, loc: Location):
    if can_remove_piece(grid, loc) == False:
        return []
    return beetle_one_move_away(grid, loc, positions_to_ignore=[loc])

def get_grasshopper_moves(grid: Grid, loc: Location):
    if can_remove_piece(grid, loc) == False:
        return []

    line_transformations = [(-1, -1), (+1, -1), (-2, 0), (+2, 0), (-1, +1), (+1, +1)]

    jumps = []
    for transform in line_transformations:
        pieces_jumped = 0
        pos = (loc[0] + transform[0], loc[1] + transform[1])
        while grid.get(pos) is not None:
            pieces_jumped += 1
            pos = (pos[0] + transform[0], pos[1] + transform[1])

        if pieces_jumped == 0:
            continue
        elif is_position_connected(grid, pos, positions_to_ignore=[loc]) == True:
            jumps.append(pos)

    return jumps

def get_queen_moves(grid: Grid, loc: Location):
    if can_remove_piece(grid, loc) == False:
        return []
    return one_move_away(grid, loc)

def get_spider_moves(grid: Grid, loc: Location):
    if can_remove_piece(grid, loc) == False:
        return []

    # first step
    walks = [[loc]]
    new_walks = []
    for walk in walks:
        for move in one_move_away(grid, walk[-1], positions_to_ignore=[loc]):
            if move not in walk:
                new_walks.append(walk + [move])

    # second step
    walks = new_walks
    new_walks = []
    for walk in walks:
        for move in one_move_away(grid, walk[-1], positions_to_ignore=[loc]):
            if move not in walk:
                new_walks.append(walk + [move])

    # third step
    walks = new_walks
    new_walks = []
    for walk in walks:
        for move in one_move_away(grid, walk[-1], positions_to_ignore=[loc]):
            if move not in walk:
                new_walks.append(walk + [move])

    return [walk[-1] for walk in new_walks]



move_functions = {PieceName.ANT: get_ant_moves,
                  PieceName.BEETLE: get_beetle_moves,
                  PieceName.GRASSHOPPER: get_grasshopper_moves,
                  PieceName.QUEEN: get_queen_moves,
                  PieceName.SPIDER: get_spider_moves
                  }

def get_possible_moves(grid: Grid, location: Location):
    stack = grid.get(location)
    if not stack:
        return []

    piece = stack[-1]
    return move_functions[piece.name](grid, location)


