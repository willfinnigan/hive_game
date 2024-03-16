
def draw_grid():
    top_grid    = "  /      \\"
    bottom_grid = "  \      /"
    empty_space = "          "


    num_cols = 10
    num_rows = 10

    grid = "\033[31m"


    # top row
    grid += empty_space
    for c in range(int(num_cols/2)):
        grid += top_grid
        grid += empty_space
    grid += "\n"

    # all other rows
    for r in range(num_rows):
        for c in range(num_cols):
            if r % 2 == 1:
                if c % 2 == 0:
                    grid += bottom_grid
                elif r == num_rows-1:
                    grid += empty_space
                else:
                    grid += top_grid
            else:
                if c % 2 == 0:
                    grid += top_grid
                else:
                    grid += bottom_grid
        grid += "\n"

    return grid

def draw_grid2():
    top_grid    = "  /      \\"
    bottom_grid = "  \      /"
    empty_space = "          "
    num_cols = 10
    num_rows = 10

    red = "\033[31m"
    blue = "\033[34m"
    lightblue = "\033[94m"
    darkblue = "\033[36m"

    grid = lightblue

    # top row
    for c in range(int(num_cols / 2)):
        grid += top_grid
        grid += empty_space
    grid += "\n"

    for r in range(-1, num_rows):
        for c in range(num_cols):
            if r % 2 == 1:
                if c % 2 == 0:
                    grid += lightblue
                    grid += bottom_grid
                else:
                    grid += darkblue
                    grid += top_grid
            else:
                if c % 2 == 0:
                    grid += lightblue
                    grid += top_grid
                else:
                    grid += darkblue
                    grid += bottom_grid
        grid += "\n"

    # bottom row
    grid += darkblue
    for c in range(int(num_cols / 2)):
        grid += empty_space
        grid += bottom_grid

    grid += "\n"

    return grid