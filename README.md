# Hive Game

A Python implementation of the board game Hive.

## What is Hive?

Hive is a strategic board game for two players that doesn't require a board. Players place and move their pieces (insects) on a virtual hexagonal grid, with the goal of surrounding the opponent's Queen Bee while protecting their own. The game is characterized by:

- No fixed board - pieces are placed adjacent to each other, forming the playing area
- Different insect pieces with unique movement abilities
- A requirement to maintain the "one hive" rule - all pieces must remain connected
- Victory achieved by completely surrounding the opponent's Queen Bee

## Project Structure

```
hive/
├── agents/         # AI implementations for computer players
├── game/           # Core game mechanics and piece definitions
│   └── pieces/     # Individual piece implementations with movement rules
├── play/           # Game controller and player logic
└── render/         # Visualization options
    ├── text/       # Terminal-based rendering
    └── web/        # Web-based rendering using Pixi.js
tests/
├── test_acceptance/ # Acceptance tests for game functionality
└── test_unit/      # Unit tests for individual components
```

## Main Components

### Game Pieces

The game includes the following pieces, each with unique movement capabilities:

1. **Queen Bee** (`Queen`): Can move one space at a time to any adjacent empty space.
2. **Ant** (`Ant`): Can move to any empty space around the perimeter of the hive.
3. **Beetle** (`Beetle`): Can move one space at a time, including climbing on top of other pieces.
4. **Grasshopper** (`GrassHopper`): Jumps in a straight line over any number of pieces to the first empty space.
5. **Spider** (`Spider`): Moves exactly three spaces around the perimeter of the hive.

All pieces must adhere to the "one hive" rule - a piece cannot be moved if doing so would break the hive into separate pieces.

### AI Agents

The project includes several AI implementations:

1. **Random AI** (`RandomAI`): Makes completely random moves from the available legal moves.
2. **Scored Board State AI** (`ScoreBoardStateAI`): Evaluates moves based on the resulting board state, particularly focusing on how surrounded each queen is.
3. **Scored Moves Based AI** (`ScoreMovesAI`): Scores moves based on piece type and position, allowing for different strategies.

### Game Controller

The `GameController` class manages the game flow:
- Alternates turns between players
- Processes moves and updates the game state
- Determines when the game ends and who the winner is

### Rendering Options

The game provides two rendering options:

1. **Text-based rendering**: Simple ASCII representation of the game board for terminal output.
2. **Web-based rendering**: Uses Pixi.js to create a graphical representation of the game in a web browser.

## Setup and Usage

### Requirements

The project requires Python and the dependencies listed in `requirements.txt`.

### Running the Game

To run a game between two AI players:

```python
from hive.game.game import Game
from hive.game.types_and_errors import WHITE, BLACK
from hive.agents.random_ai import RandomAI
from hive.play.game_controller import GameController

# Create a new game
game = Game()

# Create two AI players
player1 = RandomAI(WHITE)
player2 = RandomAI(BLACK)

# Create a game controller
controller = GameController(player1, player2, game)

# Play the game
controller.play()
```

### Rendering the Game

To render the game in text format:

```python
from hive.render.text.small_print import game_to_text

# Print the current game state
print(game_to_text(game))
```

To render the game in a web browser:

```python
from hive.render.web.render_board import render_board

# Generate an HTML file to view the game
render_board(game)
```

## Testing

The project includes both unit tests and acceptance tests:

- Unit tests verify individual components and game mechanics
- Acceptance tests validate game flow and piece movements

Tests can be run using pytest:

```
pytest tests/