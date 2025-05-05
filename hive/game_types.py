from enum import Enum
from typing import NamedTuple, List, Tuple, Dict
from typing import Tuple, Dict
from pyrsistent import PRecord, pmap_field, PMap

class PieceName(Enum):
    ANT = 'ANT'
    BEETLE = 'BEETLE'
    GRASSHOPPER = 'GRASSHOPPER'
    QUEEN = 'QUEEN'
    SPIDER = 'SPIDER'
    BLANK = 'BLANK'

Colour = str
WHITE = 'WHITE'
BLACK = 'BLACK'

Piece = NamedTuple('Piece', [('colour', Colour), ('name', PieceName), ('number', int)])
Stack = List[Piece]

Location = Tuple[int, int]
Grid = Dict[Location, Stack]

GridLocation = NamedTuple('GridLocation', [('stack', List[Piece]), ('location', Location)])


