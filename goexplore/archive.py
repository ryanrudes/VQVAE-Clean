from collections import defaultdict
from sys import getsizeof as size
from .cell import Cell

class Archive(defaultdict):
    def __init__(self):
        super().__init__(Cell)

    def __sizeof__(self):
        total = 0
        for cell in self:
            total += size(cell)
        return total
