from collections import namedtuple
import numpy as np
from typing import List, Optional
import copy


Cell = namedtuple('Cell', 'X Y')


class Block:
    HORIZONTAL = 0
    VERTICAL = 1

    def __init__(self, length: int, index: int, direction: int, previous: 'Block' = None):
        self.length = length
        self.index = index
        assert direction in (Block.HORIZONTAL, Block.VERTICAL)
        self.direction = direction
        self.previous: Block = previous
        self.next: Block = None

        if self.previous is not None:
            self.previous.next = self
            self.blocks_in_line = self.previous.blocks_in_line
        else:
            self.blocks_in_line = []
        self.blocks_in_line.append(self)

        self.possibilities: List[np.ndarray] = []
        self.solution = None
        self.solved = False

    def __str__(self):
        return f'<Block({self.length}) on {"row" if self.direction == Block.HORIZONTAL else "column"} {self.index+1}>'

    def __repr__(self):
        return str(self)

    def min_end(self):
        minimum = len(self.possibilities[0])
        for possibility in self.possibilities:
            minimum = min(minimum, np.argwhere(possibility == 1).max())
        return minimum

    def max_start(self):
        maximum = 0
        for possibility in self.possibilities:
            maximum = max(maximum, np.argwhere(possibility == 1).min())
        return maximum

    def can_fill_cell(self, cell: Cell):
        if self.direction == Block.HORIZONTAL and cell.Y != self.index:
            return False
        if self.direction == Block.VERTICAL and cell.X != self.index:
            return False
        idx = cell.X if self.direction == Block.HORIZONTAL else cell.Y
        for possibility in self.possibilities:
            if possibility[idx] == 1:
                return True
        return False

    def get_grid_line(self, grid) -> np.ndarray:
        if self.direction == Block.HORIZONTAL:
            return grid[self.index, :]
        else:
            return grid[:, self.index]

    def clear(self):
        self.possibilities.clear()
        self.solution = None
        self.solved = False

    def generate(self, grid: np.ndarray):
        line = self.get_grid_line(grid)
        for i in range(len(line) - self.length + 1):
            possibility = [0]*i + [1]*self.length + [0] * (len(line) - self.length - i)
            if i > 0:
                possibility[i-1] = 2
            if i + self.length < len(line):
                possibility[i + self.length] = 2
            self.possibilities.append(np.array(possibility, dtype=np.uint8))

    def reduce(self, grid):
        # No need to process this block if it's already solved
        if self.solved:
            return False

        line = self.get_grid_line(grid)
        progress = False

        # Check which possibilities should be thrown out based on where the neighboring blocks might be
        # Also, check which possibilities don't fit with already filled in cells in the grid
        min_start = self.previous.min_end() + 1 if self.previous is not None else 0
        max_end = self.next.max_start() if self.next is not None else len(line)
        for i, possibility in list(enumerate(self.possibilities))[::-1]:
            # Check minimum start point of possibility
            if np.argwhere(possibility > 0).min() < min_start:
                self.possibilities.pop(i)
                progress = True
                continue

            # Check maximum end point of possibility
            if np.argwhere(possibility > 0).max() >= max_end:
                self.possibilities.pop(i)
                progress = True
                continue

            # Throw out possibilities that mismatch with already filled in cells in the grid
            refline = line.copy()
            np.putmask(refline, possibility == 0, 0)        # Ignore cells outside of this possibility
            if np.any(np.logical_and(refline != possibility, refline != 0)):
                # If any cells in this possibility mismatch the grid - ignoring empty cells
                self.possibilities.pop(i)
                progress = True
                continue

        # Check if any of the squares (1) already in the grid should be `filled` by this block
        # If this block can reach it (a `possibility` in the list that covers that cell) and no other can,
        #  this block _has_ to cover it - so throw out all possibilities that don't
        for idx, value in enumerate(line):
            if self.direction == Block.HORIZONTAL:
                cell = Cell(idx, self.index)
            else:
                cell = Cell(self.index, idx)
            if value == 1 and self.can_fill_cell(cell):
                others_can_fill = False
                # Loop over all blocks in this line, and see if any of them can fill the cell
                for block in self.blocks_in_line:
                    if block is not self and block.can_fill_cell(cell):
                        others_can_fill = True
                        break
                if not others_can_fill:
                    # This block should fill this cell
                    for i, possibility in list(enumerate(self.possibilities))[::-1]:
                        if possibility[idx] != 1:
                            self.possibilities.pop(i)
                            progress = True

        equal_cells = np.all(self.possibilities == self.possibilities[0], axis=0)   # Get cells that are the same in all possibilities
        equal_cells = np.logical_and(equal_cells, self.possibilities[0] > 0)        # Filter out empty cells
        if np.any(equal_cells):
            solution = np.zeros_like(line, dtype=np.uint8)
            np.copyto(solution, self.possibilities[0], where=equal_cells)           # Copy known cells to `solution`
            if np.any(solution != self.solution):                                   # Only copy known cells if we know more than last time
                np.copyto(line, solution, where=equal_cells)                        # Copy solution to line, which is a view to the grid
                self.solution = solution
                progress = True
        return progress


class Grid(np.ndarray):
    @classmethod
    def new(cls, width, height):
        arr = np.zeros((width, height), dtype=np.uint8)
        return arr.view(Grid)

    def __str__(self):
        # ☒□■
        chars = {0: '□', 1: '■', 2: '☒'}
        # rows = []
        if len(self.shape) == 2:
            idx_width = len(str(self.shape[0]))
            header = ' '*(idx_width+1) + ''.join(chr(ord('①')+i) for i in range(self.shape[1])) + '\n'
            return header + '\n'.join(f'{i+1:{idx_width}} ' + str(self[i,:]) for i in range(self.shape[0]))
        elif len(self.shape) == 1:
            return ''.join([chars.get(self[i], '�') for i in range(self.shape[0])])
        else:
            return np.ndarray.__str__(self)


class Nonogram:
    def __init__(self, rows, cols):
        # Assert the columns and rows have the same number of blocks - basic error check
        assert sum(block for blocks in rows for block in blocks) == sum(block for blocks in cols for block in blocks)
        self.row_hints = rows
        self.col_hints = cols
        # self.grid = Grid.new(len(self.col_hints), len(self.row_hints))
        self.grid = Grid.new(len(self.row_hints), len(self.col_hints))

        self.rows: List[List[Block]] = []
        self.cols: List[List[Block]] = []
        self.create_blocks(self.row_hints, self.col_hints)

    def create_blocks(self, rows, cols):
        for row, hints in enumerate(rows):
            previous: Block = None
            self.rows.append([])
            for length in hints:
                block = Block(length, row, Block.HORIZONTAL, previous)
                self.rows[-1].append(block)
                previous = block
        for col, hints in enumerate(cols):
            previous: Block = None
            self.cols.append([])
            for length in hints:
                block = Block(length, col, Block.VERTICAL, previous)
                self.cols[-1].append(block)
                previous = block

    def clear(self):
        # Reset grid and all blocks
        for row in self.rows:
            for block in row:
                block.clear()
        for col in self.cols:
            for block in col:
                block.clear()
        self.grid[:,:] = 0

    def generate(self):
        # For each block, generate all possible locations that it can be in
        for row in self.rows:
            for block in row:
                block.generate(self.grid)
        for col in self.cols:
            for block in col:
                block.generate(self.grid)

    def reduce(self):
        # For each block, remove predicted locations that the block can no longer reach
        # Also, put crosses in cells that no block can reach
        progress = False
        for row, blocks in enumerate(self.rows):
            # Reduce all blocks in rows
            for block in blocks:
                progress |= block.reduce(self.grid)
            # Set cells that no blocks can fill to crosses (2)
            for col in range(self.grid.shape[1]):
                if self.grid[row, col] == 0:
                    cell = Cell(col, row)
                    if not any(block.can_fill_cell(cell) for block in blocks):
                        self.grid[row, col] = 2
                        progress = True

        for col, blocks in enumerate(self.cols):
            # Reduce all blocks in columns
            for block in blocks:
                progress |= block.reduce(self.grid)
            # Set cells that no blocks can fill to crosses (2)
            for row in range(self.grid.shape[0]):
                if self.grid[row, col] == 0:
                    cell = Cell(col, row)
                    if not any(block.can_fill_cell(cell) for block in blocks):
                        self.grid[row, col] = 2
                        progress = True
        return progress

    def reduce_loop(self):
        while True:
            if not self.reduce():
                break

    def solve(self):
        self.clear()
        self.generate()
        self.reduce_loop()
        return 0 not in self.grid   # If there are no empty cells (0) left, we solved the whole puzzle - return True


if __name__ == '__main__':
    # Nonogram.com Level 8
    # rows = [(3,), (5,), (6,), (1,3), (3,), (3,), (3,), (3,4), (3,1,2), (4,4)]
    # cols = [(2,), (2,4), (3,5), (10,), (6,1), (4,1), (1,1), (1,1), (3,), (3,)]

    # Nonogram.com - 18 March 2021
    # rows = [(2,2,1), (3,7), (1,4,2), (2,1,1,3), (2,2,1), (2,3,2), (3,2,2,3), (6,5), (4,7), (3,2,2,2), (1,2,1), (4,1,2), (5,3), (4,2,3), (3,2,3)]
    # cols = [(2,1,3,1), (2,2,1,1,1), (1,1,2,1,2), (2,3,2,2), (5,4,1), (2,5,2), (2,4,1), (4,1,3), (2,3,3), (3,2,2,2), (5,5,1), (3,2), (1,3,3), (2,4,2), (2,3,1)]

    # Pi π
    rows = [(1,), (2,), (3,), (18,), (18,), (18,), (17,), (2, 3, 3), (1, 3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (4, 3), (4, 3, 1), (5, 4, 2), (4, 8), (4, 8), (4, 6), (2, 3)]
    cols = [(3,), (3,), (3, 4), (4, 7), (4, 9), (21,), (18,), (13,), (4,), (4,), (4,), (4,), (20,), (21,), (22,), (4, 5), (4, 4), (3, 3), (4, 3), (3, 3), (3, 2), (2,)]
    puzzle = Nonogram(rows, cols)
    puzzle.solve()
    print(puzzle.grid)
    if 0 in puzzle.grid:
        print('Not completely solved!')
    else:
        print('Solved!')
