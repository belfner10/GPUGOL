import numpy as np
from numba import jit, cuda
import os
import time
import math


def draw_grid(cur):
    os.system('cls')
    output = '+' + ((cur.shape[1]) * '--') + '+\n'
    for row in cur:
        output += '|'
        for cell in row:
            if cell == 1:
                output += '██'
            else:
                output += '  '
        output += '|\n'
    output += ('+' + ((cur.shape[1]) * '--') + '+\n')
    print(output)


@cuda.jit
def kernal(A, B, size):
    x, y = cuda.grid(2)
    if x >= size[1] or y >= size[0]:
        return
    num_surround = 0
    for ty in range(max(0, y - 1), min(A.shape[0], y + 2)):
        for tx in range(max(0, x - 1), min(A.shape[1], x + 2)):
            num_surround += A[ty, tx]
    num_surround -= A[y, x]

    if (A[y, x] == 1 and (num_surround == 2 or num_surround == 3)) or (A[y, x] == 0 and num_surround == 3):
        B[y, x] = 1
    else:
        B[y, x] = 0


@jit(nopython=True)
def step(cells):
    updated = np.zeros(cells.shape)
    for y in range(cells.shape[0]):
        for x in range(cells.shape[1]):
            num_surround = np.sum(cells[max(0, y - 1):min(cells.shape[0], y + 2), max(0, x - 1):min(cells.shape[1], x + 2)]) - cells[y, x]
            # num_surround = np.sum(cells[y - 1:y + 2, x - 1:x + 2]) - cells[y, x]
            if (cells[y, x] == 1 and (num_surround == 2 or num_surround == 3)) or (cells[y, x] == 0 and num_surround == 3):
                updated[y, x] = 1
    return updated


def init(width, height):
    # cells = np.random.choice([0, 1], size=(height, width), p=[7 / 10, 3 / 10])
    # Taken from https://github.com/beltoforion/recreational_mathematics_with_python/blob/master/game_of_life.py
    cells = np.zeros((height, width), dtype=int)
    pattern = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int);
    pos = (3, 3)
    cells[pos[0]:pos[0] + pattern.shape[0], pos[1]:pos[1] + pattern.shape[1]] = pattern
    return cells


def main(width, height):
    size = (height, width)
    np.random.seed(100)
    cells = init(width, height)

    A_global_mem = cuda.to_device(cells)
    b = np.zeros(size, dtype=int)
    B_global_mem = cuda.to_device(b)

    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(cells.shape[1] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(cells.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start = time.perf_counter()
    cycles = 1000
    sign = True
    for _ in range(cycles):
        if sign:
            kernal[blockspergrid, threadsperblock](A_global_mem, B_global_mem, size)
            cells_nv = B_global_mem.copy_to_host()
        else:
            kernal[blockspergrid, threadsperblock](B_global_mem, A_global_mem, size)
            cells_nv = A_global_mem.copy_to_host()
        sign = not sign
        # cells = step(cells)
        # print(np.array_equal(cells, cells_nv))
        # draw_grid(cells)
        # draw_grid(cells_nv)

    end = time.perf_counter()
    print(f'Cycles per second: {cycles / (end - start):0.0f}')


if __name__ == "__main__":
    main(400, 400)
