import numpy as np

def create_grid(size, obstacles):
    grid = np.zeros(size)
    for obstacle in obstacles:
        grid[obstacle] = 1
    return grid

