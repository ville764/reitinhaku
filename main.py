from grid import create_grid
import numpy as np
from visualization import visualize_astar

grid_size = (10, 10)
obstacles = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (8, 2), (8, 1)]
start = (0, 0)
goal = (9, 9)

# luo 10*10 ruudukko
grid_data_10 = create_grid(grid_size, obstacles)

visualize_astar(grid_data_10, start, goal)

# Määrittele ruudukon koko ja esteet
grid_size = (100, 100)
obstacles = [
    (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (8, 2), (8, 1),
    (20, 20), (20, 21), (20, 22), (21, 22), (22, 22), (23, 22), (24, 22), (25, 22), (26, 22), (27, 22),
    (50, 50), (51, 50), (52, 50), (53, 50), (54, 50), (55, 50), (56, 50), (57, 50), (58, 50), (59, 50),
    (70, 70), (71, 71), (72, 72), (73, 73), (74, 74), (75, 75), (76, 76), (77, 77), (78, 78), (79, 79)
]
start = (0, 0)
goal = (99, 99)

# Luo 100*100 ruudukko ja visualisoi
grid_data_100 = create_grid(grid_size, obstacles)

#visualize_astar(grid_data_100, start, goal)



