import unittest
import numpy as np
import math
from astar import AStar, get_neighbors, octile_distance

class TestAStar(unittest.TestCase):

    def setUp(self):
        self.grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        self.astar = AStar(self.grid, heuristic=octile_distance)

    def test_get_neighbors(self):
        neighbors = get_neighbors((2, 2), self.grid)
        expected = [(1, 2), (3, 2), (2, 1), (2, 3), (1, 1), (1, 3), (3, 3)]
        self.assertEqual(set(neighbors), set(expected))

    def test_octile_distance(self):
        self.assertAlmostEqual(octile_distance((0, 0), (3, 4)), 4 + (math.sqrt(2) - 1) * 3)

    def test_find_path_simple(self):
        path, closed_set, nodes_added = self.astar.find_path((0, 0), (4, 4))
        expected_path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        self.assertEqual(path, expected_path)

    def test_find_path_blocked(self):
        # Esteet estävät suoran reitin, mutta algoritmi löytää tehokkaan diagonaalisen reitin
        path, closed_set, nodes_added = self.astar.find_path((4, 0), (4, 4))
        # Algoritmi käyttää diagonaalisia liikkeitä tehokkaasti
        expected_path = [(4, 0), (3, 0), (2, 1), (3, 2), (3, 3), (4, 4)]
        self.assertEqual(path, expected_path)
        
        # Varmistetaan että polku on oikean pituinen (octile-etäisyys)
        total_distance = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            dx = abs(next_node[0] - current[0])
            dy = abs(next_node[1] - current[1])
            # Diagonaalinen liike = √2, ortogonaalinen = 1
            distance = math.sqrt(2) if dx + dy == 2 else 1
            total_distance += distance
        
        # Tarkista että polku on optimaalinen
        expected_distance = 1 + math.sqrt(2) + math.sqrt(2) + math.sqrt(2) + 1
        self.assertAlmostEqual(total_distance, expected_distance, places=6)
    
    def test_no_path(self):
        # Täysin estetty reitti
        blocked_grid = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ])
        astar_blocked = AStar(blocked_grid)
        path, closed_set, nodes_added = astar_blocked.find_path((0, 0), (2, 2))
        self.assertIsNone(path)
    
    def test_path_validity(self):
        #Tarkistaa että löydetty polku on kelvollinen
        path, closed_set, nodes_added = self.astar.find_path((4, 0), (4, 4))
        
        # Varmista että polku alkaa ja päättyy oikeisiin pisteisiin
        self.assertEqual(path[0], (4, 0))
        self.assertEqual(path[-1], (4, 4))
        
        # Varmista että kaikki polun pisteet ovat vapaita
        for point in path:
            self.assertEqual(self.grid[point[0]][point[1]], 0)
        
        # Varmista että peräkkäiset pisteet ovat naapureita
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            dx = abs(next_point[0] - current[0])
            dy = abs(next_point[1] - current[1])
            # Naapurit voivat olla korkeintaan 1 askeleen päässä (mukaan lukien diagonaalit)
            self.assertTrue(dx <= 1 and dy <= 1 and (dx + dy) > 0)

if __name__ == '__main__':
    unittest.main()