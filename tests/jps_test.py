import unittest
import numpy as np
from jps import JPS, step, is_valid, get_neighbors, direction, is_forced, is_diagonal, get_diagonal_components, jump, identify_successors, octile_distance

class TestJPS(unittest.TestCase):

    def setUp(self):
        self.grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        self.jps = JPS(self.grid, heuristic=octile_distance)


    def test_step(self):
        self.assertEqual(step((2, 2), (1, 0)), (3, 2))
        self.assertEqual(step((2, 2), (0, 1)), (2, 3))

    def test_is_valid(self):
        self.assertTrue(is_valid((0, 0), self.grid))
        self.assertFalse(is_valid((1, 1), self.grid))
        self.assertFalse(is_valid((5, 5), self.grid))

    def test_get_neighbors(self):
        self.assertEqual(set(get_neighbors((2, 2), self.grid)), {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)})

    def test_direction(self):
        self.assertEqual(direction((2, 2), (3, 3)), (1, 1))
        self.assertEqual(direction((2, 2), (2, 3)), (0, 1))

    def test_is_forced(self):
        self.assertTrue(is_forced((2, 1), (2, 2), self.grid))
        self.assertFalse(is_forced((2, 3), (2, 2), self.grid))

    def test_is_diagonal(self):
        self.assertTrue(is_diagonal((1, 1)))
        self.assertFalse(is_diagonal((1, 0)))

    def test_get_diagonal_components(self):
        self.assertEqual(set(get_diagonal_components((1, 1))), {(1, 0), (0, 1)})

    def test_jump(self):
        self.assertEqual(jump((2, 2), (1, 0), (0, 0), (4, 4), self.grid), (3, 2))
        self.assertEqual(jump((2, 2), (0, 1), (0, 0), (4, 4), self.grid), (2, 3))

    def test_identify_successors(self):
        self.assertEqual(set(identify_successors((2, 2), (0, 0), (4, 4), self.grid)), {(3, 2), (2, 3)})

    def test_find_path(self):
        path = self.jps.find_path((0, 0), (4, 4))
        expected_path = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (4, 4)]
        self.assertEqual(path, expected_path)

if __name__ == '__main__':
    unittest.main()
