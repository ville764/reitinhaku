import unittest
import numpy as np
from jps import JPS, step, is_valid, get_neighbors, direction, is_forced, is_diagonal, get_diagonal_components, jump, identify_successors, octile_distance, prune

class TestJPS(unittest.TestCase):

    def setUp(self):
        self.grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        self.jps = JPS(self.grid, heuristic=octile_distance)


    def test_step(self):
        self.assertEqual(step((2, 2), (1, 0)), (3, 2))
        self.assertEqual(step((2, 2), (0, 1)), (2, 3))

    def test_is_valid(self):
        self.assertTrue(is_valid((0, 0), self.grid))
        self.assertFalse(is_valid((3, 1), self.grid))
        self.assertFalse(is_valid((5, 5), self.grid))

    def test_get_neighbors(self):
        self.assertEqual(set(get_neighbors((2, 2), self.grid)), {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)})

    def test_direction(self):
        self.assertEqual(direction((2, 2), (3, 3)), (1, 1))
        self.assertEqual(direction((2, 2), (2, 3)), (0, 1))
        self.assertEqual(direction((2, 2), (3, 2)), (1, 0))

    def test_is_forced(self):
        #testataan, että onko solmu pakotettu x ja y suunnassa
        self.assertTrue(is_forced((3, 2), (2, 2), self.grid))
        self.assertTrue(is_forced((2, 1), (2, 0), self.grid))
        self.assertFalse(is_forced((2, 3), (2, 2), self.grid))
        self.assertFalse(is_forced((1, 1), (1, 0), self.grid))
        self.assertFalse(is_forced((1, 1), (0, 1), self.grid))
        #testataan, että onko solmu pakotettu diagonaalisesti
        self.assertTrue(is_forced((2, 1), (3, 2), self.grid))
        #tästä en ole varma, mutta testataan
        self.assertTrue(is_forced((3, 3), (2, 2), self.grid))
        self.assertFalse(is_forced((1, 0), (0, 1), self.grid))


    def test_is_diagonal(self):
        self.assertTrue(is_diagonal((1, 1)))
        self.assertFalse(is_diagonal((1, 0)))

    def test_get_diagonal_components(self):
        self.assertEqual(set(get_diagonal_components((1, 1))), {(1, 0), (0, 1)})

    def test_prune(self):
        # Tilanne ilman vanhempaa — kaikki naapurit palautetaan
        x = (2, 2)
        neighbors = [(1, 2), (2, 1), (3, 2), (2, 3), (1, 1), (1, 3), (3, 1), (3, 3)]
        self.assertEqual(set(prune(x, neighbors)), set(neighbors))

        # Liike oikealle (2,1) -> (2,2), sallitaan vain (2,3)?
        parent = (2, 1)
        expected = [(2, 3), (1, 3), (3, 3)]
        self.assertEqual(set(prune(x, neighbors, parent)), set(expected))


        # Liike alas (1,2) -> (2,2), sallitaan vain (3,2)?
        parent = (1, 2)
        expected = [(3, 2), (3, 1), (3, 3)]
        self.assertEqual(set(prune(x, neighbors, parent)), set(expected))


        # Liike diagonaalisesti oikea-alas (1,1) -> (2,2), sallitaan vain (3,3)
        parent = (1, 1)
        expected = [(3, 3)]
        self.assertEqual(prune(x, neighbors, parent), expected)

        # Liike vasen-ylä (3,3) -> (2,2), sallitaan vain (1,1)
        parent = (3, 3)
        expected = [(1, 1)]
        self.assertEqual(prune(x, neighbors, parent), expected)

    def test_jump(self):
        self.assertEqual(jump((2, 2), (1, 0), (0, 0), (4, 4), self.grid), (3, 2)) #koska on pakotettu naapuri
        self.assertEqual(jump((2, 2), (0, 1), (0, 0), (4, 4), self.grid), None) #koska ei ole esteitä
        self.assertEqual(jump((3, 3), (1, 1), (0, 0), (4, 4), self.grid), (4, 4))
        self.assertEqual(jump((0, 0), (1, 1), (0, 0), (4, 4), self.grid), (2, 2)) # koska on pakotettu naapuri
        self.assertEqual(jump((0, 4), (1, 0), (0, 0), (4, 4), self.grid), (4, 4)) #ei esteitä eikä pakotettu naapuri
        self.assertEqual(jump((4, 4), (0, -1), (4, 4), (4, 0), self.grid), (4, 2))
        self.assertEqual(jump((0, 3), (1, -1), (0, 3), (4, 0), self.grid), (1, 2)) #pakotettu naapuri 2,1 in 3,1


    def test_identify_successors(self):

        self.assertEqual(set(identify_successors((0, 0), (0, 0), (4, 4), self.grid)), {(2, 2), (2, 0)})
        self.assertEqual(set(identify_successors((1, 4), (0, 4), (4, 4), self.grid)), {(4, 4)}) # antaa myös 2,3, miksi?
        self.assertEqual(set(identify_successors((0, 1), (0, 0), (0, 4), self.grid)), {(4, 4)})
        self.assertEqual(set(identify_successors((1, 2), (0, 1), (4, 4), self.grid)), {(3, 4)}) 

    def test_find_path(self):
        path = self.jps.find_path((0, 0), (4, 4))
        expected_path = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (4, 4)]
        self.assertEqual(path, expected_path)

if __name__ == '__main__':
    unittest.main()
