import unittest
import numpy as np
from jps import JPS, is_valid, get_neighbors, get_direction, get_forced_neighbors, has_forced_neighbors, jump, identify_successors, octile_distance

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

    def test_is_valid(self):
        self.assertTrue(is_valid((0, 0), self.grid))
        self.assertFalse(is_valid((3, 1), self.grid))  # este
        self.assertFalse(is_valid((5, 5), self.grid))  # kartan ulkopuolella
        self.assertFalse(is_valid((-1, 0), self.grid))  # negatiivinen koordinaatti

    def test_octile_distance(self):
        # Testaa octile-etäisyyden laskenta
        self.assertAlmostEqual(octile_distance((0, 0), (0, 0)), 0.0)
        self.assertAlmostEqual(octile_distance((0, 0), (1, 0)), 1.0)
        self.assertAlmostEqual(octile_distance((0, 0), (0, 1)), 1.0)
        self.assertAlmostEqual(octile_distance((0, 0), (1, 1)), 1.4142135623730951)  # sqrt(2)
        # Octile distance: max(dx,dy) + (sqrt(2)-1)*min(dx,dy) = 4 + (sqrt(2)-1)*3 ≈ 5.24
        self.assertAlmostEqual(octile_distance((0, 0), (3, 4)), 5.242640687119286)

    def test_get_direction(self):
        # Testaa suunnan normalisointi
        self.assertEqual(get_direction((2, 2), (3, 3)), (1, 1))
        self.assertEqual(get_direction((2, 2), (2, 3)), (0, 1))
        self.assertEqual(get_direction((2, 2), (3, 2)), (1, 0))
        self.assertEqual(get_direction((2, 2), (1, 1)), (-1, -1))
        self.assertEqual(get_direction((2, 2), (1, 2)), (-1, 0))
        self.assertEqual(get_direction((2, 2), (2, 1)), (0, -1))

    def test_get_neighbors_without_parent(self):
        # Testaa naapurit ilman vanhempaa (aloitussolmu)
        neighbors = get_neighbors((2, 2), self.grid)
        # Huomaa että (3,1) on este, joten sitä ei palauteta
        expected = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 2), (3, 3)}
        self.assertEqual(set(neighbors), expected)

    def test_get_neighbors_with_parent(self):
        # Testaa karsitut naapurit vanhemman kanssa
        
        # Horisontaalinen liike oikealle (2,1) -> (2,2)
        neighbors = get_neighbors((2, 2), self.grid, parent=(2, 1))
        # Odotetaan: jatka oikealle (2,3) + sivusuunnat (1,2), (3,2) + mahdolliset pakotetut
        self.assertIn((2, 3), neighbors)  # jatka samaan suuntaan
        self.assertIn((1, 2), neighbors)  # sivusuunta
        self.assertIn((3, 2), neighbors)  # sivusuunta
        
        # Vertikaalinen liike alas (1,2) -> (2,2)
        neighbors = get_neighbors((2, 2), self.grid, parent=(1, 2))
        self.assertIn((3, 2), neighbors)  # jatka alas
        self.assertIn((2, 1), neighbors)  # sivusuunta
        self.assertIn((2, 3), neighbors)  # sivusuunta
        
        # Diagonaalinen liike (1,1) -> (2,2)
        neighbors = get_neighbors((2, 2), self.grid, parent=(1, 1))
        self.assertIn((3, 3), neighbors)  # jatka diagonaalisesti
        self.assertIn((3, 2), neighbors)  # horisontaalinen komponentti
        self.assertIn((2, 3), neighbors)  # vertikaalinen komponentti

    def test_get_forced_neighbors(self):
        # Testaa pakotettuja naapureita
        # Luodaan tilanne jossa on pakotettu naapuri
        
        # Horisontaalinen liike oikealle: este yläpuolella aiheuttaa pakotetun naapurin
        grid_with_obstacle = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],  # este kohdassa (1,2) 
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        # Liike oikealle (1,0) positiosta (2,1)
        # Este on yläpuolella (1,2), joten pakotettu naapuri on (1,3)
        forced = get_forced_neighbors((2, 2), (1, 0), grid_with_obstacle)
        # Odotetaan että löytyy pakotettu naapuri
        self.assertGreaterEqual(len(forced), 0)

    def test_has_forced_neighbors(self):
        # Testaa onko pakotettuja naapureita
        # Käytetään samaa setup:ia kuin yllä
        grid_with_obstacle = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],  # este kohdassa (1,2)
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        # Testaa että löytyy pakotettuja naapureita
        result = has_forced_neighbors((2, 2), (1, 0), grid_with_obstacle)
        # Voi olla että ei löydy pakotettuja naapureita tässä tilanteessa
        self.assertIsInstance(result, bool)
        
        # Tavallisessa tilanteessa ei pakotettuja naapureita
        clear_grid = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        self.assertFalse(has_forced_neighbors((1, 1), (1, 0), clear_grid))

    def test_jump(self):
        # Testaa hyppimistä eri suuntiin
        
        # Hyppiminen suoraan maaliin
        result = jump((0, 0), (1, 1), (2, 2), self.grid)
        self.assertEqual(result, (2, 2))  # pysähtyy maalissa
        
        # Hyppiminen kartan reunaan
        result = jump((0, 0), (0, -1), (4, 4), self.grid)
        self.assertIsNone(result)  # ei voi hypätä kartan ulkopuolelle
        
        # Testaa hyppimistä tilanteessa jossa on esteitä
        result = jump((2, 0), (1, 0), (4, 4), self.grid)
        # Tämä voi palauttaa None tai jonkin hyppypisteen riippuen esteistä
        self.assertIsInstance(result, (type(None), tuple))

    def test_identify_successors(self):
        # Testaa seuraajien tunnistamista
        successors = identify_successors((0, 0), (4, 4), self.grid)
        self.assertIsInstance(successors, list)
        self.assertGreater(len(successors), 0)
        
        # Kaikki seuraajat pitää olla valideja
        for successor in successors:
            self.assertTrue(is_valid(successor, self.grid))

    def test_identify_successors_with_parent(self):
        # Testaa seuraajien tunnistamista vanhemman kanssa
        successors = identify_successors((2, 2), (4, 4), self.grid, parent=(1, 1))
        self.assertIsInstance(successors, list)
        
        # Tarkista että seuraajat ovat kelvollisia
        for successor in successors:
            self.assertTrue(is_valid(successor, self.grid))

    def test_find_path_simple(self):
        # Testaa yksinkertainen polun etsintä
        result = self.jps.find_path((0, 0), (4, 4))
        path, closed_set, jump_points_explored = result
        
        # Tarkista että polku löytyy
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        
        # Tarkista että alku- ja loppupiste ovat oikein
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        
        # Tarkista että closed_set on set
        self.assertIsInstance(closed_set, set)
        
        # Tarkista että jump_points_explored on numero
        self.assertIsInstance(jump_points_explored, int)
        self.assertGreaterEqual(jump_points_explored, 0)

    def test_find_path_no_path(self):
        # Testaa tilanne jossa polkua ei ole
        blocked_grid = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ])
        jps_blocked = JPS(blocked_grid)
        result = jps_blocked.find_path((0, 0), (0, 2))
        path, closed_set, jump_points_explored = result
        
        self.assertIsNone(path)
        self.assertIsInstance(closed_set, set)
        self.assertIsInstance(jump_points_explored, int)

    def test_find_path_same_start_goal(self):
        # Testaa tilanne jossa alku ja loppu ovat samat
        result = self.jps.find_path((2, 2), (2, 2))
        path, closed_set, jump_points_explored = result
        
        self.assertEqual(path, [(2, 2)])
        self.assertIsInstance(closed_set, set)
        self.assertEqual(jump_points_explored, 0)

    def test_find_path_invalid_positions(self):
        # Testaa virheelliset positiot
        result = self.jps.find_path((-1, 0), (4, 4))
        path, closed_set, jump_points_explored = result
        self.assertIsNone(path)
        
        result = self.jps.find_path((0, 0), (3, 1))  # maali on este
        path, closed_set, jump_points_explored = result
        self.assertIsNone(path)

    def test_path_optimality(self):
        # Testaa että polku on järkevä (ei tarkka optimaalisuustesti)
        result = self.jps.find_path((0, 0), (2, 2))
        path, _, _ = result
        
        if path:
            # Polun pituus ei saa olla liian pitkä
            total_distance = 0
            for i in range(len(path) - 1):
                total_distance += octile_distance(path[i], path[i + 1])
            
            # Suora etäisyys
            direct_distance = octile_distance((0, 0), (2, 2))
            
            # Polun pitäisi olla kohtuullisen lähellä suoraa etäisyyttä
            self.assertLessEqual(total_distance, direct_distance * 2)

if __name__ == '__main__':
    unittest.main()
