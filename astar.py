import heapq
import math

def octile_distance(a, b):
    """
    Laskee Octile etäisyyden kahden pisteen välillä.
    
    Octile etäisyys on heuristiikka, jota käytetään kun voidaan liikkua 
    kahdeksaan suuntaan (4 ortogonaalista + 4 diagonaalista).
    
    Args:
        a (tuple): Ensimmäinen piste (x, y)
        b (tuple): Toinen piste (x, y)
    
    Returns:
        float: Octile etäisyys pisteiden välillä
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

def get_neighbors(node, grid):
    """
    Palauttaa kaikki kelvolliset naapurisolmut annetulle solmulle.
    
    Tarkistaa kaikki 8 suuntaa (4 ortogonaalista + 4 diagonaalista) ja
    palauttaa ne solmut, jotka ovat ruudukon sisällä ja vapaita (arvo 0).
    
    Args:
        node (tuple): Solmun koordinaatit (x, y)
        grid (list): 2D lista joka esittää ruudukkoa (0 = vapaa, 1 = este)
    
    Returns:
        list: Lista kelvollisista naapurisolmuista
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:
            neighbors.append((x, y))
    return neighbors

class AStar:
    """
    A* algoritmin toteutus ruudukkopohjaiseen polunhakuun.
    
    A* on optimaalinen polunhakualgoritmi, joka käyttää heuristiikkaa
    löytääkseen lyhimmän reitin kahden pisteen välillä tehokkaasti.
    
    Attributes:
        grid (list): 2D ruudukko jossa 0 = vapaa, 1 = este
        heuristic (function): Heuristiikkafunktio etäisyyden arvioimiseen
        rows (int): Ruudukon rivien määrä
        cols (int): Ruudukon sarakkeiden määrä
    """
    def __init__(self, grid, heuristic=octile_distance):
        """
        Alustaa A* algoritmin.
        
        Args:
            grid (list): 2D lista joka esittää ruudukkoa (0 = vapaa, 1 = este)
            heuristic (function, optional): Heuristiikkafunktio. Oletuksena octile_distance.
        """
        self.grid = grid
        self.heuristic = heuristic
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start, goal):
        """
        Etsii lyhimmän reitin aloitussolmusta maalisolmuun A* algoritmia käyttäen.
        
        Algoritmi käyttää f-arvoa (f = g + h), jossa g on todellinen etäisyys
        aloitussolmusta ja h on heuristinen arvio etäisyydestä maalisolmuun.
        
        Args:
            start (tuple): Aloitussolmu (x, y)
            goal (tuple): Maalisolmu (x, y)
        
        Returns:
            tuple: Sisältää kolme elementtiä:
                - path (list or None): Lista solmuista jotka muodostavat reitin,
                  tai None jos reittiä ei löydy
                - closed_set (set): Joukko tutkituista solmuista
                - nodes_added (int): Avoimeen joukkoon lisättyjen solmujen määrä
        """
        g_scores = {start: 0}

        # lasketaan heuristinen etäisyys aloitussolmusta maalisolmuun

        f_scores = {start: self.heuristic(start, goal)}

        # edellinen solmu, josta on tultu nykyiseen solmuun, tätä käytetään reitin jäljittämiseen

        came_from = {}
        # luodaan joukko suljetuista solmuista

        closed_set = set()
        # luodaan prioriteettijono, johon lisätään aloitussolmu

        open_set = [(f_scores[start], start)]
        nodes_added = 1  # Aloitussolmu lisätään heti


        # kunnes löydetään maalisolmu tai jono on tyhjentynyt

        while open_set:
            # käsitellään solmu, jolla on pienin f-arvo

            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], closed_set, nodes_added
            
            # lisätään nykyinen solmu suljettuun joukkoon

            closed_set.add(current)

            # käydään läpi kaikki naapurisolmut

            for neighbor in get_neighbors(current, self.grid):
                if neighbor in closed_set:
                    continue
                # Lasketaan etäisyys (1 tai √2)
                tentative_g = g_scores[current] + (
                    math.sqrt(2) if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 2 else 1
                )
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_scores[neighbor], neighbor))
                    nodes_added += 1

        return None, closed_set, nodes_added