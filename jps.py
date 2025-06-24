import heapq
import math

# JPS (Jump Point Search) algoritmi


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


def is_valid(pos, grid):
    """
    Tarkistaa onko annettu positio kelvollinen ruudukossa.
    
    Args:
        pos (tuple): Tarkistettava positio (x, y)
        grid (list): 2D lista joka esittää ruudukkoa (0 = vapaa, 1 = este)
    
    Returns:
        bool: True jos positio on kelvollinen ja vapaa, False muuten
    """
    x, y = pos
    return (0 <= x < len(grid) and 
            0 <= y < len(grid[0]) and 
            grid[x][y] == 0)

def get_direction(from_pos, to_pos):
    """
    Palauttaa normalisoidun suuntavektorin kahden pisteen välillä.
    
    Args:
        from_pos (tuple): Lähtöpiste (x, y)
        to_pos (tuple): Kohdepiste (x, y)
    
    Returns:
        tuple: Normalisoitu suuntavektori (dx, dy) arvoilla -1, 0 tai 1
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    # Normalisoi suunta (-1, 0, 1)
    if dx != 0:
        dx = 1 if dx > 0 else -1
    if dy != 0:
        dy = 1 if dy > 0 else -1
    
    return (dx, dy)

def get_neighbors(pos, grid, parent=None):
    """
    Palauttaa naapurisolmut annetulle positiolle JPS-algoritmin mukaisesti.
    
    Jos vanhempi on None (aloitussolmu), palauttaa kaikki kelvolliset naapurit.
    Muuten palauttaa vain karsitut naapurit liikkumissuunnan perusteella.
    
    Args:
        pos (tuple): Nykyinen positio (x, y)
        grid (list): 2D ruudukko
        parent (tuple, optional): Vanhemman positio (x, y)
    
    Returns:
        list: Lista naapurisolmuista
    """
    x, y = pos
    neighbors = []
    
    if parent is None:
        # Aloitussolmu - palauta kaikki kelvolliset naapurit
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            new_pos = (x + dx, y + dy)
            if is_valid(new_pos, grid):
                neighbors.append(new_pos)
    else:
        # Karsitut naapurit liikkumissuunnan perusteella
        direction = get_direction(parent, pos)
        dx, dy = direction
        
        # Luonnolliset naapurit (jatketaan samaan suuntaan)
        if dx != 0 and dy != 0:
            # Diagonaalinen liike
            # Jatka diagonaalisesti
            new_pos = (x + dx, y + dy)
            if is_valid(new_pos, grid):
                neighbors.append(new_pos)
            
            # Jatka horisontaalisesti ja vertikaalisesti
            if is_valid((x + dx, y), grid):
                neighbors.append((x + dx, y))
            if is_valid((x, y + dy), grid):
                neighbors.append((x, y + dy))
        else:
            # Ortogonaalinen liike
            if dx != 0:
                # Horisontaalinen liike
                if is_valid((x + dx, y), grid):
                    neighbors.append((x + dx, y))
                if is_valid((x, y + 1), grid):
                    neighbors.append((x, y + 1))
                if is_valid((x, y - 1), grid):
                    neighbors.append((x, y - 1))
            else:
                # Vertikaalinen liike
                if is_valid((x, y + dy), grid):
                    neighbors.append((x, y + dy))
                if is_valid((x + 1, y), grid):
                    neighbors.append((x + 1, y))
                if is_valid((x - 1, y), grid):
                    neighbors.append((x - 1, y))
        
        # Lisää pakotetut naapurit
        forced = get_forced_neighbors(pos, direction, grid)
        neighbors.extend(forced)
    
    return neighbors

def get_forced_neighbors(pos, direction, grid):
    """
    Palauttaa pakotetut naapurit annetulle positiolle ja suunnalle.
    
    Pakotetut naapurit ovat solmuja, jotka on tutkittava koska este 
    pakottaa reitin kulkemaan kyseisen solmun kautta.
    
    Args:
        pos (tuple): Nykyinen positio (x, y)
        direction (tuple): Liikkumissuunta (dx, dy)
        grid (list): 2D ruudukko
    
    Returns:
        list: Lista pakotetuista naapureista
    """
    x, y = pos
    dx, dy = direction
    forced = []
    
    if dx != 0 and dy != 0:
        # Diagonaalinen liike
        # Tarkista vasen/oikea puoli
        if not is_valid((x - dx, y), grid) and is_valid((x - dx, y + dy), grid):
            forced.append((x - dx, y + dy))
        if not is_valid((x, y - dy), grid) and is_valid((x + dx, y - dy), grid):
            forced.append((x + dx, y - dy))
    
    elif dx != 0:
        # Horisontaalinen liike
        if not is_valid((x, y + 1), grid) and is_valid((x + dx, y + 1), grid):
            forced.append((x + dx, y + 1))
        if not is_valid((x, y - 1), grid) and is_valid((x + dx, y - 1), grid):
            forced.append((x + dx, y - 1))
    
    elif dy != 0:
        # Vertikaalinen liike
        if not is_valid((x + 1, y), grid) and is_valid((x + 1, y + dy), grid):
            forced.append((x + 1, y + dy))
        if not is_valid((x - 1, y), grid) and is_valid((x - 1, y + dy), grid):
            forced.append((x - 1, y + dy))
    
    return forced

def has_forced_neighbors(pos, direction, grid):
    """
    Tarkistaa onko annetulla positiolla pakotettuja naapureita tietyssä suunnassa.
    
    Args:
        pos (tuple): Tarkistettava positio (x, y)
        direction (tuple): Liikkumissuunta (dx, dy)
        grid (list): 2D ruudukko
    
    Returns:
        bool: True jos pakotettuja naapureita löytyy, False muuten
    """
    return len(get_forced_neighbors(pos, direction, grid)) > 0

def jump(start_pos, direction, goal, grid):
    """
    Suorittaa hyppäämisen annettuun suuntaan kunnes löytyy hyppypiste, maali tai este.
    
    Hyppypiste on solmu jossa on:
    - Pakotettuja naapureita
    - Maalisolmu
    - Diagonaalisessa liikkeessa: ortogonaalinen hyppypiste
    
    Args:
        start_pos (tuple): Aloituspositio (x, y)
        direction (tuple): Hyppäämissuunta (dx, dy)
        goal (tuple): Maalisolmu (x, y)
        grid (list): 2D ruudukko
    
    Returns:
        tuple or None: Hyppypisteen koordinaatit tai None jos hyppypistettä ei löydy
    """
    current = start_pos
    dx, dy = direction
    
    while True:
        # Siirry seuraavaan positioon
        current = (current[0] + dx, current[1] + dy)
        
        # Tarkista onko positio kelvollinen
        if not is_valid(current, grid):
            return None
        
        # Tarkista onko maali
        if current == goal:
            return current
        
        # Tarkista onko pakotettuja naapureita
        if has_forced_neighbors(current, direction, grid):
            return current
        
        # Diagonaalinen liike: tarkista ortogonaalisia suuntia
        if dx != 0 and dy != 0:
            # Tarkista horisontaalinen suunta
            if jump(current, (dx, 0), goal, grid) is not None:
                return current
            
            # Tarkista vertikaalinen suunta
            if jump(current, (0, dy), goal, grid) is not None:
                return current
            
def identify_successors(pos, goal, grid, parent=None):
    """
    Tunnistaa ja palauttaa kaikki hyppypiste-seuraajat annetulle positiolle.
    
    Käy läpi kaikki naapurit ja suorittaa hyppäämisen jokaiseen suuntaan
    löytääkseen hyppypisteet.
    
    Args:
        pos (tuple): Nykyinen positio (x, y)
        goal (tuple): Maalisolmu (x, y)
        grid (list): 2D ruudukko
        parent (tuple, optional): Vanhemman positio polunhakua varten
    
    Returns:
        list: Lista hyppypiste-seuraajista
    """
    successors = []
    neighbors = get_neighbors(pos, grid, parent)
    
    for neighbor in neighbors:
        direction = get_direction(pos, neighbor)
        jump_point = jump(pos, direction, goal, grid)
        
        if jump_point is not None:
            successors.append(jump_point)
    
    return successors

class JPS:
    """
    Jump Point Search (JPS) algoritmin toteutus.
    
    JPS on optimoitu versio A* algoritmista ruudukkopohjaiseen polunhakuun.
    Se vähentää tutkittavien solmujen määrää tunnistamalla "hyppypisteet"
    ja hyppäämällä suoraan niihin välitutkimatta kaikkia solmuja.
    
    Attributes:
        grid (list): 2D ruudukko jossa 0 = vapaa, 1 = este
        heuristic (function): Heuristiikkafunktio etäisyyden arvioimiseen
    """
 
    def __init__(self, grid, heuristic=octile_distance):
        """
        Alustaa JPS-algoritmin.
        
        Args:
            grid (list): 2D lista joka esittää ruudukkoa (0 = vapaa, 1 = este)
            heuristic (function, optional): Heuristiikkafunktio. Oletuksena octile_distance.
        """
        self.grid = grid
        self.heuristic = heuristic

    def find_path(self, start, goal):
        """
        Etsii lyhimmän reitin aloitussolmusta maalisolmuun JPS-algoritmia käyttäen.
        
        Args:
            start (tuple): Aloitussolmu (x, y)
            goal (tuple): Maalisolmu (x, y)
        
        Returns:
            tuple: Sisältää kolme elementtiä:
                - path (list or None): Lista solmuista jotka muodostavat reitin, 
                  tai None jos reittiä ei löydy
                - closed_set (set): Joukko tutkituista solmuista
                - jump_points_explored (int): Tutkittujen hyppypisteiden määrä
        """
        if not is_valid(start, self.grid) or not is_valid(goal, self.grid):
            return None, set(), 0
        
        if start == goal:
            return [start], set(), 0
        
        # Alustetaan tietorakenteet
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        jump_points_explored = 0
        
        # Pitää kirjaa siitä, mitkä solmut ovat open_set:ssä
        in_open_set = {start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            in_open_set.discard(current)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if current == goal:
                # Rakenna polku
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], closed_set, jump_points_explored
            
            # Hae seuraajat
            parent = came_from.get(current)
            successors = identify_successors(current, goal, self.grid, parent)
            jump_points_explored += len(successors)
            
            for successor in successors:
                if successor in closed_set:
                    continue
                
                tentative_g = g_score[current] + octile_distance(current, successor)
                
                if successor not in g_score or tentative_g < g_score[successor]:
                    came_from[successor] = current
                    g_score[successor] = tentative_g
                    f_score[successor] = tentative_g + self.heuristic(successor, goal)
                    
                    if successor not in in_open_set:
                        heapq.heappush(open_set, (f_score[successor], successor))
                        in_open_set.add(successor)
        
        return None, closed_set, jump_points_explored # Palautetaan None, jos reittiä ei löydy, ja suljettu joukko sekä hyppypisteiden määrä