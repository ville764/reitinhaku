import heapq
import math

# JPS (Jump Point Search) algoritmi 

# lasketan Octile etäisyys kahden solmun välillä. Octile etäisyys on käytössä, 
# kun voidaan liikkua kahdeksaan suuntaan (ylös, alas, vasemmalle, oikealle ja diagonaalisesti).
def octile_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# luodaan tieto naapurisolmuista jokaiselle solmulle
def get_neighbors(node, grid):
    neighbors = []
    rows, cols = grid.shape
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < rows and 0 <= y < cols:
            neighbors.append((x, y))
    return neighbors

# askel solmusta n suuntaan direction
def step(x, direction):
    return (x[0] + direction[0], x[1] + direction[1])

# tarkistetaan onko solmu vapaa (0) vai este (1)
def is_valid(n, grid):
    rows, cols = len(grid), len(grid[0])
    x, y = n
    return 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0  # 0 = vapaa, 1 = este

# tarkistetaan onko naapuri pakotettu
def is_forced(neigh, n, grid):
    dx = neigh[0] - n[0]
    dy = neigh[1] - n[1]

    # Tarkastellaan vain naapureita, jotka ovat yhden askeleen päässä
    if abs(dx) > 1 or abs(dy) > 1:
        return False

    # Ortogonaalinen liike
    if dx == 0 or dy == 0:
        # Esimerkiksi: liikutaan oikealle, tarkistetaan ylä- ja alapuoli
        if dx == 1:  # alas
            return (is_valid((n[0], n[1] - 1), grid) and not is_valid((n[0] + 1, n[1] - 1), grid)) or \
                   (is_valid((n[0], n[1] + 1), grid) and not is_valid((n[0] + 1, n[1] + 1), grid))
        if dx == -1:  # ylös
            return (is_valid((n[0], n[1] - 1), grid) and not is_valid((n[0] - 1, n[1] - 1), grid)) or \
                   (is_valid((n[0], n[1] + 1), grid) and not is_valid((n[0] - 1, n[1] + 1), grid))
        if dy == 1:  # oikealle
            return (is_valid((n[0] - 1, n[1]), grid) and not is_valid((n[0] - 1, n[1] + 1), grid)) or \
                   (is_valid((n[0] + 1, n[1]), grid) and not is_valid((n[0] + 1, n[1] + 1), grid))
        if dy == -1:  # vasemmalle
            return (is_valid((n[0] - 1, n[1]), grid) and not is_valid((n[0] - 1, n[1] - 1), grid)) or \
                   (is_valid((n[0] + 1, n[1]), grid) and not is_valid((n[0] + 1, n[1] - 1), grid))

    # Diagonaalinen liike
    if abs(dx) == 1 and abs(dy) == 1:
        return (is_valid((n[0] - dx, n[1]), grid) and not is_valid((n[0] - dx, n[1] + dy), grid)) or \
               (is_valid((n[0], n[1] - dy), grid) and not is_valid((n[0] + dx, n[1] - dy), grid))

    return False


# tarkistetaan onko suunta diagonaalinen
def is_diagonal(direction):
    return abs(direction[0]) == 1 and abs(direction[1]) == 1

# haetaan diagonaalisen suunnan komponentit
def get_diagonal_components(direction):
    dx, dy = direction
    return [(dx, 0), (0, dy)]

# karsitaan naapurit, jotka eivät ole linjassa vanhemman kanssa
def prune(x, neighbors, parent=None):
    if parent is None:
        return neighbors  # Jos ei ole vanhempaa, palautetaan kaikki naapurit
    # Lasketaan suunta vanhemmasta nykyiseen solmuun
    px, py = parent
    # Lasketaan suunta nykyisestä solmusta vanhempaan
    dx = (x[0] - px) // max(abs(x[0] - px), 1)
    dy = (x[1] - py) // max(abs(x[1] - py), 1)

    # Karsitaan naapurit, jotka eivät ole linjassa nykyisen solmun kanssa
    pruned = []
    for n in neighbors:
        ndx = n[0] - x[0]
        ndy = n[1] - x[1]
        if (dx == 0 or ndx == dx) and (dy == 0 or ndy == dy):
            pruned.append(n)
    return pruned

# määritellään suunta nykyisestä solmusta naapuriin
def direction(x, n):
    dx = n[0] - x[0]
    dy = n[1] - x[1]
    return (dx // max(abs(dx), 1), dy // max(abs(dy), 1))

# hyppääminen solmusta n suuntaan direction, kunnes saavutetaan maali tai este
def jump(x, direction, start, goal, grid):
    n = step(x, direction)

    if not is_valid(n, grid):
        return None

    if n == goal:
        return n
    # Tarkistetaan, onko naapuri pakotettu
    if any(is_forced(neigh, n, grid) for neigh in get_neighbors(n, grid)):
        return n

    # Jos suunta on diagonaalinen, tarkistetaan kaikki diagonaaliset komponentit
    if is_diagonal(direction):
        directions = get_diagonal_components(direction)
        for d in directions:
            if jump(n, d, start, goal, grid) is not None:
                return n
    # Jos suunta on ortogonaalinen, tarkistetaan vain se suunta

    return jump(n, direction, start, goal, grid)


# Tunnistetaan seuraajat solmulle x, jotka ovat hyppypisteitä
# suhteessa aloitus- ja maalisolmuun
def identify_successors(x, start, goal, grid):
    successors = set()
    neighbors = prune(x, get_neighbors(x, grid))

    for n in neighbors:
        direction_to_n = direction(x, n)
        jump_point = jump(x, direction_to_n, start, goal, grid)
        if jump_point is not None:
            successors.add(jump_point)

    return successors

# JPS (Jump Point Search) algoritmin luokka
class JPS:
    # Tämän luokan avulla voidaan etsiä reitti aloitussolmusta maalisolmuun
    # käyttäen JPS-algoritmia, joka on optimoitu reittien etsimiseen ruudukossa.
    # Tämän luokan avulla voidaan määritellä ruudukko ja heuristinen funktio, jota käytetään reitin etsimiseen.
    def __init__(self, grid, heuristic):
        self.grid = grid
        self.heuristic = heuristic

    # JPS-algoritmin find_path-metodi etsii reitin aloitussolmusta maalisolmuun
    # Tämä metodi käyttää JPS-algoritmia reitin etsimiseen ja palauttaa löydetyn reitin sekä suljetun joukon.
    # start: aloitussolmu (rivi, sarake)
    # goal: maalisolmu (rivi, sarake)
    # Palauttaa: reitin (lista solmuista) ja suljetun joukon (set solmuista)
    def find_path(self, start, goal):
        jump_points_added = 0 # testataan hyppypisteiden määrää
        rows, cols = len(self.grid), len(self.grid[0])
        g_scores = {start: 0}
        f_scores = {start: self.heuristic(start, goal)}
        came_from = {}
        closed_set = set()
        queue = []
        heapq.heappush(queue, (f_scores[start], start))

        while queue:
            current = heapq.heappop(queue)[1]
            closed_set.add(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], closed_set, jump_points_added  # Palautetaan reitti käänteisessä järjestyksessä, suljettu joukko ja hyppypisteiden määrä
            
            for neighbor in identify_successors(current, start, goal, self.grid):
                tentative_g_score = g_scores[current] + octile_distance(current, neighbor)
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(queue, (f_scores[neighbor], neighbor))
                    jump_points_added += 1


        return None, closed_set # Palautetaan None, jos reittiä ei löydy, ja suljettu joukko
