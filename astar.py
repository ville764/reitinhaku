
import heapq
import math

# lasketan Octile etäisyys kahden solmun välillä. Octile etäisyys on käytössä, kun voidaan liikkua kahdeksaan suuntaan (ylös, alas, vasemmalle, oikealle ja diagonaalisesti).
def octile_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# luodaan tieto naapurisolmuista jokaiselle solmulle
def get_neighbors(node, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0:
            neighbors.append((x, y))
    return neighbors

# A* algoritmin luokka ruudukolle
class AStar:
    def __init__(self, grid, heuristic=octile_distance):
        self.grid = grid
        self.heuristic = heuristic
        self.rows = len(grid)
        self.cols = len(grid[0])

    # etsi reitti aloitussolmusta maalisolmuun
    def find_path(self, start, goal):
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
