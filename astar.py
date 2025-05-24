import heapq
import math

# lasketan Octile etäisyys kahden solmun välillä. Octile etäisyys on käytössä, kun voidaan liikkua kahdeksaan suuntaan (ylös, alas, vasemmalle, oikealle ja diagonaalisesti).
def octile_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# luodaan tieto naapurisolmuista jokaiselle solmulle
def get_neighbors(node, rows, cols):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < rows and 0 <= y < cols:
            neighbors.append((x, y))
    return neighbors

class AStar:
    def __init__(self, nodes, heuristic):
        self.nodes = nodes
        self.graph = {node: [] for node in nodes}
        self.heuristic = heuristic
    
    #märitellä kaari solmusta a solmuun b
    def add_edge(self, node_a, node_b, weight):
        self.graph[node_a].append((node_b, weight))

    # etsi reitti aloitussolmusta maalisolmuun
    def find_path(self, start_node, goal_node):
        g_scores = {node: float("inf") for node in self.nodes}
        g_scores[start_node] = 0

        f_scores = {node: float("inf") for node in self.nodes}
        # lasketaan heuristinen etäisyys aloitussolmusta maalisolmuun
        f_scores[start_node] = self.heuristic(start_node, goal_node)

        # edellinen solmu, josta on tultu nykyiseen solmuun, tätä käytetään reitin jäljittämiseen
        came_from = {}
        # luodaan joukko suljetuista solmuista
        closed_set = set()

        # luodaan prioriteettijono, johon lisätään aloitussolmu
        queue = []
        heapq.heappush(queue, (f_scores[start_node], start_node))
  

        # kunnes löydetään maalisolmu tai jono on tyhjentynyt
        while queue:
            # käsitellään solmu, jolla on pienin f-arvo
            current = heapq.heappop(queue)[1]

            # lisätään nykyinen solmu suljettuun joukkoon
            closed_set.add(current)

            # jos nykyinen solmu on maalisolmu, reitti on löytynyt
            if current == goal_node:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                return path[::-1], closed_set
            
            # käydään läpi kaikki naapurisolmut
            for neighbor, weight in self.graph[current]:
                tentative_g_score = g_scores[current] + weight
                if tentative_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                    heapq.heappush(queue, (f_scores[neighbor], neighbor))

        return None, closed_set
