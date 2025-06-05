import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from astar import AStar, octile_distance, get_neighbors
from jps import JPS, octile_distance   



# tässä funktiossa visualisoidaan A* algoritmin toimintaa
# ja reitin löytämistä ruudukossa
# käytetään matplotlib-kirjastoa visualisointiin
def visualize_astar(grid, start, goal):
    rows, cols = grid.shape
    # luodaan lista solmuista, jotka ovat esteettömiä (arvo 0)
    # ja lisätään ne A* algoritmin solmuiksi
    nodes = [(i, j) for i in range(rows) for j in range(cols) if grid[i, j] == 0]
    # kutsutaan A* algoritmia ja määritellään heuristinen funktio
    astar = AStar(nodes, heuristic=octile_distance)

    # lisätään kaaret solmujen välille
    # ja määritellään painot (1 tai 1.4) riippuen siitä, onko liikkuminen diagonaalisesti vai ei
    for node in nodes:
        for neighbor in get_neighbors(node, rows, cols):
            if grid[neighbor] == 0:
                weight = 1.4 if abs(node[0] - neighbor[0]) == 1 and abs(node[1] - neighbor[1]) == 1 else 1
                astar.add_edge(node, neighbor, weight)
    # etsitään reitti aloitussolmusta maalisolmuun
    # ja saadaan suljettu joukko solmuista, jotka on käsitelty
    # ja reitti, joka on löydetty

    path, closed_set = astar.find_path(start, goal)

    # luodaan kuvaaja, johon piirretään ruudukko
    # ja asetetaan solmujen väliin ruudukko
    fig, ax = plt.subplots()
    # asetetaan ruudukon koko ja värit
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.imshow(grid, cmap='gray_r')

    # piirretään aloitussolmu vihreänä ja maalisolmu punaisena
    # ja asetetaan niiden koordinaatit
    def update(frame):
        ax.clear()
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.imshow(grid, cmap='gray_r')
        # piirretään kaikki solmut, jotka on käsitelty
        # ja asetetaan niiden väri siniseksi
        # ja piirretään reitti punaisena
        # ja asetetaan niiden koordinaatit
        for node in list(closed_set)[:frame]:

            ax.plot(node[1], node[0], 'bo')
        if path:
            ax.plot([node[1] for node in path], [node[0] for node in path], 'r-')
        ax.plot(start[1], start[0], 'go')
        ax.plot(goal[1], goal[0], 'rx')


    ani = animation.FuncAnimation(fig, update, frames=len(closed_set), repeat=False)
    plt.show()

# Visualisoi Jump Point Search (JPS) algoritmin toimintaa
def visualize_jps(grid, start, goal):
    rows, cols = grid.shape
    jps = JPS(grid, heuristic=octile_distance)
    path, closed_set = jps.find_path(start, goal)

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.imshow(grid, cmap='gray_r')

    def update(frame):
        ax.clear()
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.imshow(grid, cmap='gray_r')

        # Piirrä käsitellyt jump pointit
        for node in list(closed_set)[:frame]:
            ax.plot(node[1], node[0], 'bo')  # sininen piste

        # Viimeisessä ruudussa piirrä myös reitti
        if frame == len(closed_set) and path:
            ax.plot([node[1] for node in path], [node[0] for node in path], 'r-', linewidth=2)

        # Piirrä alku- ja loppupisteet
        ax.plot(start[1], start[0], 'go')  # vihreä: alku
        ax.plot(goal[1], goal[0], 'rx')    # punainen: maali

    ani = animation.FuncAnimation(fig, update, frames=len(closed_set) + 1, repeat=False)
    plt.show()
