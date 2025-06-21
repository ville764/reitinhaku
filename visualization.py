import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from astar import AStar, octile_distance
from jps import JPS, octile_distance

# tässä funktiossa visualisoidaan A* algoritmin toimintaa
# ja reitin löytämistä ruudukossa
# käytetään matplotlib-kirjastoa visualisointiin
def visualize_astar(grid, start, goal, ax=None, title="A* Algoritmi"):
    # luodaan A* algoritmi uudella toteutuksella
    astar = AStar(grid, heuristic=octile_distance)

    # etsitään reitti aloitussolmusta maalisolmuun
    # ja saadaan suljettu joukko solmuista, jotka on käsitelty
    # ja reitti, joka on löydetty
    path, closed_set, nodes_added = astar.find_path(start, goal)

    # Jos ax ei ole annettu, luodaan uusi kuvaaja
    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False
    
    # piirretään aloitussolmu vihreänä ja maalisolmu punaisena
    # ja asetetaan niiden koordinaatit
    def update(frame):
        ax.clear()
        ax.set_title(title)
        # asetetaan ruudukon koko ja värit
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.imshow(grid, cmap='gray_r')
        
        # piirretään kaikki solmut, jotka on käsitelty
        # ja asetetaan niiden väri siniseksi
        for node in list(closed_set)[:frame]:
            ax.plot(node[1], node[0], 'bo')
            
        # piirretään reitti punaisena
        if path:
            ax.plot([node[1] for node in path], [node[0] for node in path], 'r-', linewidth=2)
            
        # piirretään alku- ja loppupisteet
        ax.plot(start[1], start[0], 'go')  # vihreä: alku
        ax.plot(goal[1], goal[0], 'rx')    # punainen: maali

    ani = animation.FuncAnimation(ax.figure if not standalone else fig, update, frames=len(closed_set), repeat=False)
    
    if standalone:
        plt.show()
    
    return ani

# Visualisoi Jump Point Search (JPS) algoritmin toimintaa
def visualize_jps(grid, start, goal, ax=None, title="JPS Algoritmi"):
    rows, cols = grid.shape
    jps = JPS(grid, octile_distance)
    path, closed_set, jump_points_added = jps.find_path(start, goal)

    # Jos ax ei ole annettu, luodaan uusi kuvaaja
    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False
    
    def update(frame):
        ax.clear()
        ax.set_title(title)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.imshow(grid, cmap='gray_r')

        # Piirrä käsitellyt jump pointit
        for node in list(closed_set)[:frame]:
            ax.plot(node[1], node[0], 'bo')  # sininen piste

        # Piirrä reitti jokaisessa ruudussa jos se on löydetty
        if path:
            ax.plot([node[1] for node in path], [node[0] for node in path], 'r-', linewidth=2)

        # Piirrä alku- ja loppupisteet
        ax.plot(start[1], start[0], 'go')  # vihreä: alku
        ax.plot(goal[1], goal[0], 'rx')    # punainen: maali

    ani = animation.FuncAnimation(ax.figure if not standalone else fig, update, frames=len(closed_set), repeat=False)
    
    if standalone:
        plt.show()
        
    return ani

# Visualisoi molemmat algoritmit allekkain vertailua varten
def visualize_comparison(grid, start, goal, scenario_name=""):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Aseta pääotsikko
    if scenario_name:
        fig.suptitle(f'Skenaario {scenario_name}: Algoritmien vertailu\nAlku: {start}, Maali: {goal}', 
                    fontsize=14, fontweight='bold')
    
    # Visualisoi A* yläpuolelle
    ani1 = visualize_astar(grid, start, goal, ax1, "A* Algoritmi")
    
    # Visualisoi JPS alapuolelle  
    ani2 = visualize_jps(grid, start, goal, ax2, "JPS Algoritmi")
    
    # Pienennä väliä subplottien välillä
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    
    return ani1, ani2

# Funktio skenaarion valintaan ja visualisointiin
def visualize_selected_scenarios(summary, np_map):
    print("\nKäytettävissä olevat skenaariot:")
    for i, row in enumerate(summary):
        print(f"{i+1}. Skenaario {row['Skenaario']}: Alku {row['Alku']}, Loppu {row['Loppu']}, "
              f"Optimaalinen pituus {row['Optimaalinen']:.2f}")
    
    while True:
        try:
            print("\nValitse visualisoitava skenaario:")
            print("- Anna numeron (1-{}) visualisoidaksesi yhden skenaarion".format(len(summary)))
            print("- Anna 'kaikki' visualisoidaksesi kaikki skenaariot")
            print("- Anna 'lopeta' lopettaaksesi")
            
            choice = input("Valintasi: ").strip().lower()
            
            if choice == 'lopeta':
                break
            elif choice == 'kaikki':
                for i, row in enumerate(summary):
                    start = row['Alku']
                    goal = row['Loppu']
                    optimal_length = row['Optimaalinen']
                    scenario_name = row['Skenaario']
                    
                    print(f"\nVisualisoidaan skenaario {scenario_name}")
                    visualize_comparison(np_map, start, goal, scenario_name)
                break
            else:
                index = int(choice) - 1
                if 0 <= index < len(summary):
                    row = summary[index]
                    start = row['Alku']
                    goal = row['Loppu']
                    optimal_length = row['Optimaalinen']
                    scenario_name = row['Skenaario']
                    
                    print(f"\nVisualisoidaan skenaario {scenario_name}: "
                          f"Alkupiste {start}, Loppupiste {goal}, "
                          f"Optimaalinen pituus {optimal_length:.2f}")
                    
                    visualize_comparison(np_map, start, goal, scenario_name)
                else:
                    print("Virheellinen valinta. Valitse numero välillä 1-{}.".format(len(summary)))
        except ValueError:
            print("Virheellinen syöte. Anna numero tai 'kaikki' tai 'lopeta'.")
        except KeyboardInterrupt:
            print("\nVisualisointi keskeytetty.")
            break