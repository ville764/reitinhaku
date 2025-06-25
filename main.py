"""
Reitinhakualgoritmien vertailu - JPS vs A*

Tämä ohjelma vertailee Jump Point Search (JPS) ja A* algoritmien suorituskykyä
käyttäen karttadataa ja skenaarioita. Ohjelma laskee polkujen pituudet, 
suoritusajat ja vertaa niitä optimaalisiin polkuihin.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import map_loader as ml
from astar import AStar, octile_distance, get_neighbors as astar_get_neighbors 
from visualization import visualize_selected_scenarios
from jps import JPS, octile_distance
import numpy as np
import os


def load_map_and_scenarios():
    """
    Lataa karttadatan ja skenaariot tiedostoista.
    
    Returns:
        tuple: (karttadata numpy-taulukkona, skenaariolista)
    """
    # Selvitetään nykyisen tiedoston sijainti
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Määritetään polku karttatiedostoihin suhteessa projektin juureen
    map_path = os.path.join(base_dir, 'maps', 'rmtst03.map.txt')
    scen_path = os.path.join(base_dir, 'maps', 'rmtst03.map.scen.txt')

    # Ladataan kartta ja skenaariot
    map_data = ml.load_map(map_path)
    np_map = ml.map_to_numpy(map_data)
    scenarios = ml.load_scenarios(scen_path)
    print(f"Ladattu kartta {map_path} ja skenaariot {scen_path}.")
    print(f"Kartta koko: {np_map.shape[0]} riviä, {np_map.shape[1]} saraketta.")
    print(f"Ladattu {len(scenarios)} skenaariota.")
    
    return np_map, scenarios


def test_jps_performance(scenarios, np_map, num_scenarios=10):
    """
    Testaa JPS-algoritmin suorituskykyä annetuilla skenaarioilla.
    
    Args:
        scenarios (list): Lista skenaarioita (alku, loppu, optimaalinen_pituus)
        np_map (numpy.ndarray): Karttadata numpy-taulukkona
        num_scenarios (int): Testattavien skenaarioiden määrä
    """
    print("Verrataan JPS:n suorituskykyä optimaalisesti laskettuun polkuun, 10 ensimmäistä polkua:\n")
    
    # Käydään läpi ensimmäiset 10 skenaariota ja lasketaan JPS-polun pituus
    # sekä verrataan sitä optimaalisesti laskettuun polun pituuteen
    # Tulostetaan tulokset
    for i, (start, goal, optimal_length) in enumerate(scenarios[:num_scenarios]):
        jps = JPS(np_map, heuristic=octile_distance)

        # Aloitetaan ajan mittaus
        start_time = time.time()
        path,closed_set,jump_points_added = jps.find_path(start, goal)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if path:
            path_length = sum(np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path)-1))
            print(f"Skenaario {i+1}:")
            print(f"  Alkupiste: {start}, Loppupiste: {goal}")
            print(f"  Optimaalinen pituus: {optimal_length:.2f}")
            print(f"  JPS algoritmin laskema polun pituus: {path_length:.2f}")
            print(f"  Erotus: {abs(path_length - optimal_length):.2f}")
            print(f"  Suoritusaika: {elapsed_time:.6f} sekuntia")
            print(f"  Hyppypisteiden määrä: {jump_points_added}\n")
        else:
            print(f"Skenaario {i+1}: Ei reittiä löydetty {start} -> {goal}")
            print(f"  Suoritusaika: {elapsed_time:.6f} sekuntia\n")


def test_astar_performance(scenarios, np_map, num_scenarios=10):
    """
    Testaa A*-algoritmin suorituskykyä annetuilla skenaarioilla.
    
    Args:
        scenarios (list): Lista skenaarioita (alku, loppu, optimaalinen_pituus)
        np_map (numpy.ndarray): Karttadata numpy-taulukkona
        num_scenarios (int): Testattavien skenaarioiden määrä
    """
    print("Verrataan A*-algoritmin suorituskykyä optimaalisesti laskettuun polkuun, 10 ensimmäistä polkua:\n")

    for i, (start, goal, optimal_length) in enumerate(scenarios[:num_scenarios]):
        astar = AStar(np_map, heuristic=octile_distance)

        start_time = time.time()
        path, closed_set, nodes_added = astar.find_path(start, goal)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if path:
            path_length = sum(np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path)-1))
            print(f"Skenaario {i+1}:")
            print(f"  Alkupiste: {start}, Loppupiste: {goal}")
            print(f"  Optimaalinen pituus: {optimal_length:.2f}")
            print(f"  A* algoritmin laskema polun pituus: {path_length:.2f}")
            print(f"  Erotus: {abs(path_length - optimal_length):.2f}")
            print(f"  Suoritusaika: {elapsed_time:.6f} sekuntia")
            print(f"  Solmuja lisätty avoimeen joukkoon: {nodes_added}\n")
        else:
            print(f"Skenaario {i+1}: Ei reittiä löydetty {start} -> {goal}")
            print(f"  Suoritusaika: {elapsed_time:.6f} sekuntia\n")


def run_comprehensive_comparison(scenarios, np_map):
    """
    Suorittaa kattavan vertailun JPS- ja A*-algoritmien välillä kaikilla skenaarioilla.
    
    Args:
        scenarios (list): Lista skenaarioita (alku, loppu, optimaalinen_pituus)
        np_map (numpy.ndarray): Karttadata numpy-taulukkona
        
    Returns:
        list: Lista yhteenvetodictejä jokaisesta suoritetusta skenaarista
    """
    summary = []

    for i, (start, goal, optimal_length) in enumerate(scenarios[:]):
        # JPS
        jps = JPS(np_map, heuristic=octile_distance)
        start_time = time.time()
        jps_path, _, jump_points = jps.find_path(start, goal)
        jps_time = time.time() - start_time    
        if jps_path:
            jps_length = sum(np.hypot(jps_path[i+1][0] - jps_path[i][0], jps_path[i+1][1] - jps_path[i][1]) for i in range(len(jps_path)-1))
            jps_error = abs(jps_length - optimal_length)
        else:
            jps_length = None
            jps_error = None
            jump_points = 0

        # A* (ruudukkopohjainen)
        astar = AStar(np_map, heuristic=octile_distance)
        start_time = time.time()
        astar_path, _, nodes_added = astar.find_path(start, goal)
        astar_time = time.time() - start_time
        if astar_path:
            astar_length = sum(np.hypot(astar_path[i+1][0] - astar_path[i][0], astar_path[i+1][1] - astar_path[i][1]) for i in range(len(astar_path)-1))
            astar_error = abs(astar_length - optimal_length)
        else:
            astar_length = None
            astar_error = None
            nodes_added = 0

        summary.append({
            "Skenaario": i + 1,
            "Alku": start,
            "Loppu": goal,
            "Optimaalinen": round(optimal_length, 2),
            "JPS pituus": round(jps_length, 2) if jps_length else None,
            "JPS virhe": round(jps_error, 2) if jps_error else None,
            "JPS aika": round(jps_time, 4),
            "JPS hyppypisteet": jump_points,
            "A* pituus": round(astar_length, 2) if astar_length else None,
            "A* virhe": round(astar_error, 2) if astar_error else None,
            "A* aika": round(astar_time, 4),
            "A* open set": nodes_added
        })
    
    return summary


def print_summary(summary):
    """
    Tulostaa yhteenvedon algoritmien suorituskyvystä taulukkomuodossa.
    
    Args:
        summary (list): Lista yhteenvetosanakirjoja suoritetuista skenaarioista
    """
    # Yhteenvedon tulostus
    print(f"{'Skenaario':<9}{'Alku':<15}{'Loppu':<15}{'Optimaalinen':<13}"
          f"{'JPS pituus':<12}{'JPS virhe':<12}{'JPS aika':<10}{'JPS hypyt':<12}"
          f"{'A* pituus':<12}{'A* virhe':<12}{'A* aika':<10}{'A* open set':<14}")

    for row in summary:
        print(f"{row['Skenaario']:<9}{str(row['Alku']):<15}{str(row['Loppu']):<15}{row['Optimaalinen']:<13}"
              f"{str(row['JPS pituus']):<12}{str(row['JPS virhe']):<12}{row['JPS aika']:<10}{row['JPS hyppypisteet']:<12}"
              f"{str(row['A* pituus']):<12}{str(row['A* virhe']):<12}{row['A* aika']:<10}{row['A* open set']:<14}")


def calculate_average_times(summary):
    """
    Laskee JPS:n ja A*:n keskimääräiset reitinhakuajat.
    
    Args:
        summary (list): Lista yhteenvetosanakirjoja suoritetuista skenaarioista
    """
    # Lasketaan JPS:n ja A*:n keskimääräiset reitinhakuajat
    jps_ajat = [row["JPS aika"] for row in summary if "JPS aika" in row]
    astar_ajat = [row["A* aika"] for row in summary if "A* aika" in row]

    jps_keskiarvo = sum(jps_ajat) / len(jps_ajat) if jps_ajat else 0
    astar_keskiarvo = sum(astar_ajat) / len(astar_ajat) if astar_ajat else 0

    print(f"JPS:n keskimääräinen reitinhakuaika: {jps_keskiarvo:.6f} sekuntia")
    print(f"A*:n keskimääräinen reitinhakuaika: {astar_keskiarvo:.6f} sekuntia")


def visualize_results(summary):
    """
    Visualisoi algoritmien suorituskykyä pylväsdiagrammien avulla.

    Args:
        summary (list): Lista yhteenvetosanakirjoja suoritetuista skenaarioista.
    """
    scenarios = [row['Skenaario'] for row in summary]
    jps_times = [row['JPS aika'] for row in summary]
    astar_times = [row['A* aika'] for row in summary]
    jps_jumps = [row['JPS hyppypisteet'] for row in summary]
    astar_open_set = [row['A* open set'] for row in summary]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Suoritusaikojen vertailu
    ax1.bar(scenarios, jps_times, width=0.4, label='JPS aika', align='center')
    ax1.bar(scenarios, astar_times, width=0.4, label='A* aika', align='edge')
    ax1.set_ylabel('Aika (s)')
    ax1.set_title('Suoritusaikojen vertailu')
    ax1.legend()

    # 2. Hyppypisteiden ja open setin vertailu
    ax2.bar(scenarios, jps_jumps, width=0.4, label='JPS hyppypisteet', align='center')
    ax2.bar(scenarios, astar_open_set, width=0.4, label='A* open set', align='edge')
    ax2.set_ylabel('Määrä')
    ax2.set_title('JPS hyppypisteet ja A* open set -solmujen määrä')
    ax2.legend()

    plt.xlabel('Skenaario')
    plt.tight_layout()
    plt.show()


def main():
    """
    Pääfunktio, joka suorittaa koko reitinhakualgoritmien vertailun.
    """
    # Ladataan kartta ja skenaariot
    np_map, scenarios = load_map_and_scenarios()
    
    # Testataan JPS:n suorituskykyä
    #test_jps_performance(scenarios, np_map)
    
    # Testataan A*:n suorituskykyä
    #test_astar_performance(scenarios, np_map)
    
    # Suoritetaan kattava vertailu
    summary = run_comprehensive_comparison(scenarios, np_map)
    
    # Tulostetaan yhteenveto
    print_summary(summary)
    
    # Lasketaan keskimääräiset ajat
    calculate_average_times(summary)
    
    # Visualisoidaan tulokset
    visualize_results(summary)
    
    # Visualisoidaan halutut skenaariot eli reitit JPS- ja A*-algoritmeilla
    visualize_selected_scenarios(summary, np_map)


if __name__ == "__main__":
    main()