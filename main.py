
import time
import matplotlib.pyplot as plt
import numpy as np
import map_loader as ml
from astar import AStar, octile_distance, get_neighbors as astar_get_neighbors 
from visualization import visualize_selected_scenarios
from jps import JPS, octile_distance
import numpy as np
import os



# Selvitetään nykyisen tiedoston sijainti
base_dir = os.path.dirname(os.path.abspath(__file__))

# Määritetään polku karttatiedostoihin suhteessa projektin juureen
map_path = os.path.join(base_dir, 'maps', 'rmtst03.map.txt')
scen_path = os.path.join(base_dir, 'maps', 'rmtst03.map.scen.txt')

# Ladataan kartta ja skenaariot
map_data = ml.load_map(map_path)
np_map = ml.map_to_numpy(map_data)
scenarios = ml.load_scenarios(scen_path)

# Ladataan kartta ja skenaariot
#map_path = '/Users/Ville/Documents/Opinnot/Algoritmit/reitinhaku/maps/rmtst03.map.txt'
#scen_path = '/Users/Ville/Documents/Opinnot/Algoritmit/reitinhaku/maps/rmtst03.map.scen.txt'
#map_data = ml.load_map(map_path)
#np_map = ml.map_to_numpy(map_data)
#scenarios = ml.load_scenarios(scen_path)

#verrataan JPS-algoritmin suorituskykyä optimaalisesti laskettuun polkuun
print("Verrataan JPS:n suorituskykyä optimaalisesti laskettuun polkuun, 10 ensimmäistä polkua:\n")
# Käydään läpi ensimmäiset 10 skenaariota ja lasketaan JPS-polun pituus
# sekä verrataan sitä optimaalisesti laskettuun polun pituuteen
# Tulostetaan tulokset


print("Verrataan JPS:n suorituskykyä optimaalisesti laskettuun polkuun, 10 ensimmäistä polkua:\n")

for i, (start, goal, optimal_length) in enumerate(scenarios[:10]):
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

# Verrataan A*-algoritmin suorituskykyä optimaalisesti laskettuun polkuun


print("Verrataan A*-algoritmin suorituskykyä optimaalisesti laskettuun polkuun, 10 ensimmäistä polkua:\n")

for i, (start, goal, optimal_length) in enumerate(scenarios[:10]):
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


summary = []

for i, (start, goal, optimal_length) in enumerate(scenarios[:100]):
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

# Yhteenvedon tulostus
print(f"{'Skenaario':<9}{'Alku':<15}{'Loppu':<15}{'Optimaalinen':<13}"
      f"{'JPS pituus':<12}{'JPS virhe':<12}{'JPS aika':<10}{'JPS hypyt':<12}"
      f"{'A* pituus':<12}{'A* virhe':<12}{'A* aika':<10}{'A* open set':<14}")

for row in summary:
    print(f"{row['Skenaario']:<9}{str(row['Alku']):<15}{str(row['Loppu']):<15}{row['Optimaalinen']:<13}"
          f"{str(row['JPS pituus']):<12}{str(row['JPS virhe']):<12}{row['JPS aika']:<10}{row['JPS hyppypisteet']:<12}"
          f"{str(row['A* pituus']):<12}{str(row['A* virhe']):<12}{row['A* aika']:<10}{row['A* open set']:<14}")


# Testien visualisointi
scenarios = [row['Skenaario'] for row in summary]
jps_times = [row['JPS aika'] for row in summary]
astar_times = [row['A* aika'] for row in summary]
jps_errors = [row['JPS virhe'] for row in summary]
astar_errors = [row['A* virhe'] for row in summary]

# Korvataan None-arvot nollilla, jotta matplotlib ei kaadu
jps_errors = [err if err is not None else 0 for err in jps_errors]
astar_errors = [err if err is not None else 0 for err in astar_errors]

# Luodaan pylväsdiagrammi suoritusaikojen ja virheiden vertailuun

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Suoritusaikojen visualisointi
ax1.bar(scenarios, jps_times, width=0.4, label='JPS aika', align='center')
ax1.bar(scenarios, astar_times, width=0.4, label='A* aika', align='edge')
ax1.set_ylabel('Aika (s)')
ax1.set_title('Suoritusaikojen vertailu')
ax1.legend()

# Virheiden visualisointi
ax2.bar(scenarios, jps_errors, width=0.4, label='JPS virhe', align='center')
ax2.bar(scenarios, astar_errors, width=0.4, label='A* virhe', align='edge')
ax2.set_ylabel('Polun pituuden virhe')
ax2.set_title('Polun pituuden virheiden vertailu')
ax2.legend()

plt.xlabel('Skenaario')
plt.tight_layout()
plt.show()

# Visualisoidaan halutut skenaariot eli reitit JPS- ja A*-algoritmeilla
visualize_selected_scenarios(summary, np_map)