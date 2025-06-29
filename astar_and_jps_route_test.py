import numpy as np
import matplotlib.pyplot as plt
import math
from astar import AStar
from jps import JPS  

def calculate_path_length(path):
    """
    Laskee reitin todellisen pituuden käyttäen Octile-etäisyyttä.
    
    Args:
        path: Lista koordinaattipareista [(x1, y1), (x2, y2), ...]
        
    Returns:
        float: Reitin kokonaispituus
    """
    if not path or len(path) < 2:
        return 0.0
    
    total_length = 0.0
    
    # Käy läpi kaikki peräkkäiset pisteet polulla
    for i in range(len(path) - 1):
        current_point = path[i]
        next_point = path[i + 1]
        
        # Laske etäisyys nykyisen ja seuraavan pisteen välillä
        distance = calculate_octile_distance(current_point, next_point)
        total_length += distance
    
    return total_length

def calculate_octile_distance(a, b):
    """Laskee Octile etäisyyden kahden pisteen välillä"""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

def create_test_grid(map_size=100, obstacle_size=26):
    """
    Luo testikartan keskellä olevalla neliönmuotoisella esteellä.
    
    Args:
        map_size: Kartan koko (map_size x map_size)
        obstacle_size: Keskellä olevan esteen koko
        
    Returns:
        numpy.array: Testikartta (0 = vapaa, 1 = este)
    """
    grid = np.zeros((map_size, map_size), dtype=int)
    
    # Määritä keskellä oleva este
    start = (map_size - obstacle_size) // 2
    end = start + obstacle_size
    
    # Täytä keskellä oleva alue esteillä
    grid[start:end, start:end] = 1
    
    return grid

def get_test_routes():
    """
    Määrittää 8 testireittiä eri suuntiin.
    
    Returns:
        dict: Testireitit avaimin ja reittidata
    """
    return {
        "Pohjoinen (N)": {
            "start": (90, 50),
            "goal": (10, 50),
            "direction": "North",
            "description": "Testaa reitinhakua suoraan pohjoiseen"
        },
        
        "Koillinen (NE)": {
            "start": (85, 15),
            "goal": (15, 85),
            "direction": "Northeast",
            "description": "Testaa reitinhakua koilliseen (diagonal)"
        },
        
        "Itä (E)": {
            "start": (50, 10),
            "goal": (50, 90),
            "direction": "East", 
            "description": "Testaa reitinhakua suoraan itään"
        },
        
        "Kaakko (SE)": {
            "start": (15, 15),
            "goal": (85, 85),
            "direction": "Southeast",
            "description": "Testaa reitinhakua kaakkoon (diagonal)"
        },
        
        "Etelä (S)": {
            "start": (10, 50),
            "goal": (90, 50),
            "direction": "South",
            "description": "Testaa reitinhakua suoraan etelään"
        },
        
        "Lounas (SW)": {
            "start": (15, 85),
            "goal": (85, 15),
            "direction": "Southwest", 
            "description": "Testaa reitinhakua lounaaseen (diagonal)"
        },
        
        "Länsi (W)": {
            "start": (50, 90),
            "goal": (50, 10),
            "direction": "West",
            "description": "Testaa reitinhakua suoraan länteen"
        },
        
        "Luode (NW)": {
            "start": (85, 85),
            "goal": (15, 15),
            "direction": "Northwest",
            "description": "Testaa reitinhakua luoteeseen (diagonal)"
        }
    }

def run_pathfinding_tests(grid, test_routes):
    """
    Suorittaa reitinhakutestit molemmilla algoritmeilla.
    
    Args:
        grid: Testikartta numpy array
        test_routes: Testireitit dictionary
        
    Returns:
        dict: Päivitetyt testireitit tuloksineen
    """
    grid_list = grid.tolist()
    
    # Luo algoritmi-instanssit
    astar = AStar(grid_list)
    jps = JPS(grid_list)
    
    # Suorita testit molemmilla algoritmeilla
    for route_name, route_data in test_routes.items():
        start_pos = route_data["start"]
        goal_pos = route_data["goal"]
        
        # A* testi
        astar_path, astar_closed, astar_nodes = astar.find_path(start_pos, goal_pos)
        if astar_path:
            route_data["astar_distance"] = round(calculate_path_length(astar_path), 2)
            route_data["astar_path"] = astar_path
        else:
            route_data["astar_distance"] = None
            route_data["astar_path"] = None
        
        route_data["astar_nodes_explored"] = len(astar_closed)
        route_data["astar_nodes_added"] = astar_nodes
        
        # JPS testi
        jps_path, jps_closed, jps_jump_points = jps.find_path(start_pos, goal_pos)
        if jps_path:
            route_data["jps_distance"] = round(calculate_path_length(jps_path), 2)
            route_data["jps_path"] = jps_path
        else:
            route_data["jps_distance"] = None
            route_data["jps_path"] = None
            
        route_data["jps_nodes_explored"] = len(jps_closed)
        route_data["jps_jump_points"] = jps_jump_points
    
    return test_routes

def visualize_astar_results(grid, test_routes):
    """
    Visualisoi A* algoritmin tulokset.
    
    Args:
        grid: Testikartta
        test_routes: Testireitit tuloksineen
    """
    fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
    axes1 = axes1.flatten()
    
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']
    
    for i, (route_name, route_data) in enumerate(test_routes.items()):
        ax = axes1[i]
        ax.imshow(grid, cmap='gray_r', alpha=0.7)
        
        start_pos = route_data["start"]
        goal_pos = route_data["goal"]
        
        # Merkitse lähtö- ja maalipisteet
        ax.plot(start_pos[1], start_pos[0], 'go', markersize=8, label='Start')
        ax.plot(goal_pos[1], goal_pos[0], 'ro', markersize=8, label='Goal')
        
        # Piirrä A* reitti
        if route_data["astar_path"]:
            path = route_data["astar_path"]
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, color=colors[i], linewidth=2, alpha=0.8, label='A* reitti')
            
            astar_dist = route_data["astar_distance"]
            astar_nodes = route_data["astar_nodes_explored"]
            ax.set_title(f"A* - {route_name}\nPituus: {astar_dist:.1f}\nTutkittu: {astar_nodes} solmua", fontsize=10)
        else:
            ax.set_title(f"A* - {route_name}\nEi reittiä", fontsize=10)
        
        ax.set_xlim(0, 99)
        ax.set_ylim(99, 0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.suptitle("A* Algoritmin testireitit - 8 suuntaa", fontsize=16, y=0.98)
    plt.show()

def visualize_jps_results(grid, test_routes):
    """
    Visualisoi JPS algoritmin tulokset.
    
    Args:
        grid: Testikartta
        test_routes: Testireitit tuloksineen
    """
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    axes2 = axes2.flatten()
    
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']
    
    for i, (route_name, route_data) in enumerate(test_routes.items()):
        ax = axes2[i]
        ax.imshow(grid, cmap='gray_r', alpha=0.7)
        
        start_pos = route_data["start"]
        goal_pos = route_data["goal"]
        
        # Merkitse lähtö- ja maalipisteet
        ax.plot(start_pos[1], start_pos[0], 'go', markersize=8, label='Start')
        ax.plot(goal_pos[1], goal_pos[0], 'ro', markersize=8, label='Goal')
        
        # Piirrä JPS reitti ja hyppypisteet
        if route_data["jps_path"]:
            path = route_data["jps_path"]
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, color=colors[i], linewidth=2, alpha=0.8, label='JPS reitti')
            
            # Merkitse hyppypisteet erikseen
            ax.plot(path_x, path_y, 'o', color=colors[i], markersize=4, alpha=0.7, label='Hyppypisteet')
            
            jps_dist = route_data["jps_distance"]
            jps_nodes = route_data["jps_nodes_explored"]
            jps_jumps = route_data["jps_jump_points"]
            ax.set_title(f"JPS - {route_name}\nPituus: {jps_dist:.1f}\nTutkittu: {jps_nodes} solmua\nHyppypisteet: {jps_jumps}", fontsize=10)
        else:
            ax.set_title(f"JPS - {route_name}\nEi reittiä", fontsize=10)
        
        ax.set_xlim(0, 99)
        ax.set_ylim(99, 0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.suptitle("JPS Algoritmin testireitit - 8 suuntaa", fontsize=16, y=0.98)
    plt.show()

def create_efficiency_comparison(test_routes):
    """
    Luo tehokkuusvertailun kaaviot.
    
    Args:
        test_routes: Testireitit tuloksineen
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    route_names = list(test_routes.keys())
    astar_nodes = [route_data["astar_nodes_explored"] for route_data in test_routes.values()]
    jps_nodes = [route_data["jps_nodes_explored"] for route_data in test_routes.values()]
    astar_distances = [route_data["astar_distance"] if route_data["astar_distance"] else 0 for route_data in test_routes.values()]
    jps_distances = [route_data["jps_distance"] if route_data["jps_distance"] else 0 for route_data in test_routes.values()]
    
    # Tutkittujen solmujen vertailu
    x = np.arange(len(route_names))
    width = 0.35
    
    ax1.bar(x - width/2, astar_nodes, width, label='A*', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, jps_nodes, width, label='JPS', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Testireitit')
    ax1.set_ylabel('Tutkittuja solmuja')
    ax1.set_title('Tutkittujen solmujen määrä')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.split(' ')[0] for name in route_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reitin pituuksien vertailu
    valid_routes = [(i, name) for i, name in enumerate(route_names) 
                    if astar_distances[i] > 0 and jps_distances[i] > 0]
    if valid_routes:
        valid_indices, valid_names = zip(*valid_routes)
        valid_astar_dist = [astar_distances[i] for i in valid_indices]
        valid_jps_dist = [jps_distances[i] for i in valid_indices]
        
        x_valid = np.arange(len(valid_names))
        ax2.bar(x_valid - width/2, valid_astar_dist, width, label='A*', alpha=0.8, color='skyblue')
        ax2.bar(x_valid + width/2, valid_jps_dist, width, label='JPS', alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Testireitit')
        ax2.set_ylabel('Reitin pituus')
        ax2.set_title('Reitin pituuksien vertailu')
        ax2.set_xticks(x_valid)
        ax2.set_xticklabels([name.split(' ')[0] for name in valid_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Tehokkuus (vähemmän solmuja = parempi)
    efficiency_ratios = []
    route_labels = []
    for i, (route_name, route_data) in enumerate(test_routes.items()):
        if route_data["astar_nodes_explored"] > 0 and route_data["jps_nodes_explored"] > 0:
            ratio = route_data["astar_nodes_explored"] / route_data["jps_nodes_explored"]
            efficiency_ratios.append(ratio)
            route_labels.append(route_name.split(' ')[0])
    
    if efficiency_ratios:
        ax3.bar(range(len(efficiency_ratios)), efficiency_ratios, alpha=0.8, color='lightgreen')
        ax3.set_xlabel('Testireitit')
        ax3.set_ylabel('A* solmut / JPS solmut')
        ax3.set_title('JPS tehokkuus (korkeampi = parempi)')
        ax3.set_xticks(range(len(route_labels)))
        ax3.set_xticklabels(route_labels, rotation=45)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Tasavertainen')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_comparison_table(test_routes):
    """
    Tulostaa vertailutaulukon.
    
    Args:
        test_routes: Testireitit tuloksineen
    """
    print("=" * 80)
    print("A* vs JPS ALGORITMIN VERTAILU")
    print("=" * 80)
    
    print(f"{'Reitti':<15} {'Suunta':<10} {'A* Pituus':<12} {'JPS Pituus':<12} {'A* Solmut':<12} {'JPS Solmut':<12} {'Tehokkuus':<10}")
    print("-" * 80)
    
    for route_name, route_data in test_routes.items():
        short_name = route_name.split(' ')[0]
        direction = route_data['direction'][:9]
        
        astar_dist = f"{route_data['astar_distance']:.2f}" if route_data['astar_distance'] else "Ei reittiä"
        jps_dist = f"{route_data['jps_distance']:.2f}" if route_data['jps_distance'] else "Ei reittiä"
        
        astar_nodes = route_data['astar_nodes_explored']
        jps_nodes = route_data['jps_nodes_explored']
        
        if astar_nodes > 0 and jps_nodes > 0:
            efficiency = f"{astar_nodes/jps_nodes:.1f}x"
        else:
            efficiency = "N/A"
        
        print(f"{short_name:<15} {direction:<10} {astar_dist:<12} {jps_dist:<12} {astar_nodes:<12} {jps_nodes:<12} {efficiency:<10}")
    
    print("=" * 80)

def print_summary_statistics(test_routes):
    """
    Tulostaa yhteenvedon statistiikot.
    
    Args:
        test_routes: Testireitit tuloksineen
    """
    valid_tests = [route_data for route_data in test_routes.values() 
                   if route_data['astar_distance'] and route_data['jps_distance']]
    
    if valid_tests:
        total_astar_nodes = sum(route_data['astar_nodes_explored'] for route_data in valid_tests)
        total_jps_nodes = sum(route_data['jps_nodes_explored'] for route_data in valid_tests)
        avg_efficiency = total_astar_nodes / total_jps_nodes if total_jps_nodes > 0 else 0
        
        print(f"\nYHTEENVETO:")
        print(f"Onnistuneet testit: {len(valid_tests)}/{len(test_routes)}")
        print(f"A* tutkitut solmut yhteensä: {total_astar_nodes}")
        print(f"JPS tutkitut solmut yhteensä: {total_jps_nodes}")
        print(f"JPS keskimääräinen tehokkuus: {avg_efficiency:.1f}x parempi")
        print(f"JPS säästää keskimäärin: {((total_astar_nodes - total_jps_nodes) / total_astar_nodes * 100):.1f}% solmuista")
    
    print("\nHUOM: Tehokkuus = A* solmut / JPS solmut (korkeampi arvo = JPS parempi)")
    print("=" * 80)

def print_jps_test_template(test_routes):
    """
    Tulostaa JPS testikoodin templaten.
    
    Args:
        test_routes: Testireitit tuloksineen
    """
    print("\n" + "=" * 80)
    print("JPS TESTIKOODIN RAKENNE:")
    print("=" * 80)
    
    print("""
# Tuo JPS algoritmi
from jps import JPS
import numpy as np

# Luo sama kartta
map_size = 100
grid = np.zeros((map_size, map_size), dtype=int)
rect_size = 26
start_rect = (map_size - rect_size) // 2
end_rect = start_rect + rect_size
grid[start_rect:end_rect, start_rect:end_rect] = 1
grid_list = grid.tolist()

# Luo JPS instanssi
jps = JPS(grid_list)

# Testireitit JPS:lle
jps_test_cases = [""")
    
    for i, (route_name, route_data) in enumerate(test_routes.items()):
        start = route_data["start"]
        goal = route_data["goal"]
        direction = route_data["direction"]
        jps_dist = route_data["jps_distance"]
        jps_nodes = route_data["jps_nodes_explored"]
        comma = "," if i < len(test_routes) - 1 else ""
        
        print(f"    {{'start': {start}, 'goal': {goal}, 'name': '{route_name}', 'direction': '{direction}', 'expected_distance': {jps_dist}, 'expected_nodes': {jps_nodes}}}{comma}")
    
    print("""]

# Suorita JPS testit
print("JPS ALGORITMIN TESTIT:")
print("=" * 50)

for i, test_case in enumerate(jps_test_cases, 1):
    print(f"\\nTEST {i}: {test_case['name']} ({test_case['direction']})")
    print(f"Lähtö: {test_case['start']} -> Maali: {test_case['goal']}")
    
    # Suorita JPS
    path, closed_set, jump_points = jps.find_path(test_case['start'], test_case['goal'])
    
    if path:
        actual_distance = calculate_path_length(path)
        nodes_explored = len(closed_set)
        
        print(f"✓ Reitti löytyi!")
        print(f"  Reitin pituus: {actual_distance:.2f} (odotettu: {test_case['expected_distance']})")
        print(f"  Tutkittuja solmuja: {nodes_explored} (odotettu: {test_case['expected_nodes']})")
        print(f"  Hyppypisteitä löydetty: {jump_points}")
        print(f"  Hyppypisteet reitillä: {len(path)}")
        
        # Tarkista tarkkuus
        if abs(actual_distance - test_case['expected_distance']) < 0.01:
            print(f"  ✓ Reitin pituus oikein!")
        else:
            print(f"  ✗ Reitin pituus eroaa odotuksesta")
            
    else:
        print(f"✗ Reittiä ei löytynyt!")
        nodes_explored = len(closed_set)
        print(f"  Tutkittuja solmuja: {nodes_explored}")
    
    print("-" * 40)
""")

def run_benchmark(show_visualizations=True, show_comparison_charts=True, show_results_table=True, show_jps_template=False):
    """
    Suorittaa täydellisen JPS vs A* benchmark testin.
    
    Args:
        show_visualizations (bool): Näytä algoritmikohtaiset visualisoinnit
        show_comparison_charts (bool): Näytä tehokkuusvertailun kaaviot
        show_results_table (bool): Tulosta tulostaulukko
        show_jps_template (bool): Tulosta JPS testikoodin template
        
    Returns:
        tuple: (grid, test_routes) - Testikartta ja tulokset
    """
    print("Aloitetaan JPS vs A* benchmark testi...")
    
    # 1. Luo testikartta
    print("Luodaan testikartta...")
    grid = create_test_grid()
    
    # 2. Määritä testireitit
    print("Määritetään testireitit...")
    test_routes = get_test_routes()
    
    # 3. Suorita reitinhakutestit
    print("Suoritetaan reitinhakutestit...")
    test_routes = run_pathfinding_tests(grid, test_routes)
    
    # 4. Visualisoi tulokset
    if show_visualizations:
        print("Luodaan A* visualisointi...")
        visualize_astar_results(grid, test_routes)
        
        print("Luodaan JPS visualisointi...")
        visualize_jps_results(grid, test_routes)
    
    # 5. Luo tehokkuusvertailu
    if show_comparison_charts:
        print("Luodaan tehokkuusvertailun kaaviot...")
        create_efficiency_comparison(test_routes)
    
    # 6. Tulosta tulokset
    if show_results_table:
        print("Tulostetaan vertailutaulukko...")
        print_comparison_table(test_routes)
        print_summary_statistics(test_routes)
    
    # 7. Tulosta JPS testikoodin template
    if show_jps_template:
        print_jps_test_template(test_routes)
    
    print("\nBenchmark testi valmis!")
    return grid, test_routes

def main():
    """
    Pääfunktio - suorittaa täydellisen benchmark testin.
    """
    # Suorita täydellinen benchmark kaikilla ominaisuuksilla
    grid, results = run_benchmark(
        show_visualizations=True,
        show_comparison_charts=True, 
        show_results_table=True,
        show_jps_template=True
    )
    
    return grid, results

if __name__ == "__main__":
    main()