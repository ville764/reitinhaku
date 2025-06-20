import numpy as np


# tällä ladataan kartta ja skenaariot tiedostoista
def load_map(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        map_data = [list(line.strip()) for line in lines[4:]]  # otsikko jätetään pois
    return map_data


#ladataan skenaariot eli valmiiksi lasketut lähtö- ja maali-koordinaatit sekä reitin pituus tiedostosta
def load_scenarios(filename):
    scenarios = []
    with open(filename, 'r') as file:
        lines = file.readlines()[1:]  # otsikko jätetään pois
        for line in lines:
            parts = line.split()
            start_x = int(parts[4])
            start_y = int(parts[5])
            goal_x = int(parts[6])
            goal_y = int(parts[7])
            optimal_length = float(parts[8])
            scenarios.append(((start_y, start_x), (goal_y, goal_x), optimal_length))
    return scenarios

# Muutetaan kartta numpy-taulukoksi, jossa 0 on esteetön solu ja 1 on esteellinen solu

def map_to_numpy(map_data):
    return np.array([[0 if cell == '.' else 1 for cell in row] for row in map_data])
