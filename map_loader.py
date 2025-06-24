import numpy as np


def load_map(filename):
    """
    Lataa kartan tiedostosta ja palauttaa sen listana.
    
    Tiedoston oletetaan olevan muotoa, jossa ensimmäiset 4 riviä ovat
    otsikkotietoja ja varsinainen karttadata alkaa 5. riviltä.
    
    Args:
        filename (str): Ladattavan karttatieoston nimi
    
    Returns:
        list: 2D lista joka sisältää kartan merkkeinä
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        map_data = [list(line.strip()) for line in lines[4:]]  # otsikko jätetään pois
    return map_data


def load_scenarios(filename):
    """
    Lataa skenaariot tiedostosta ja palauttaa ne listana.
    
    Jokainen skenaario sisältää lähtöpisteen, maalipisteen ja optimaalisen
    reitin pituuden. Tiedoston ensimmäinen rivi oletetaan olevan otsikko.
    Koordinaatit muunnetaan (x,y) -> (y,x) muotoon.
    
    Args:
        filename (str): Ladattavan skenaariotiedoston nimi
    
    Returns:
        list: Lista tupleja muodossa ((start_y, start_x), (goal_y, goal_x), optimal_length)
    """
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


def map_to_numpy(map_data):
    """
    Muuntaa karttadatan numpy-taulukoksi polunhakualgoritmeja varten.
    
    Muuntaa merkki-pohjaisen kartan numeeriseksi taulukoksi, jossa:
    - '.' (piste) muuttuu 0:ksi (vapaa solu)
    - Kaikki muut merkit muuttuvat 1:ksi (este)
    
    Args:
        map_data (list): 2D lista joka sisältää kartan merkkeinä
    
    Returns:
        numpy.ndarray: 2D numpy-taulukko jossa 0 = vapaa, 1 = este
    """
    return np.array([[0 if cell == '.' else 1 for cell in row] for row in map_data])