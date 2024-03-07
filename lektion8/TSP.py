"""
Created on Wed Mar 3 16:39:18 2021
@author: Sila
"""

import pandas as pd
import random
import math
import matplotlib.pyplot as plt


# Reading cities from file with pandas
def read_cities_data(filename):
    cities_data = pd.read_csv(filename, sep='\s+', header=None)
    cities_data = pd.DataFrame(cities_data)
    return cities_data


# Creating a random route that travels through all cities. Shuffles the order of the cities
def generate_random_route(number_of_cities):
    route = [[i] for i in range(number_of_cities)]
    random.shuffle(route)
    return route


# Plotting the route in 2D, with matplotlib. Gets the x and y coordinates of the cities
def plot_city_route(route, x_coords, y_coords):
    plt.figure()
    for i in range(len(route) - 1):
        x_coordinates = [x_coords[route[i]], x_coords[route[i + 1]]]
        y_coordinates = [y_coords[route[i]], y_coords[route[i + 1]]]
        plt.plot(x_coordinates, y_coordinates, 'ro-')


# Calculating the distance between two cities given coordinates
def calculate_distance(city1_x, city1_y, city2_x, city2_y):
    x_distance = abs(city1_x - city2_x)
    y_distance = abs(city1_y - city2_y)
    distance = math.sqrt((x_distance ** 2) + (y_distance ** 2))
    return distance


# Calculating total distance of a route, by summing the distances between all cities
def calculate_total_distance(route, x_coords, y_coords):
    total_distance = 0
    for i in range(len(route) - 1):
        city1 = route[i][0]
        city2 = route[i + 1][0]
        total_distance += calculate_distance(x_coords[city1], y_coords[city1], x_coords[city2], y_coords[city2])
    return total_distance


# Finding elite routes. Finds the best route in the given generation, based on fitness.
def find_elites(generation, percentage):
    sorted_generation = sorted(generation, key=lambda x: calculate_total_distance(x, x_coords, y_coords))
    return sorted_generation[:math.floor(len(sorted_generation) * percentage)]


# Mutating a route. Selects two random positions in the route and swaps the cities at those positions
def mutate_route(route):
    mut_pos1 = random.randint(0, len(route) - 1)
    mut_pos2 = random.randint(0, len(route) - 1)
    if mut_pos1 == mut_pos2:
        return route
    route[mut_pos1], route[mut_pos2] = route[mut_pos2], route[mut_pos1]


# Breeding routes
def breed_routes(parent1, parent2):
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(0, len(parent1) - 1)

    while start == end:
        end = random.randint(0, len(parent1) - 1)

    if start > end:
        start, end = end, start

    child = parent1[start:end]
    remaining_cities = [city for city in parent2 if city not in child]
    child.extend(remaining_cities)

    if random.random() < 0.1:
        mutate_route(child)

    return child


# Performing crossover. Takes elite routes from the current generation and breeds them to create the next generation
def crossover(generation):
    elites = find_elites(generation, 0.1)
    next_generation = []

    for _ in range(len(generation)):
        parent1 = elites[random.randint(0, len(elites) - 1)]
        parent2 = elites[random.randint(0, len(elites) - 1)]
        next_generation.append(breed_routes(parent1, parent2))
    return next_generation


# Finding the best route in a generation
def find_best_route(generation):
    best_score = float('inf')
    best_route = []
    for route in generation:
        current_score = calculate_total_distance(route, x_coords, y_coords)
        if current_score < best_score:
            best_score = current_score
            best_route = route
    return best_route, best_score


# Creating initial routes
def create_initial_routes(number_of_routes, number_of_cities):
    routes = []
    for _ in range(number_of_routes):
        routes.append(generate_random_route(number_of_cities))
    return routes


# Main program
if __name__ == "__main__":
    cities_data = read_cities_data('TSPcities1000.txt')
    x_coords = cities_data[1]
    y_coords = cities_data[2]

    number_of_cities = 50
    population_size = 500
    epochs = 75

    current_generation = create_initial_routes(population_size, number_of_cities)
    best_score_progress = []

    best_route, best_score = find_best_route(current_generation)
    plot_city_route(best_route, x_coords, y_coords)
    best_score_progress.append(best_score)
    print('Starting best score:', best_score)

    for i in range(epochs):
        current_generation = crossover(current_generation)
        best_route, best_score = find_best_route(current_generation)
        best_score_progress.append(best_score)
        print(i, best_score)

    print('End best score:', best_score)

    plot_city_route(best_route, x_coords, y_coords)
    plt.show()

    plt.plot(best_score_progress)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Route Length)')
    plt.show()
