"""
Created on Wed Mar 3 16:39:18 2021
@author: Sila
"""

# Gettting started methods for TSP GA algorithm
# - Read cities from file
#

import pandas as pd
import random
import math

data = pd.read_csv(
    r'C:\Users\Bruger\Desktop\4.Semester\MachineLearning\lektion8\TSPcities1000.txt',
    sep='\s+', header=None)
data = pd.DataFrame(data)

import matplotlib.pyplot as plt

x = data[1]
y = data[2]


# plt.plot(x, y,'r.')
# plt.show()


def createRandomRoute():
    tour = [[i] for i in range(1000)]
    random.shuffle(tour)
    return tour


# plot the tour - Adjust range 0..len, if you want to plot only a part of the tour.
def plotCityRoute(route, start, end):
    for i in range(start, end):
        plt.plot(x[i:i + 2], y[i:i + 2], 'ro-')
    plt.show()


# Alternativ kode:
#  for i in range(0, len(route)-1):
#     plt.plot([x[route[i]],x[route[i+1]]], [y[route[i]],y[route[i+1]]], 'ro-')

# calculate distance between cities
def distancebetweenCities(city1x, city1y, city2x, city2y):
    xDistance = abs(city1x - city2x)
    yDistance = abs(city1y - city2y)
    distance = math.sqrt((xDistance ** 2) + (yDistance ** 2))
    return distance


def calculateTotalDistance(route):
    totalDistance = 0
    for i in range(0, number_of_citites - 1):
        city1 = route[i][0]
        city2 = route[i + 1][0]
        totalDistance += distancebetweenCities(x[city1], y[city1], x[city2], y[city2])
    return totalDistance


def find_elites(generation, percentage):
    sorted_gen = sorted(generation, key=calculateTotalDistance)  # no reverse because we want the lowest distance
    return sorted_gen[:math.floor(len(sorted_gen) * percentage)]


def mutate(route):
    # two random indices:
    mut_pos1 = random.randint(0, number_of_citites - 1)
    mut_pos2 = random.randint(0, number_of_citites - 1)

    # if they're the same, skip to the chase
    if mut_pos1 == mut_pos2:
        return route

    # Otherwise swap them:
    city1 = route[mut_pos1]
    city2 = route[mut_pos2]

    route[mut_pos2] = city1
    route[mut_pos1] = city2


def breed(mum, dad):
    child = [None] * number_of_citites
    # fill with mum routes
    start_at = random.randint(0, number_of_citites - 1)
    for i in range(start_at, number_of_citites):
        child[i] = mum[i]

    # fill rest with dad routes
    j = 0
    for i in range(0, number_of_citites):
        if child[i] == None:
            while dad[j] in child:
                j += 1
            child[i] = dad[j]
            j += 1

    if random.random() < 0.1:
        mutate(child)

    return child


def breed2(mum, dad):
    start = random.randint(0, number_of_citites - 1)
    end = random.randint(0, number_of_citites - 1)

    while start == end:
        end = random.randint(0, number_of_citites - 1)

    if start > end:
        start, end = end, start

    child = mum[start:end]

    remaining_citites = [city for city in dad if city not in child]
    child.extend(remaining_citites)

    return child


def crossover(generation):
    elites = find_elites(generation, 0.1)
    number_elites = len(elites)
    # next_gen = elites
    next_gen = []

    for i in range(len(generation)):
        mum = elites[random.randint(0, number_elites - 1)]
        dad = elites[random.randint(0, number_elites - 1)]
        next_gen.append(breed2(mum, dad))
    return next_gen


def find_best_route(generation):
    best_score = 1e20
    best_route = []
    for i in range(len(generation)):
        curr_score = calculateTotalDistance(generation[i])
        if curr_score < best_score:
            best_score = curr_score
            best_route = generation[i]
    return best_route, best_score


def create_routes(number_of_routes):
    routes = []
    for i in range(number_of_routes):
        routes.append(createRandomRoute())
    return routes


# distance between city number 100 and city number 105
# dist= distancebetweenCities(x[100], y[100], x[105], y[105])
# print('Distance, % target: ', dist)

number_of_citites = 10
size = 200
epochs = 10
current_gen = create_routes(size)
best_score_progress = []  # Tracks progress

best_route, best_score = find_best_route(current_gen)
best_score_progress.append(best_score)
print('Starting best score, % target: ', best_score)
for i in range(epochs):
    current_gen = crossover(current_gen)
    best_route, best_score = find_best_route(current_gen)
    best_score_progress.append(best_score)
    print(i, best_score)

best_route, best_score = find_best_route(current_gen)
# GA has completed required generation
print('End best score, % target: ', best_score)

plotCityRoute(best_route, 0, 100)

plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best Fitness - route length - in Generation')
# plt.show()
plt.show()
