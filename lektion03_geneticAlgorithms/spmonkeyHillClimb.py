import string
import random

target_string = "to be or not to be that is the question"
population_size = 10000
string_length = len(target_string)

def generate_random_string(length):
    letters = "abcdefghijklmnopqrstuvwxyz "
    return ''.join(random.choice(letters) for _ in range(length))

def calculate_fitness(string, target):
    return sum(1 for expected, actual in zip(target, string) if expected == actual)

# Genererer 10000 random strings
population = [generate_random_string(string_length) for _ in range(population_size)]

fitness_scores = [calculate_fitness(string, target_string) for string in population]

best_index = fitness_scores.index(max(fitness_scores))

# Printer den bedste string og dens fitness
print("Best string:", population[best_index])
print("Fitness:", fitness_scores[best_index])
