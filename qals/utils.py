import csv
import numpy as np
import pandas as pd
import random

from datetime import datetime


def now():
    return datetime.now().strftime("%H:%M:%S")


def csv_write(csv_file, row):
    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def numpy_vector_to_string(vector):
    return '[' + ' '.join([str(x) for x in vector]) + ']'


def tabu_to_string(S):
    string = '[' + ' '.join(['['+' '.join([str(x) for x in row])+']' for row in S]) + ']'
    return string


def generate_and_save_numbers(num_values, max_value, data_file):
    vect = [0 for _ in range(num_values)]
    for index in range(num_values):
        vect[index] = random.randint(0, max_value)

    # Save to csv file
    pd.DataFrame([vect], dtype=int).to_csv(data_file, index=False, header=False)

    return vect


def load_numbers(data_filepath):
    df = pd.read_csv(data_filepath, header=None)

    return list(df.to_numpy()[0])


def build_NPP_QUBO_problem(numbers):
    num_values = len(numbers)
    c = 0
    for i in range(num_values):
        c += numbers[i]

    QUBO = np.zeros((num_values, num_values))
    col_max = 0
    col = 0
    for row in range(num_values):
        col_max += 1
        while col < col_max:
            if row == col:
                QUBO[row][col] = numbers[row] * (numbers[row] - c)
            else:
                QUBO[row][col] = numbers[row] * numbers[col]
                QUBO[col][row] = QUBO[row][col]
            col += 1
        col = 0

    return QUBO, c


def read_integers(filename: str):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


def generate_QAP_QUBO_matrix(flow, distance, penalty):
    num_values = len(flow)
    q = np.einsum("ij,kl->ikjl", flow, distance).astype(np.float)
    i = range(len(q))

    q[i, :, i, :] += penalty
    q[:, i, :, i] += penalty
    q[i, i, i, i] -= 4 * penalty

    return q.reshape(num_values ** 2, num_values ** 2)


def build_QAP_QUBO_problem(data_filepath):
    file_iterator = iter(read_integers(data_filepath))

    n = next(file_iterator)
    flow = [[next(file_iterator) for _ in range(n)] for _ in range(n)]
    distance = [[next(file_iterator) for _ in range(n)] for _ in range(n)]

    kron_product = np.kron(flow, distance)

    penalty = (kron_product.max() * 2.25)
    matrix = generate_QAP_QUBO_matrix(flow, distance, penalty)
    y = penalty * (len(flow) + len(distance))

    return matrix, penalty, len(matrix), y
