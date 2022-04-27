import csv
import dwave_networkx as dnx
import networkx as nx
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


def generate_and_save_S(num_values, max_value, data_file):
    vect = [0 for _ in range(num_values)]
    for index in range(num_values):
        vect[index] = random.randint(0, max_value)

    # Save to csv file
    pd.DataFrame([vect], dtype=int).to_csv(data_file, index=False, header=False)

    return vect


def load_S(data_filepath):
    df = pd.read_csv(data_filepath, header=None)

    return list(df.to_numpy()[0])


def build_NPP_QUBO_problem(S):
    num_values = len(S)
    c = 0
    for i in range(num_values):
        c += S[i]

    QUBO = np.zeros((num_values, num_values))
    col_max = 0
    col = 0
    for row in range(num_values):
        col_max += 1
        while col < col_max:
            if row == col:
                QUBO[row][col] = S[row]*(S[row]-c)
            else:
                QUBO[row][col] = S[row] * S[col]
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


def generate_chimera_topology(qubits_num):
    G = dnx.chimera_graph(16)
    tmp = nx.to_dict_of_lists(G)

    rows = []
    cols = []
    for i in range(qubits_num):
        rows.append(i)
        cols.append(i)
        for j in tmp[i]:
            if j < qubits_num:
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))


def generate_pegasus_topology(qubits_num):
    G = dnx.pegasus_graph(16)
    tmp = nx.to_numpy_matrix(G)

    rows = []
    cols = []
    for i in range(qubits_num):
        rows.append(i)
        cols.append(i)
        for j in range(qubits_num):
            if tmp.item(i, j):
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))
