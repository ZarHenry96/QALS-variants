import numpy as np
import os

from problems_utils.utils import select_input_data


def load_qap_params(config):
    filepath, problem_name = None, None
    if len(config['problem_params']) != 0:
        if 'qap_data_file' in config['problem_params'] and \
                os.path.exists(config['problem_params']['qap_data_file']):
            filepath = config['problem_params']['qap_data_file']
            problem_name = os.path.basename(config['problem_params']['qap_data_file']).rsplit('.')[0]

    if filepath is None or problem_name is None:
        filepath, problem_name = select_input_data('qap')
        config['problem_params']['qap_data_file'] = filepath

    return filepath, problem_name


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
