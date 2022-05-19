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


def read_integers(filepath):
    with open(filepath) as f:
        return [int(elem) for elem in f.read().split()]


def build_Q(flow, distance, penalty):
    n = len(flow)

    Q = np.einsum("ij,kl->ikjl", flow, distance).astype(np.float)

    index_range = range(n)
    Q[index_range, :, index_range, :] += penalty
    Q[:, index_range, :, index_range] += penalty
    for j in range(n):
        Q[index_range, j, index_range, j] -= 4 * penalty

    return Q.reshape(n ** 2, n ** 2)


def build_QAP_QUBO_problem(data_filepath):
    file_iterator = iter(read_integers(data_filepath))

    n = next(file_iterator)
    flow = [[next(file_iterator) for _ in range(n)] for _ in range(n)]
    distance = [[next(file_iterator) for _ in range(n)] for _ in range(n)]

    kron_product = np.kron(flow, distance)
    penalty = (np.amax(kron_product) * 2.25)

    Q = build_Q(flow, distance, penalty)
    offset = penalty * (len(flow) + len(distance))

    return Q, len(Q), penalty, offset
