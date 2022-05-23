#
#
#
#   THE CODE CONTAINED IN THIS PYTHON SCRIPT HAS BEEN WRITTEN
#          STARTING FROM THE FOLLOWING REPOSITORY:
#       https://github.com/BOHRTECHNOLOGY/quantum_tsp
#
#
#
import itertools
import numpy as np
import os
import pandas as pd
import random
import time

from datetime import timedelta
from dwave.system import LeapHybridSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

from problems_utils.utils import select_input_data

from qals.qals_algorithm import function_f
from qals.solvers import annealing, hybrid, stub_solver
from qals.utils import Colors, now


def delete_extra_params_keys(config, num_nodes):
    if num_nodes is not None:
        del config['problem_params']['num_nodes']

    return None


def load_tsp_params(config):
    data_filepath, num_nodes = None, None
    if len(config['problem_params']) != 0:
        data_filepath = config['problem_params']['tsp_data_file'] \
            if 'tsp_data_file' in config['problem_params'] and \
               os.path.exists(config['problem_params']['tsp_data_file']) \
            else data_filepath
        num_nodes = config['problem_params']['num_nodes'] if 'num_nodes' in config['problem_params'] else num_nodes

    if data_filepath is not None:
        num_nodes = delete_extra_params_keys(config, num_nodes)
    else:
        if num_nodes is None or num_nodes <= 0:
            num_nodes = delete_extra_params_keys(config, num_nodes)

            input_data_selection = input("Do you want to use an existent TSP problem file? (y/n) ")
            while input_data_selection not in ['y', 'n']:
                input_data_selection = input("Do you want to use an existent TSP problem file? (y/n) ")

            if input_data_selection == 'y':
                data_filepath, _ = select_input_data('tsp')
                config['problem_params']['tsp_data_file'] = data_filepath
            else:
                num_nodes = int(input("Insert the desired number of nodes (cities): "))
                while num_nodes <= 0:
                    num_nodes = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid number of nodes" + Colors.ENDC
                                          + "] Insert the desired number of nodes (cities): "))
                config['problem_params']['num_nodes'] = num_nodes

    bruteforce, dwave, hybrid = False, False, False
    if len(config['additional_params']) != 0:
        bruteforce = config['additional_params']['bruteforce'] if 'bruteforce' in config['additional_params']\
                        else bruteforce
        dwave = config['additional_params']['dwave'] if 'dwave' in config['additional_params'] else dwave
        hybrid = config['additional_params']['hybrid'] if 'hybrid' in config['additional_params'] else hybrid

    return data_filepath, num_nodes, bruteforce, dwave, hybrid


def generate_and_save_nodes(num_nodes, filepath):
    num_coordinates, coordinates_range = 2, 10
    nodes_array = np.array([
        [random.random() * coordinates_range for _ in range(0, num_coordinates)]
        for _ in range(0, num_nodes)
    ])

    pd.DataFrame(data=nodes_array, columns=["x", "y"]).to_csv(filepath, index=False)

    return nodes_array


def load_nodes(filepath):
    data = pd.read_csv(filepath)
    nodes_array = np.array([[x, y] for x, y in zip(data['x'], data['y'])])

    return nodes_array


def distance(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)


def get_tsp_matrix(nodes_array):
    num_nodes = len(nodes_array)
    matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            matrix[i][j] = distance(nodes_array[i], nodes_array[j])
            matrix[j][i] = matrix[i][j]

    return matrix


def add_cost_objective(distance_matrix, cost_constant, qubo_dict):
    num_nodes = len(distance_matrix)
    for t in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                qubit_a = t * num_nodes + i
                qubit_b = (t + 1) % num_nodes * num_nodes + j
                qubo_dict[(qubit_a, qubit_b)] = cost_constant * distance_matrix[i][j]


def add_time_constraints(distance_matrix, constraint_constant, qubo_dict):
    num_nodes = len(distance_matrix)
    for t in range(num_nodes):
        for i in range(num_nodes):
            qubit_a = t * num_nodes + i
            if (qubit_a, qubit_a) not in qubo_dict.keys():
                qubo_dict[(qubit_a, qubit_a)] = -constraint_constant
            else:
                qubo_dict[(qubit_a, qubit_a)] += -constraint_constant
            for j in range(i+1, num_nodes):
                qubit_b = t * num_nodes + j
                qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_constant


def add_position_constraints(distance_matrix, constraint_constant, qubo_dict):
    num_nodes = len(distance_matrix)
    for i in range(num_nodes):
        for t1 in range(num_nodes):
            qubit_a = t1 * num_nodes + i
            if (qubit_a, qubit_a) not in qubo_dict.keys():
                qubo_dict[(qubit_a, qubit_a)] = -constraint_constant
            else:
                qubo_dict[(qubit_a, qubit_a)] += -constraint_constant
            for t2 in range(t1+1, num_nodes):
                qubit_b = t2 * num_nodes + i
                qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_constant


def build_TSP_QUBO_problem(nodes_array, qubo_size):
    qubo_problem = dict()

    tsp_matrix = get_tsp_matrix(nodes_array)
    constraints_constant_A = np.amax(tsp_matrix) * len(tsp_matrix)
    cost_constant_B = 1

    add_cost_objective(tsp_matrix, cost_constant_B, qubo_problem)
    add_time_constraints(tsp_matrix, constraints_constant_A, qubo_problem)
    add_position_constraints(tsp_matrix, constraints_constant_A, qubo_problem)

    Q = np.zeros((qubo_size, qubo_size))
    for x, y in qubo_problem.keys():
        Q[x][y] = qubo_problem[x, y]

    return qubo_problem, Q, tsp_matrix, (constraints_constant_A, cost_constant_B)


def calculate_cost(cost_matrix, solution):
    cost = 0
    for i in range(len(solution)):
        a = i
        b = (i+1) % len(solution)
        cost += cost_matrix[solution[a]][solution[b]]

    return cost


def solve_tsp_brute_force(cost_matrix):
    number_of_nodes = len(cost_matrix)
    all_permutations = [list(x) for x in itertools.permutations(range(0, number_of_nodes))]

    best_permutation = all_permutations[0]
    best_cost = calculate_cost(cost_matrix, all_permutations[0])
    for permutation in all_permutations[1:]:
        current_cost = calculate_cost(cost_matrix, permutation)
        if current_cost < best_cost:
            best_permutation = permutation
            best_cost = current_cost
    
    return np.array(best_permutation), round(best_cost, 2)


def solve_tsp_annealer(qubo_dict, k):
    response = annealing(qubo_dict, EmbeddingComposite(DWaveSampler()), k)

    return np.array(response)


def solve_tsp_hybrid(qubo_dict):
    response = hybrid(qubo_dict, LeapHybridSampler())

    return np.array(response)


def check_solution_validity(z, num_nodes):
    valid = True

    nodes_in_positions = np.split(z, num_nodes)
    visited_nodes = list()
    for nodes_in_position in nodes_in_positions:
        nodes = np.where(nodes_in_position == 1)[0]
        if len(nodes) != 1 or nodes[0] in visited_nodes:
            valid = False
        else:
            visited_nodes.append(nodes[0])

    return valid


def points_order_to_binary_state(points_order):
    number_of_points = len(points_order)

    binary_state = np.zeros(number_of_points ** 2, dtype=int)
    for p in range(0, number_of_points):
        j = points_order[p]
        binary_state[number_of_points * p + j] = 1

    return binary_state


def binary_state_to_points_order(binary_state):
    points_order = []
    number_of_points = int(np.sqrt(len(binary_state)))

    for p in range(number_of_points):
        for j in range(number_of_points):
            if binary_state[number_of_points * p + j] == 1:
                points_order.append(j)

    return np.array(points_order)


def refine_TSP_solution(original_solution):
    num_nodes = int(np.sqrt(len(original_solution)))

    all_nodes = list(range(num_nodes))
    raw = dict()
    for i in range(num_nodes):
        raw[i] = list()
    keep = set()
    diff = list()
    indexes = list()
    left = list()

    refined_solution = np.array([-1 for _ in range(num_nodes)])
    for i in range(num_nodes):
        for j in range(num_nodes):
            if original_solution[num_nodes * i + j] == 1:
                raw[i].append(j)
    
    for i in range(num_nodes):
        if len(raw[i]) == 1:
            refined_solution[i] = raw[i][0]
            keep.add(raw[i][0])

    for i in range(num_nodes):
        if len(raw[i]) > 1:
            for node in raw[i]:
                if node not in keep:
                    diff.append(node)
            
            if len(diff) > 0:
                random_node = random.choice(diff)
                refined_solution[i] = random_node
                keep.add(random_node)
                diff.clear()

    for i in range(num_nodes):
        for j in range(num_nodes):
            if refined_solution[j] == i:
                indexes.append(j)

        if len(indexes) > 1:
            random.shuffle(indexes)
            random_pos = indexes[0]
            for pos in indexes:
                if pos == random_pos:
                    refined_solution[pos] = i
                else:
                    refined_solution[pos] = -1

        indexes.clear()

    for node in all_nodes:
        if node not in keep:
            left.append(node)
        
    for i in range(num_nodes):
        if refined_solution[i] == -1:
            random_node = random.choice(left)
            refined_solution[i] = random_node
            left.remove(random_node)

    return np.array(refined_solution)


def refine_TSP_solution_and_format_output(method, z_star, num_nodes, Q, log_string, tsp_matrix, avg_iteration_time,
                                          total_timedelta, min_value_found, convergence, iterations_num,
                                          penalty_coefficients):
    output_dict = dict()
    output_dict['type'] = method

    valid = check_solution_validity(z_star, num_nodes)
    if not valid:
        output_dict['solution'] = refine_TSP_solution(z_star)
        output_dict['refinement'] = True
        output_dict['refined_z_star'] = points_order_to_binary_state(output_dict['solution'])
        output_dict['refined_qubo_image'] = function_f(Q, output_dict['refined_z_star'])

        if log_string is not None:
            padding = 10
            log_string += "[" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC + "]" + " "*padding \
                          + "Solution not valid.\n"
            log_string += "[" + Colors.BOLD + Colors.WARNING + "VALID" + Colors.ENDC + "]" + " "*padding \
                          + "Refinement required \n"
    else:
        output_dict['solution'] = binary_state_to_points_order(z_star)
        output_dict['refinement'] = False
        output_dict['refined_z_star'], output_dict['refined_qubo_image'] = [], None

    output_dict['cost'] = round(calculate_cost(tsp_matrix, output_dict['solution']), 2)

    output_dict['avg_iter_time'] = avg_iteration_time
    output_dict['tot_time'] = total_timedelta

    output_dict['z_star'] = z_star
    output_dict['qubo_image'] = min_value_found

    output_dict['convergence'] = convergence
    output_dict['iterations'] = iterations_num

    output_dict['A_penalty'], output_dict['B_penalty'] = penalty_coefficients[0], penalty_coefficients[1]

    return output_dict, log_string


def add_TSP_info_to_out_df(df, dictionary):
    df['solution'][dictionary['type']] = dictionary['solution']
    df['cost'][dictionary['type']] = dictionary['cost']
    df['refinement'][dictionary['type']] = dictionary['refinement']
    df['avg. iteration time'][dictionary['type']] = dictionary['avg_iter_time']
    df['total time (w/o refinement)'][dictionary['type']] = dictionary['tot_time']
    df['z*'][dictionary['type']] = dictionary['z_star']
    df['f_Q(z*)'][dictionary['type']] = dictionary['qubo_image']
    df['refined(z*)'][dictionary['type']] = dictionary['refined_z_star']
    df['f_Q(refined(z*))'][dictionary['type']] = dictionary['refined_qubo_image']
    df['convergence'][dictionary['type']] = dictionary['convergence']
    df['iterations'][dictionary['type']] = dictionary['iterations']
    df['A penalty'][dictionary['type']] = dictionary['A_penalty']
    df['B penalty'][dictionary['type']] = dictionary['B_penalty']


def solve_TSP(tsp_matrix, qubo_problem_dict, Q, out_df, random_seeds, penalty_coefficients,
              bruteforce=True, d_wave=True, hybrid=True):
    if bruteforce or d_wave or hybrid:
        print("\t\t" + Colors.BOLD + Colors.HEADER + " TSP PROBLEM SOLVER..." + Colors.ENDC)

    # bruteforce
    random.seed(random_seeds[0])
    if bruteforce:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Solving problem with bruteforce ... ")
        bf = dict()
        bf['type'] = 'Bruteforce'

        start_bf = time.time()
        bf['solution'], bf['cost'] = solve_tsp_brute_force(tsp_matrix)
        bf['refinement'], bf['avg_iter_time'] = False, None
        bf['tot_time'] = timedelta(seconds=(time.time()-start_bf))
        bf['z_star'] = points_order_to_binary_state(bf['solution'])
        bf['qubo_image'] = function_f(Q, bf['z_star'])
        bf['refined_z_star'], bf['refined_qubo_image'] = [], None
        bf['convergence'], bf['iterations'] = None, None
        bf['A_penalty'], bf['B_penalty'] = penalty_coefficients[0], penalty_coefficients[1]

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + f"] Bruteforce completed ")

        add_TSP_info_to_out_df(out_df, bf)

    num_nodes = len(tsp_matrix)

    # D-Wave quantum annealing
    random.seed(random_seeds[1])
    if d_wave:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Start computing D-Wave solution ... ")
        num_reads = 1000

        start_qa = time.time()
        z_star = solve_tsp_annealer(qubo_problem_dict, num_reads)
        total_time = timedelta(seconds=(time.time()-start_qa))

        qa, _ = refine_TSP_solution_and_format_output('D-Wave', z_star, num_nodes, Q, None, tsp_matrix,
                                                      None, total_time, function_f(Q, z_star), None, None,
                                                      penalty_coefficients)
        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] D-Wave solution computed")

        add_TSP_info_to_out_df(out_df, qa)

    # Hybrid
    random.seed(random_seeds[2])
    if hybrid:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Start computing Hybrid solution ... ")

        start_hy = time.time()
        z_star = solve_tsp_hybrid(qubo_problem_dict)
        total_time = timedelta(seconds=(time.time()-start_hy))

        hy, _ = refine_TSP_solution_and_format_output('Hybrid', z_star, num_nodes, Q, None, tsp_matrix,
                                                      None, total_time, function_f(Q, z_star), None, None,
                                                      penalty_coefficients)
        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] Hybrid solution computed")

        add_TSP_info_to_out_df(out_df, hy)

    if bruteforce or d_wave or hybrid:
        print("\n\t" + Colors.BOLD + Colors.HEADER + "        TSP PROBLEM SOLVER END" + Colors.ENDC)
