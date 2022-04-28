#!/usr/local/bin/python3
#
#
#
#   THE CODE IN THIS PYTHON SCRIPT IS NOT WRITTEN BY ME
#       I ARRANGED AN ALREADY EXISTING SCRIPT FROM:
#      https://github.com/BOHRTECHNOLOGY/quantum_tsp
#
#
#
import itertools
import numpy as np
import pandas as pd
import random
import time

from datetime import timedelta
from dwave.system import LeapHybridSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

from qals.colors import Colors
from qals.solvers import annealer, hybrid, stub_solver
from qals.utils import now


def generate_and_save_nodes(num_nodes, filepath):
    nodes_list = []
    for i in range(num_nodes):
        nodes_list.append(np.array([random.random() for _ in range(0, 2)]) * 10)
    nodes_array = np.array(nodes_list)

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
            for j in range(num_nodes):
                qubit_b = t * num_nodes + j
                if i != j:
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
            for t2 in range(num_nodes):
                qubit_b = t2 * num_nodes + i
                if t1 != t2:
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_constant


def build_TSP_QUBO_problem(nodes_array, qubo_size):
    qubo_problem = dict()

    tsp_matrix = get_tsp_matrix(nodes_array)
    constraint_constant = np.amax(tsp_matrix) * len(tsp_matrix)
    cost_constant = 1

    add_cost_objective(tsp_matrix, cost_constant, qubo_problem)
    add_time_constraints(tsp_matrix, constraint_constant, qubo_problem)
    add_position_constraints(tsp_matrix, constraint_constant, qubo_problem)

    Q = np.zeros((qubo_size, qubo_size))
    for x, y in qubo_problem.keys():
        Q[x][y] = qubo_problem[x, y]

    return qubo_problem, Q, tsp_matrix


def calculate_cost(cost_matrix, solution):
    cost = 0
    for i in range(len(solution)):
        a = i % len(solution)
        b = (i+1) % len(solution)
        cost += cost_matrix[solution[a]][solution[b]]

    return cost


def solve_tsp_brute_force(nodes_array):
    number_of_nodes = len(nodes_array)
    initial_order = range(0, number_of_nodes)
    all_permutations = [list(x) for x in itertools.permutations(initial_order)]
    cost_matrix = get_tsp_matrix(nodes_array)
    best_permutation = all_permutations[0]
    best_cost = calculate_cost(cost_matrix, all_permutations[0])

    for permutation in all_permutations:
        current_cost = calculate_cost(cost_matrix, permutation)
        if current_cost < best_cost:
            best_permutation = permutation
            best_cost = current_cost
    
    return np.array(best_permutation), round(best_cost, 2)


def binary_state_to_points_order(binary_state):
    points_order = []
    number_of_points = int(np.sqrt(len(binary_state)))

    for p in range(number_of_points):
        for j in range(number_of_points):
            if binary_state[number_of_points * p + j] == 1:
                points_order.append(j)

    return points_order


def solve_tsp_annealer(qubo_dict, k):
    response = annealer(qubo_dict, EmbeddingComposite(DWaveSampler()), k)

    return np.array(response)


def solve_tsp_hybrid(qubo_dict):
    response = hybrid(qubo_dict, LeapHybridSampler())

    return np.array(response)


def check_solution_validity(z, num_nodes):
    valid = True

    nodes_positions = np.split(z, num_nodes)
    visited_nodes = list()
    for node_position in nodes_positions:
        if np.count_nonzero(node_position == 1) != 1:
            valid = False
        where = str(np.where(node_position == 1))
        if str(np.where(node_position == 1)) in visited_nodes:
            valid = False
        else:
            visited_nodes.append(where)

    return valid


def advance(iter, rnd):
    iterator = next(iter)
    while random.random() > rnd:
        iterator = next(iter, iterator)

    return iterator


def refine_TSP_solution(original_solution):
    num_nodes = int(np.sqrt(len(original_solution)))

    raw = dict()
    for i in range(num_nodes):
        raw[i] = list()
    keep = list()
    all_ = list()
    diff = list()
    indexes = list()

    refined_solution = np.array([-1 for i in range(num_nodes)])
    for i in range(num_nodes):
        for j in range(num_nodes):
            if original_solution[num_nodes * i + j] == 1:
                raw[i].append(j)
    
    for i in range(num_nodes):
        if len(raw[i]) == 1:
            keep.append(raw[i][0])
            refined_solution[i] = raw[i][0]
        all_.append(i)

    for i in range(num_nodes):
        if len(raw[i]) > 1:
            for it in raw[i]:
                if it not in keep: 
                    diff.append(it)
            
            if len(diff) > 0:
                it = advance(iter(diff), random.random() % len(diff))
                refined_solution[i] = it
                keep.append(it)
                diff.clear()

    for i in range(num_nodes):
        for j in range(num_nodes):
            if refined_solution[j] == i:
                indexes.append(j) 

        if len(indexes) > 1:
            random.shuffle(indexes)
            index = indexes[0]
            for it in indexes:
                if it == index: 
                    refined_solution[it] = i
                else:
                    refined_solution[it] = -1
            keep.append(i)

        indexes.clear()

    for it in all_:
        if it not in keep: 
            diff.append(it)
        
    for i in range(num_nodes):
        if refined_solution[i] == -1 and len(diff) != 0:
            it = advance(iter(diff), random.random() % len(diff))
            refined_solution[i] = it
            diff.remove(it)

    return refined_solution


def refine_TSP_solution_and_format_output(method, z_star, num_nodes, log_string, tsp_matrix, avg_response_time,
                                          total_timedelta):
    output_dict = dict()
    output_dict['type'] = method

    valid = check_solution_validity(z_star, num_nodes)
    if not valid:
        output_dict['solution'] = list(refine_TSP_solution(z_star))
        output_dict['refinement'] = True
        if log_string is not None:
            padding = 10
            log_string += "[" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC + "]" + " "*padding \
                          + "Solution not valid.\n"
            log_string += "[" + Colors.BOLD + Colors.WARNING + "VALID" + Colors.ENDC + "]" + " "*padding \
                          + "Refinement required \n"
    else:
        output_dict['solution'] = binary_state_to_points_order(z_star)
        output_dict['refinement'] = False

    output_dict['cost'] = round(calculate_cost(tsp_matrix, output_dict['solution']), 2)

    output_dict['avg_resp_time'] = avg_response_time
    output_dict['tot_time'] = total_timedelta

    output_dict['z_star'] = z_star

    return output_dict, log_string


def add_TSP_info_to_out_df(df, dictionary):
    df['Solution'][dictionary['type']] = dictionary['solution']
    df['Cost'][dictionary['type']] = dictionary['cost']
    df['Refinement'][dictionary['type']] = dictionary['refinement']
    df['Avg. response time'][dictionary['type']] = dictionary['avg_resp_time']
    df['Total time (w/o refinement)'][dictionary['type']] = dictionary['tot_time']
    df['Z*'][dictionary['type']] = dictionary['z_star']


def solve_TSP(nodes_array, qubo_problem, tsp_matrix, out_df, random_seeds, bruteforce=True, d_wave=True, hybrid=True):
    print("\t\t" + Colors.BOLD + Colors.HEADER + " TSP PROBLEM SOLVER..." + Colors.ENDC)

    # bruteforce
    random.seed(random_seeds[0])
    if bruteforce:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Solving problem with bruteforce ... ")
        bf = dict()
        bf['type'] = 'Bruteforce'

        start_bf = time.time()
        bf['solution'], bf['cost'] = solve_tsp_brute_force(nodes_array)
        bf['refinement'], bf['avg_resp_time'] = False, None
        bf['tot_time'] = timedelta(seconds=int(time.time()-start_bf)) if int(time.time()-start_bf) > 0 \
            else time.time()-start_bf
        bf['z_star'] = []

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + f"] Bruteforce completed ")

        add_TSP_info_to_out_df(out_df, bf)

    num_nodes = len(nodes_array)

    # D-Wave quantum annealing
    random.seed(random_seeds[1])
    if d_wave:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Start computing D-Wave solution ... ")

        start_qa = time.time()
        z_star = solve_tsp_annealer(qubo_problem, 1000)
        total_time = timedelta(seconds=int(time.time()-start_qa)) if int(time.time()-start_qa) > 0 \
            else time.time()-start_qa

        qa, _ = refine_TSP_solution_and_format_output('D-Wave', z_star, num_nodes, None, tsp_matrix, None, total_time)
        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] D-Wave solution computed")

        add_TSP_info_to_out_df(out_df, qa)

    # Hybrid
    random.seed(random_seeds[2])
    if hybrid:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Start computing Hybrid solution ... ")

        start_hy = time.time()
        z_star = solve_tsp_hybrid(qubo_problem)
        total_time = timedelta(seconds=int(time.time()-start_hy)) if int(time.time()-start_hy) > 0 \
            else time.time()-start_hy

        hy, _ = refine_TSP_solution_and_format_output('Hybrid', z_star, num_nodes, None, tsp_matrix, None, total_time)
        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] Hybrid solution computed")

        add_TSP_info_to_out_df(out_df, hy)
        
    print("\n\t" + Colors.BOLD + Colors.HEADER + "        TSP PROBLEM SOLVER END" + Colors.ENDC)
    
    return tsp_matrix, qubo_problem
