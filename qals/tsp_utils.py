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
import time
import numpy as np
from datetime import datetime, timedelta
from qals.colors import Colors
from qals.solvers import annealer, hybrid, stub_annealer
from dwave.system.samplers import DWaveSampler           
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler
import pandas as pd
import random


def create_nodes_array(n):
    nodes_list = []
    for i in range(n):
        nodes_list.append(np.array([random.random() for _ in range(0, 2)]) * 10)

    return np.array(nodes_list)


def get_nodes(n, filepath):
    try:
        data = pd.read_csv(filepath)
        nodes_array = np.array([[x, y] for x, y in zip(data['x'], data['y'])])
    except FileNotFoundError:
        nodes_array = create_nodes_array(n)
        pd.DataFrame(data=nodes_array, columns=["x", "y"]).to_csv(filepath, index=False)

    return nodes_array


def distance(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)


def get_tsp_matrix(nodes_array):
    n = len(nodes_array)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            matrix[i][j] = distance(nodes_array[i], nodes_array[j])
            matrix[j][i] = matrix[i][j]

    return matrix


def add_cost_objective(distance_matrix, cost_constant, qubo_dict):
    n = len(distance_matrix)
    for t in range(n):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                qubit_a = t * n + i
                qubit_b = (t + 1) % n * n + j
                qubo_dict[(qubit_a, qubit_b)] = cost_constant * distance_matrix[i][j]


def add_time_constraints(distance_matrix, constraint_constant, qubo_dict):
    n = len(distance_matrix)
    for t in range(n):
        for i in range(n):
            qubit_a = t * n + i
            if (qubit_a, qubit_a) not in qubo_dict.keys():
                qubo_dict[(qubit_a, qubit_a)] = -constraint_constant
            else:
                qubo_dict[(qubit_a, qubit_a)] += -constraint_constant
            for j in range(n):
                qubit_b = t * n + j
                if i != j:
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_constant


def add_position_constraints(distance_matrix, constraint_constant, qubo_dict):
    n = len(distance_matrix)
    for i in range(n):
        for t1 in range(n):
            qubit_a = t1 * n + i
            if (qubit_a, qubit_a) not in qubo_dict.keys():
                qubo_dict[(qubit_a, qubit_a)] = -constraint_constant
            else:
                qubo_dict[(qubit_a, qubit_a)] += -constraint_constant
            for t2 in range(n):
                qubit_b = t2 * n + i
                if t1 != t2:
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_constant


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

    start = time.time()
    for permutation in all_permutations:
        current_cost = calculate_cost(cost_matrix, permutation)
        if current_cost < best_cost:
            best_permutation = permutation
            best_cost = current_cost
    
    return np.array(best_permutation), round(best_cost, 2), time.time()-start


def binary_state_to_points_order(binary_state):
    points_order = []
    number_of_points = int(np.sqrt(len(binary_state)))

    for p in range(number_of_points):
        for j in range(number_of_points):
            if binary_state[(number_of_points) * p + j] == 1:
                points_order.append(j)

    return points_order


def solve_tsp_annealer(qubo_dict, k):
    response = annealer(qubo_dict, EmbeddingComposite(DWaveSampler()), k)

    return np.array(response)


def solve_tsp_hybrid(qubo_dict):
    response = hybrid(qubo_dict, LeapHybridSampler())

    return np.array(response)


def advance(iter, rnd):
    iterator = next(iter)
    while random.random() > rnd:
        iterator = next(iter, iterator)

    return iterator


def get_TSP_solution(response, refinement=True):
    n = int(np.sqrt(len(response)))

    raw = dict()
    for i in range(n):
        raw[i] = list()
    keep = list()
    all_ = list()
    diff = list()
    indexes = list()

    if not refinement:
        solution = list()
        for i in range(n):
            for j in range(n):
                if(response[n*i + j] == 1):
                    last = j
            if (last != -1):
                solution.append(last)
            last = -1
    else:
        solution = np.array([-1 for i in range(n)])
        for i in range(n):
            for j in range(n):
                if (response[n*i +j] == 1):
                    raw[i].append(j)
        
        for i in range(n):
            if len(raw[i]) == 1:
                keep.append(raw[i][0])
                solution[i] = raw[i][0]
            all_.append(i)            

        for i in range(n):
            if len(raw[i]) > 1:
                for it in raw[i]:
                    if it not in keep: 
                        diff.append(it)
                
                if len(diff) > 0:
                    it = advance(iter(diff), random.random() % len(diff))
                    solution[i] = it
                    keep.append(it)
                    diff.clear()

        for i in range(n):
            for j in range(n):
                if solution[j] == i:
                    indexes.append(j) 

            if len(indexes) > 1:
                random.shuffle(indexes)
                index = indexes[0]
                for it in indexes:
                    if it == index: 
                        solution[it] = i 
                    else:
                        solution[it] = -1
                keep.append(i)

            indexes.clear()

        for it in all_:
            if it not in keep: 
                diff.append(it)
            
        for i in range(n):
            if solution[i] == -1 and len(diff) != 0:
                it = advance(iter(diff), random.random() % len(diff))
                solution[i] = it
                diff.remove(it)

    return solution


def add_TSP_info_to_df(df, dictionary):
    df['Solution'][dictionary['type']] = dictionary['sol']
    df['Cost'][dictionary['type']] = dictionary['cost']
    df['Fixed solution'][dictionary['type']] = dictionary['fixsol']
    df['Fixed cost'][dictionary['type']] = dictionary['fixcost']
    df['Response time'][dictionary['type']] = dictionary['rtime']
    df['Total time'][dictionary['type']] = dictionary['ttime']
    df['Response'][dictionary['type']] = dictionary['response']


def now():
    return datetime.now().strftime("%H:%M:%S")


def generate_and_solve_TSP(n, data_filepath, out_df, bruteforce=True, d_wave=True, hybrid=True):
    print("\t\t" + Colors.BOLD + Colors.HEADER + "TSP PROBLEM SOLVER..." + Colors.ENDC)

    qubo = dict()
    nodes_array = get_nodes(n, data_filepath)
    
    tsp_matrix = get_tsp_matrix(nodes_array)
    constraint_constant = tsp_matrix.max() * len(tsp_matrix)
    cost_constant = 1    

    add_cost_objective(tsp_matrix, cost_constant, qubo)
    add_time_constraints(tsp_matrix, constraint_constant, qubo)
    add_position_constraints(tsp_matrix, constraint_constant, qubo)

    # bruteforce
    if bruteforce:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Solving problem with bruteforce ... ")
        bf = dict()
        bf['type'] = 'Bruteforce'

        start_bf = time.time()
        bf['sol'], bf['cost'], bf['rtime'] = solve_tsp_brute_force(nodes_array)
        bf['ttime'] = timedelta(seconds=int(time.time()-start_bf)) if int(time.time()-start_bf) > 0 \
            else time.time()-start_bf

        bf['fixsol'], bf['fixcost'], bf['response'] = [], [], []

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + f"] Bruteforce completed ")

        add_TSP_info_to_df(out_df, bf)


    # D-Wave quantum annealing
    if d_wave:
        print(now() +" [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Start computing D-Wave response ... ")
        qa = dict()
        qa['type'] = 'D-Wave'

        start_qa = time.time()
        qa['response'] = solve_tsp_annealer(qubo,1000)
        qa['rtime'] = timedelta(seconds=int(time.time()-start_qa)) if int(time.time()-start_qa) > 0 \
            else time.time()-start_qa

        qa['sol'] = binary_state_to_points_order(qa['response'])
        qa['cost'] = round(calculate_cost(tsp_matrix, qa['sol']), 2)

        qa['fixsol'] = list(get_TSP_solution(qa['response'], refinement=True))
        qa['fixcost'] = round(calculate_cost(tsp_matrix, qa['fixsol']), 2)

        qa['ttime'] = timedelta(seconds=int(time.time()-start_qa)) if int(time.time()-start_qa) > 0 \
            else time.time()-start_qa
        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] D-Wave response computed")

        add_TSP_info_to_df(out_df, qa)
    
    
    # Hybrid
    if hybrid:
        print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC
              + "] Start computing Hybrid response ... ")
        hy = dict()
        hy['type'] = 'Hybrid'

        start_hy = time.time()
        hy['response'] = solve_tsp_hybrid(qubo)
        hy['rtime'] = timedelta(seconds=int(time.time()-start_hy)) if int(time.time()-start_hy) > 0 \
            else time.time()-start_hy

        hy['sol'] = binary_state_to_points_order(hy['response'])
        hy['cost'] = round(calculate_cost(tsp_matrix, hy['sol']), 2)

        hy['fixsol'] = list(get_TSP_solution(hy['response'], refinement=True))
        hy['fixcost'] = round(calculate_cost(tsp_matrix, hy['fixsol']), 2)

        hy['ttime'] = timedelta(seconds=int(time.time()-start_hy)) if int(time.time()-start_hy) > 0 \
            else time.time()-start_hy

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] Hybrid response computed")

        add_TSP_info_to_df(out_df, hy)
        
    print("\n\t" + Colors.BOLD + Colors.HEADER + "   TSP PROBLEM SOLVER END" + Colors.ENDC)
    
    return tsp_matrix, qubo