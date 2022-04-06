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
# import os
import csv
import numpy as np
# import sys
from datetime import datetime, timedelta
from qals.colors import Colors
from qals.solvers import annealer, hybrid
from dwave.system.samplers import DWaveSampler           
from dwave.system.composites import EmbeddingComposite   
# import neal
from dwave.system import LeapHybridSampler
import pandas as pd
from random import SystemRandom
random = SystemRandom()


def create_nodes_array(N):
    nodes_list = []
    for i in range(N):
        nodes_list.append(np.random.rand(2) * 10)
    return np.array(nodes_list)


def get_nodes(n,DIR):
    nodes_array = 0
    try:
        DATA = pd.read_csv(DIR)
        nodes_array = np.array([[x,y] for x,y in zip(DATA['x'], DATA['y'])])
    except FileNotFoundError:
        nodes_array = create_nodes_array(n)
        pd.DataFrame(data=nodes_array, columns=["x", "y"]).to_csv(DIR,index=False)

    return nodes_array


def distance(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)


def get_tsp_matrix(nodes_array):
    n = len(nodes_array)
    matrix = np.zeros((n,n))
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
                qubit_b = (t + 1)%n * n + j
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
        a = i%len(solution)
        b = (i+1)%len(solution)
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


def fix_solution(response, validate):
    n = int(np.sqrt(len(response)))
    solution = np.array(n)
    raw = dict()
    for i in range(n):
        raw[i] = list()
    keep = list()
    all_ = list()
    diff = list()
    indexes = list()

    if not validate:
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


# def csv_write(DIR, l):
#     with open(DIR, 'a') as file:
#         writer = csv.writer(file)
#         writer.writerow(l)


def write_TSP_csv(df, dictionary):
    # "Solution", "Cost", "Fixed solution", "Fixed cost", "Response time", "Total time", "Response"
    
    df['Solution'][dictionary['type']] = dictionary['sol']
    df['Cost'][dictionary['type']] = dictionary['cost']
    df['Fixed solution'][dictionary['type']] = dictionary['fixsol']
    df['Fixed cost'][dictionary['type']] = dictionary['fixcost']
    df['Response time'][dictionary['type']] = dictionary['rtime']
    df['Total time'][dictionary['type']] = dictionary['ttime']
    df['Response'][dictionary['type']] = dictionary['response']


def now():
    return datetime.now().strftime("%H:%M:%S")


def tsp(n, DIR, DATA, df, bruteforce=True, DWave=True, Hybrid=True):
    print("\t\t" + Colors.BOLD + Colors.HEADER + "TSP PROBLEM SOLVER..." + Colors.ENDC)
    
    columns = ["Type", "solution", "cost", "fixed solution", "fixed cost", "response time", "total time", "response"]
    
    qubo = dict()
    nodes_array = get_nodes(n, DATA)
    
    tsp_matrix = get_tsp_matrix(nodes_array)
    constraint_constant = tsp_matrix.max() * len(tsp_matrix)
    cost_constant = 1    

    add_cost_objective(tsp_matrix,cost_constant,qubo)
    add_time_constraints(tsp_matrix,constraint_constant,qubo)
    add_position_constraints(tsp_matrix,constraint_constant,qubo)

    ### BRUTEFORCE
    if bruteforce:
        print(now() +" [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC + "] Solving problem with bruteforce ... ")
        BF = dict()
        BF['type'] = 'Bruteforce'
        start_BF = time.time()
        BF['sol'], BF['cost'], BF['rtime'] = solve_tsp_brute_force(nodes_array)
        BF['ttime'] = timedelta(seconds = int(time.time()-start_BF)) if int(time.time()-start_BF) > 0 \
            else time.time()-start_BF

        BF['fixsol'], BF['fixcost'], BF['response'] = [],[],[]

        print(now() +" [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + f"] Bruteforce completed ")

        write_TSP_csv(df, BF)


    ### D-WAVE
    if DWave:
        print(now() +" [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC + "] Start computing D-Wave response ... ")
        QA = dict()
        QA['type'] = 'D-Wave'
        start_QA = time.time()
        QA['response'] = solve_tsp_annealer(qubo,1000)
        QA['rtime'] = timedelta(seconds = int(time.time()-start_QA)) if int(time.time()-start_QA) > 0 \
            else time.time()-start_QA

        QA['sol'] = binary_state_to_points_order(QA['response'])
        QA['cost'] = round(calculate_cost(tsp_matrix,QA['sol']),2)

        QA['fixsol'] = list(fix_solution(QA['response'], True))
        QA['fixcost'] = round(calculate_cost(tsp_matrix,QA['fixsol']),2)

        QA['ttime'] = timedelta(seconds = int(time.time()-start_QA)) if int(time.time()-start_QA) > 0 \
            else time.time()-start_QA
        print(now() +" [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] D-Wave response computed")

        write_TSP_csv(df, QA)
    
    
    ### HYBRID
    if Hybrid:
        print(now() +" [" + Colors.BOLD + Colors.OKBLUE + "LOG" + Colors.ENDC + "] Start computing Hybrid response ... ")
        HY = dict()
        HY['type'] = 'Hybrid'
        start_HY = time.time()
        HY['response'] = solve_tsp_hybrid(qubo)
        HY['rtime'] = timedelta(seconds = int(time.time()-start_QA)) if int(time.time()-start_QA) > 0 \
            else time.time()-start_QA

        HY['sol'] = binary_state_to_points_order(HY['response'])
        HY['cost'] = round(calculate_cost(tsp_matrix,HY['sol']),2)

        HY['fixsol'] = list(fix_solution(HY['response'], True))
        HY['fixcost'] = round(calculate_cost(tsp_matrix,HY['fixsol']),2)

        HY['ttime'] = timedelta(seconds = int(time.time()-start_HY)) if int(time.time()-start_HY) > 0 \
            else time.time()-start_HY
        print(now() +" [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] Hybrid response computed")
        write_TSP_csv(df, HY)
        
    print("\n\t" + Colors.BOLD + Colors.HEADER + "   TSP PROBLEM SOLVER END" + Colors.ENDC)
    
    return tsp_matrix, qubo
