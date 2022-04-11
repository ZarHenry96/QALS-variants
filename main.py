#!/usr/local/bin/python3

import argparse
import json
import pandas as pd
import os
import datetime
import time
from qals import utils, qals, tsp_utils
from qals.colors import Colors
from os import listdir, makedirs
from os.path import isfile, join
import sys
import numpy as np
import csv
import random

np.set_printoptions(threshold=sys.maxsize)


def log_write(tpe, var):
    return "[" + Colors.BOLD + str(tpe) + Colors.ENDC + "]\t" + str(var) + "\n"


def csv_write(csv_file, row):
    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def select_qap_problem():
    qap_files = [f for f in listdir("QAP/") if isfile(join("QAP/", f))]

    for i, element in enumerate(qap_files):
        print(f"\tWrite {i} for the problem {element.rsplit('.')[0]}")
    
    problem = int(input("Which problem do you want to solve? "))
    filepath = "QAP/"+qap_files[problem]

    return filepath, qap_files[problem].rsplit('.')[0]


def convert_to_numpy_Q(qubo_dict, qubo_size):
    Q = np.zeros((qubo_size, qubo_size))
    for x, y in qubo_dict.keys():
        Q[x][y] = qubo_dict[x, y]

    return Q


def main(config):
    makedirs(config['root_out_dir'], exist_ok=True)
    random.seed(config['random_seed'])
    qals_config = config['qals_params']

    # Select (or load) the problem to run
    print("\t\t" + Colors.BOLD + Colors.WARNING + "  BUILDING PROBLEM..." + Colors.ENDC)
    if config['problem'] is not None and config['problem'] != '':
        problem = config['problem']
        print(problem)
    else:
        problem = input(Colors.OKCYAN + "Which problem would you like to run? (NPP, QAP, TSP)  " + Colors.ENDC)
        config['problem'] = problem.upper()

    npp, qap, tsp = False, False, False
    if problem.lower() == "npp":
        npp = True
    elif problem.lower() == "qap":
        qap = True
    elif problem.lower() == "tsp":
        tsp = True
    else:
        print("[" + Colors.ERROR + "ERROR" + Colors.ENDC + "] string " + Colors.BOLD + problem + Colors.ENDC
              + " is not valid", file=sys.stderr)
        exit(0)

    # Select (or load) the problem parameters and build the QUBO matrix
    out_dir = None
    if npp:  # NPP problem
        n, max_value = None, None
        if len(config['problem_params']) != 0:
            n = config['problem_params']['n'] if 'n' in config['problem_params'] else n
            max_value = config['problem_params']['max_value'] if 'max_value' in config['problem_params'] else max_value

        if n is None:
            n = int(input("Insert n (number of values): "))
            while n <= 0:
                n = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid n" + Colors.ENDC
                              + "] Insert n (number of values): "))
            config['problem_params']['n'] = n
        if max_value is None:
            max_value = int(input("Insert the upper limit of the generation interval: "))
            while max_value <= 0:
                max_value = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid value" + Colors.ENDC
                                      + "] Insert the upper limit of the generation interval: "))
            config['problem_params']['max_value'] = max_value

        S = utils.generate_S(n, max_value)
        Q, c = utils.generate_NPP_QUBO_problem(S)

        out_dir = os.path.join(config['root_out_dir'], 'NPP', f'n_{n}_range_{max_value}',
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        csv_log_file = os.path.join(out_dir, f'npp_{n}_{max_value}_log.csv')

    elif qap:  # QAP problem
        filepath, problem_name = None, None
        if len(config['problem_params']) != 0:
            if 'qap_filename' in config['problem_params'] and \
                    os.path.exists(os.path.join('QAP', config['problem_params']['qap_filename'])):
                filepath = os.path.join('QAP', config['problem_params']['qap_filename'])
                problem_name = config['problem_params']['qap_filename'].rsplit('.')[0]

        if filepath is None or problem_name is None:
            filepath, problem_name = select_qap_problem()
            config['problem_params']['qap_filename'] = f'{problem_name}.txt'

        Q, penalty, n, y = utils.generate_QAP_QUBO_problem(filepath)

        out_dir = os.path.join(config['root_out_dir'], 'QAP', f'{problem_name}',
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        csv_log_file = os.path.join(out_dir, f'qap_{problem_name}_log.csv')

    elif tsp:  # TSP problem
        n = None
        if len(config['problem_params']) != 0:
            n = config['problem_params']['n'] if 'n' in config['problem_params'] else n

        if n is None or (n <= 0 or n > 12):
            n = int(input("Insert n (number of cities, allowed range [0, 11]): "))
            while n <= 0 or n > 12:
                n = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid n" + Colors.ENDC
                              + "] Insert n (number of cities, allowed range [0, 11]): "))
            config['problem_params']['n'] = n
        qubo_size = n ** 2

        out_dir = os.path.join(config['root_out_dir'], 'TSP', f'n_{n}',
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        csv_log_file = os.path.join(out_dir, f'tsp_{n}_log.csv')
        csv_write(csv_file=csv_log_file, row=["i", "f'", "f*", "p", "e", "d", "lambda", "z'", "z*"])

        out_df = pd.DataFrame(
            columns=["Solution", "Cost", "Fixed solution", "Fixed cost", "Response time", "Total time", "Response"],
            index=['Bruteforce', 'D-Wave', 'Hybrid', 'QALS']
        )
        tsp_matrix, qubo_dict = tsp_utils.generate_and_solve_TSP(n, os.path.join(out_dir, f'tsp_{n}_data.csv'), out_df,
                                                                 bruteforce=False, d_wave=False, hybrid=False)
        Q = convert_to_numpy_Q(qubo_dict, qubo_size)

    # Save the experiment configuration
    with open(os.path.join(out_dir, 'config.json'), 'w') as json_config:
        json.dump(config, json_config, ensure_ascii=False, indent=4)

    print("\t\t" + Colors.BOLD + Colors.OKGREEN + "   PROBLEM BUILDED" + Colors.ENDC + "\n\n\t\t" +
          Colors.BOLD + Colors.OKGREEN + "   START ALGORITHM" + Colors.ENDC + "\n")
    if npp:
        print("[" + Colors.BOLD + Colors.OKCYAN + "S" + Colors.ENDC + f"] {S}")

    # Solve the problem using QALS
    start_time = time.time()
    z, r_time = qals.run(d_min=qals_config['d_min'], eta=qals_config['eta'], i_max=qals_config['i_max'],
                         k=qals_config['k'], lambda_zero=qals_config['lambda_zero'], n=n if npp or qap else qubo_size,
                         N=qals_config['N'], N_max=qals_config['N_max'], p_delta=qals_config['p_delta'],
                         q=qals_config['q'], Q=Q, topology=config['topology'], csv_log_file=csv_log_file,
                         tabu_type=config['tabu_type'], simulation=config['simulation'])
    min_z = qals.function_f(Q, z).item()
    total_timedelta = datetime.timedelta(seconds=int(time.time() - start_time))

    # Prepare the output string and files
    print("\t\t\t" + Colors.BOLD + Colors.OKGREEN + "RESULTS" + Colors.ENDC + "\n")
    string = str()
    if n < 16:
        string += log_write("Z", z)
    else:
        string += log_write("Z", "Too big to print, see "+out_dir+"_solution.csv for the complete result")
    string += log_write("fQ", round(min_z, 2))

    if npp:
        diff_squared = (c**2 + 4*min_z)
        string += log_write("c", c) + log_write("C", c**2) + log_write("DIFF", round(diff_squared, 2)) + \
            log_write("diff", np.sqrt(diff_squared))

        solution_file = os.path.join(out_dir, f'npp_{n}_{max_value}_solution.csv')
        csv_write(csv_file=solution_file, row=["c", "c**2", "diff**2", "diff", "S", "z", "Q"])
        csv_write(csv_file=solution_file, row=[c, c ** 2, diff_squared, np.sqrt(diff_squared), S, z,
                                                         Q if n < 5 else "too big"])
    elif qap:
        string += log_write("y", y) + log_write("Penalty", penalty) + log_write("Difference", round(y+min_z, 2))
        solution_file = os.path.join(out_dir, f'qap_{problem_name}_solution.csv')
        csv_write(csv_file=solution_file, row=["problem", "y", "penalty", "difference (y+minimum)", "z", "Q"])
        csv_write(csv_file=solution_file, row=[problem_name, y, penalty, y + min_z, np.atleast_2d(z).T, Q])

    elif tsp:
        out_dict = dict()
        out_dict['type'] = 'QALS'
        out_dict['response'] = z

        res = np.split(z, n)
        valid = True

        fix_sol = list()
        for split in res:
            if np.count_nonzero(split == 1) != 1:
                valid = False
            where = str(np.where(split == 1))
            if str(np.where(split == 1)) in fix_sol:
                valid = False
            else:
                fix_sol.append(where)

        if not valid:
            string += "[" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC + "] Result is not valid.\n"
            out_dict['fixsol'] = list(tsp_utils.get_TSP_solution(z, refinement=True))
            string += "[" + Colors.BOLD + Colors.WARNING + "VALID" + Colors.ENDC + "] Refinement occurred \n"
        else:
            out_dict['fixsol'] = []
        out_dict['fixcost'] = round(tsp_utils.calculate_cost(tsp_matrix, out_dict['fixsol']), 2)

        out_dict['sol'] = tsp_utils.binary_state_to_points_order(z)
        out_dict['cost'] = tsp_utils.calculate_cost(tsp_matrix, out_dict['sol'])

        out_dict['rtime'] = r_time
        out_dict['ttime'] = total_timedelta

        tsp_utils.add_TSP_info_to_df(out_df, out_dict)
        out_df.to_csv(os.path.join(out_dir, f'tsp_{n}_solution.csv'))
    
    print(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running experiments on the KNN-classifier pipeline')
    parser.add_argument('config_file', metavar='config_file', type=str, nargs='?', default=None,
                        help='file (.json) containing the configuration for the experiment')
    args = parser.parse_args()

    # system('cls' if name == 'nt' else 'clear')

    with open(args.config_file) as cf:
        config = json.load(cf)

    main(config)
