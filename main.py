#!/usr/local/bin/python3

import argparse
import json
import pandas as pd
import os
import datetime
import time
from qals import utils, qals, tsp_utils
from qals.colors import Colors
from os import listdir, makedirs, system, name
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
    makedirs(config['out_dir'], exist_ok=True)
    random.seed(config['random_seed'])
    qals_config = config['qals_params']

    print("\t\t" + Colors.BOLD + Colors.WARNING + "  BUILDING PROBLEM..." + Colors.ENDC)
    pr = input(Colors.OKCYAN + "Which problem would you like to run? (NPP, QAP, TSP)  " + Colors.ENDC)
    if pr == "NPP":
        npp, qap, tsp = True, False, False
    elif pr == "QAP":
        npp, qap, tsp = False, True, False
    elif pr == "TSP":
        npp, qap, tsp = False, False, True
    else:
        npp, qap, tsp = False, False, False
        print("[" + Colors.ERROR + "ERROR" + Colors.ENDC + "] string " + Colors.BOLD + pr + Colors.ENDC
              + " is not valid", file=sys.stderr)
        exit(0)

    out_dir = None
    if npp:
        n = int(input("Insert n (number of values): "))
        while n <= 0:
            n = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid n" + Colors.ENDC
                          + "] Insert n (number of values): "))
        max_range = int(input("Insert the upper limit of the generation interval: "))
        while max_range <= 0:
            max_range = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid value" + Colors.ENDC
                                  + "] Insert the upper limit of the generation interval: "))

        S = utils.generate_S(n, max_range)
        Q, c = utils.generate_NPP_QUBO_problem(S)

        out_dir = os.path.join(config['out_dir'], 'NPP', f'n_{n}_range_{max_range}',
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        csv_log_file = os.path.join(out_dir, f'npp_{n}_{max_range}_log.csv')
    elif qap:
        filepath, problem_name = select_qap_problem()
        Q, penalty, n, y = utils.generate_QAP_QUBO_problem(filepath)

        out_dir = os.path.join(config['out_dir'], 'QAP', f'{problem_name}',
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        csv_log_file = os.path.join(out_dir, f'qap_{problem_name}_log.csv')
    elif tsp:
        n = int(input("Insert n (number of cities, allowed range [0, 11]): "))
        while n <= 0 or n > 12:
            n = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid n" + Colors.ENDC
                          + "] Insert n (number of cities, allowed range [0, 11]): "))
        qubo_size = n ** 2

        out_dir = os.path.join(config['out_dir'], 'TSP', f'n_{n}',
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        csv_log_file = os.path.join(out_dir, f'tsp_{n}_log.csv')
        csv_write(csv_file=csv_log_file, row=["i", "f'", "f*", "p", "e", "d", "lambda", "z'", "z*"])

        df = pd.DataFrame(
            columns=["Solution", "Cost", "Fixed solution", "Fixed cost", "Response time", "Total time", "Response"],
            index=['Bruteforce', 'D-Wave', 'Hybrid', 'QALS']
        )
        tsp_matrix, qubo_dict = tsp_utils.tsp(n, os.path.join(out_dir, f'tsp_{n}_data.csv'), df,
                                              d_wave=False, hybrid=False)
        Q = convert_to_numpy_Q(qubo_dict, qubo_size)

    with open(os.path.join(out_dir, 'config.json'), 'w') as json_config:
        json.dump(config, json_config, ensure_ascii=False, indent=4)
    
    print("\t\t" + Colors.BOLD + Colors.OKGREEN + "   PROBLEM BUILDED" + Colors.ENDC + "\n\n\t\t" +
          Colors.BOLD + Colors.OKGREEN + "   START ALGORITHM" + Colors.ENDC + "\n")
    if npp:
        print("[" + Colors.BOLD + Colors.OKCYAN + "S" + Colors.ENDC + f"] {S}")

    start_time = time.time()
    z, r_time = qals.run(d_min=qals_config['d_min'], eta=qals_config['eta'], i_max=qals_config['i_max'],
                         k=qals_config['k'], lambda_zero=qals_config['lambda_zero'], n=n if npp or qap else qubo_size,
                         N=qals_config['N'], N_max=qals_config['N_max'], p_delta=qals_config['p_delta'],
                         q=qals_config['q'], Q=Q, topology=config['topology'], csv_log_file=csv_log_file,
                         tabu_type=config['tabu_type'], simulation=config['simulation'])
    min_z = qals.function_f(Q, z).item()
    total_timedelta = datetime.timedelta(seconds=int(time.time() - start_time))

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

        solution_file = os.path.join(out_dir, f'npp_{n}_{max_range}_solution.csv')
        csv_write(csv_file=solution_file, row=["c", "c**2", "diff**2", "diff", "S", "z", "Q"])
        csv_write(csv_file=solution_file, row=[c, c ** 2, diff_squared, np.sqrt(diff_squared), S, z,
                                                         Q if n < 5 else "too big"])
    elif qap:
        string += log_write("y", y) + log_write("Penalty", penalty) + log_write("Difference", round(y+min_z, 2))
        solution_file = os.path.join(out_dir, f'qap_{problem_name}_solution.csv')
        csv_write(csv_file=solution_file, row=["problem", "y", "penalty", "difference (y+minimum)", "z", "Q"])
        csv_write(csv_file=solution_file, row=[problem_name, y, penalty, y + min_z, np.atleast_2d(z).T, Q])
    else:
        dw = dict()
        dw['type'] = 'QALS'
        dw['response'] = z

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
            dw['fixsol'] = list(tsp_utils.get_TSP_solution(z, refinement=True))
            string += "[" + Colors.BOLD + Colors.WARNING + "VALID" + Colors.ENDC + "] Refinement occurred \n"
        else:
            dw['fixsol'] = []
        dw['fixcost'] = round(tsp_utils.calculate_cost(tsp_matrix, dw['fixsol']), 2)

        dw['sol'] = tsp_utils.binary_state_to_points_order(z)
        dw['cost'] = tsp_utils.calculate_cost(tsp_matrix, dw['sol'])

        dw['rtime'] = r_time
        dw['ttime'] = total_timedelta

        tsp_utils.add_TSP_info_to_df(df, dw)
        df.to_csv(os.path.join(out_dir, f'tsp_{n}_solution.csv'))
    
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
