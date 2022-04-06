#!/usr/local/bin/python3

import argparse
import json
import pandas as pd
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


# def write(dir, string):
#     file = open(dir, 'a')
#     file.write(string+'\n')
#     file.close()


def csv_write(DIR, l):
    with open(DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(l)


def select_qap_problem():
    qap_files = [f for f in listdir("QAP/") if isfile(join("QAP/", f))]

    i = 0
    elements = list()
    for element in qap_files:
        elements.append(element)
        element = element[:-4]
        print(f"Write {i} for the problem {element}")
        i += 1
    
    problem = int(input("Which problem do you want to select? "))
    DIR = "QAP/"+qap_files[problem]

    return DIR, qap_files[problem]


def generate_npp_file(_n: int, out_dir):
    not_ok = True
    i = 0
    max_range = 100000
    _dir = "NPP_"+str(_n)+"_"+str(max_range)
    while(not_ok):
        try:
            with open(out_dir+"/"+_dir.replace("NPP", "NPP_LOG")+".csv", "r") as file:
                pass
            max_range = int(max_range/10)
            if (max_range < 10):
                exit("File output terminati")
            _dir = "NPP_"+str(_n)+"_"+str(max_range)
            i += 1
        except FileNotFoundError:
            not_ok = False
        
    DIR = out_dir+"/"+_dir

    return DIR, max_range


def generate_qap_file(name, out_dir):
    not_ok = True
    i = 0
    _dir = "QAP_" + str(name)
    while (not_ok):
        try:
            with open(out_dir+"/" + _dir + ".csv", "r") as _:
                pass
            i += 1
            _dir = "QAP_" + str(name) + "_" + str(i)
        except FileNotFoundError:
            not_ok = False

    DIR = out_dir+"/"+_dir

    return DIR


def generate_tsp_file(n: int, out_dir):
    not_ok = True
    i = 1
    _dir = "TSP_"+str(n)+"_"+str(i)
    while(not_ok):
        try:
            with open(out_dir+"/"+_dir.replace("TSP", "TSP_LOG")+".csv", "r") as _:
                pass
            i += 1
            _dir = "TSP_"+str(n)+"_"+str(i)
        except FileNotFoundError:
            not_ok = False
        
    DIR = out_dir+"/"+_dir

    return DIR


def convert_to_numpy_Q(qubo, n):
    Q = np.zeros((n, n))
    for x, y in qubo.keys():
        Q[x][y] = qubo[x, y]

    return Q


def main(config):
    makedirs(config['out_dir'], exist_ok=True)
    random.seed(config['random_seed'])

    print("\t\t" + Colors.BOLD + Colors.WARNING + "  BUILDING PROBLEM..." + Colors.ENDC)
    pr = input(Colors.OKCYAN + "Which problem would you like to run? (NPP, QAP, TSP)  " + Colors.ENDC)
    if pr == "NPP":
        NPP = True
        QAP = False
        TSP = False
    elif pr == "QAP":
        NPP = False
        QAP = True
        TSP = False
    elif pr == "TSP":
        NPP = False
        QAP = False
        TSP = True
    else:
        NPP, QAP, TSP = False, False, False
        print("[" + Colors.FAIL + "ERROR" + Colors.ENDC + "] string " + Colors.BOLD + pr + Colors.ENDC + " is not valid, exiting...")
        exit(2)

    if NPP:
        nn = int(input("Insert n: "))
        while nn <= 0:
            nn = int(input("[" + Colors.FAIL + Colors.BOLD + "Invalid n" + Colors.ENDC + "] Insert n: "))
        _DIR, max_range = generate_npp_file(nn, config['out_dir'])
        S = utils.generate_S(nn, max_range)
        _Q, c = utils.generate_NPP_QUBO_problem(S)
        log_DIR = _DIR.replace("NPP", "NPP_LOG") + ".csv"
    elif QAP:
        _dir, name = select_qap_problem()
        _Q, penalty, nn, y = utils.generate_QAP_QUBO_problem(_dir)
        name = name.replace(".txt", "")
        _DIR = generate_qap_file(name, config['out_dir'])
        log_DIR = _DIR.replace("QAP", "QAP_LOG") + ".csv"
    elif TSP:
        nn = int(input("Insert n: "))
        while nn <= 0 or nn > 12:
            nn = int(input("[" + Colors.FAIL + Colors.BOLD + "Invalid n" + Colors.ENDC + "] Insert n: "))
        _DIR = generate_tsp_file(nn, config['out_dir'])
        log_DIR = _DIR.replace("TSP", "TSP_LOG") + ".csv"
        csv_write(DIR=log_DIR, l=["i", "f'", "f*", "p", "e", "d", "lambda", "z'", "z*"])
        df = pd.DataFrame(
            columns=["Solution", "Cost", "Fixed solution", "Fixed cost", "Response time", "Total time", "Response"],
            index=['Bruteforce', 'D-Wave', 'Hybrid', 'QALS']
        )
        tsp_matrix, qubo = tsp_utils.tsp(nn, _DIR + "_solution.csv", _DIR[:-1] + "DATA.csv", df)
        _Q = convert_to_numpy_Q(qubo, nn**2)
    
    print("\t\t" + Colors.BOLD + Colors.OKGREEN + "   PROBLEM BUILDED" + Colors.ENDC + "\n\n\t\t" + Colors.BOLD + Colors.OKGREEN +
          "   START ALGORITHM" + Colors.ENDC + "\n")
    
    if NPP:
        print("[" + Colors.BOLD + Colors.OKCYAN + "S" + Colors.ENDC + f"] {S}")

    start = time.time()
    z, r_time = qals.run(d_min=70, eta=0.01, i_max=10, k=1, lambda_zero=3 / 2,
                         n=nn if NPP or QAP else nn ** 2, N=10, N_max=100, p_delta=0.1, q=0.2,
                         topology='pegasus', Q=_Q, log_DIR=log_DIR, sim=False)
    conv = datetime.timedelta(seconds=int(time.time() - start))

    min_z = qals.function_f(_Q, z).item()
    print("\t\t\t" + Colors.BOLD + Colors.OKGREEN + "RESULTS" + Colors.ENDC + "\n")
    string = str()
    if nn < 16:
        string += log_write("Z", z)
    else:
        string += log_write("Z", "Too big to print, see "+_DIR+"_solution.csv for the complete result")
    string += log_write("fQ", round(min_z, 2))

    if NPP:
        diff2 = (c**2 + 4*min_z)
        string += log_write("c", c) + log_write("C", c**2) + log_write("DIFF", round(diff2, 2)) + \
            log_write("diff", np.sqrt(diff2))
        csv_write(DIR=_DIR+"_solution.csv", l=["c", "c**2", "diff**2", "diff", "S", "z", "Q"])
        csv_write(DIR=_DIR+"_solution.csv", l=[c, c**2, diff2, np.sqrt(diff2), S, z, _Q if nn < 5 else "too big"])
    elif QAP:
        string += log_write("y", y) + log_write("Penalty", penalty) + log_write("Difference", round(y+min_z, 2))
        csv_write(DIR=_DIR+"_solution.csv", l=["problem", "y", "penalty", "difference (y+minimum)", "z", "Q"])
        csv_write(DIR=_DIR+"_solution.csv", l=[name, y, penalty, y+min_z, np.atleast_2d(z).T, _Q])
    else:
        DW = dict()
        DW['type'] = 'QALS'
        DW['response'] = z
        res = np.split(z, nn)
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
            string += "[" + Colors.BOLD + Colors.FAIL + "ERROR" + Colors.ENDC + "] Result is not valid.\n"
            DW['fixsol'] = list(tsp_utils.fix_solution(z, True))
            string += "[" + Colors.BOLD + Colors.WARNING + "VALID" + Colors.ENDC + "] Validation occurred \n"
        else:
            DW['fixsol'] = []

        DW['fixcost'] = round(tsp_utils.calculate_cost(tsp_matrix, DW['fixsol']), 2)
        DW['sol'] = tsp_utils.binary_state_to_points_order(z)
        DW['cost'] = tsp_utils.calculate_cost(tsp_matrix, DW['sol'])
        DW['rtime'] = r_time
        DW['ttime'] = conv

        tsp_utils.write_TSP_csv(df, DW)

        df.to_csv(_DIR+"_solution.csv")
    
    print(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running experiments on the KNN-classifier pipeline')
    parser.add_argument('config_file', metavar='config_file', type=str, nargs='?', default=None,
                        help='file (.json) containing the configuration for the experiment')
    args = parser.parse_args()

    system('cls' if name == 'nt' else 'clear')

    with open(args.config_file) as cf:
        config = json.load(cf)

    main(config)
