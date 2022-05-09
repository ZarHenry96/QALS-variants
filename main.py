import argparse
import datetime
import json
import numpy as np
import os
import pandas as pd
import random
import shutil
import sys
import time

from os import listdir, makedirs
from os.path import isfile, join

from qals import qals, tsp_utils, utils
from qals.colors import Colors

np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)


def add_to_log_string(variable, value):
    padding = 5 + max(10, len(variable)) - len(str(variable))
    return "[" + Colors.BOLD + str(variable) + Colors.ENDC + "]" + " "*padding + str(value) + "\n"


def load_npp_params(config):
    data_filepath, num_values, max_value = None, None, None
    if len(config['problem_params']) != 0:
        data_filepath = config['problem_params']['npp_data_file'] \
            if 'npp_data_file' in config['problem_params'] and \
               os.path.exists(config['problem_params']['npp_data_file']) \
            else data_filepath
        num_values = config['problem_params']['num_values'] if 'num_values' in config['problem_params'] else num_values
        max_value = config['problem_params']['max_value'] if 'max_value' in config['problem_params'] else max_value

    if data_filepath is None:
        if num_values is None:
            num_values = int(input("Insert the desired number of values: "))
            while num_values <= 0:
                num_values = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid number of values" + Colors.ENDC
                                       + "] Insert the desired number of values: "))
            config['problem_params']['num_values'] = num_values
        if max_value is None:
            max_value = int(input("Insert the upper limit of the generation interval: "))
            while max_value <= 0:
                max_value = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid value" + Colors.ENDC
                                      + "] Insert the upper limit of the generation interval: "))
            config['problem_params']['max_value'] = max_value

    return data_filepath, num_values, max_value


def select_qap_problem():
    qap_files = sorted([f for f in listdir("QAP/") if isfile(join("QAP/", f))])

    for i, element in enumerate(qap_files):
        print(f"    Write {i} for the problem {element.rsplit('.')[0]}")

    problem = int(input("Which problem do you want to solve? "))
    filepath = "QAP/" + qap_files[problem]

    return filepath, qap_files[problem].rsplit('.')[0]


def load_qap_params(config):
    filepath, problem_name = None, None
    if len(config['problem_params']) != 0:
        if 'qap_data_file' in config['problem_params'] and \
                os.path.exists(config['problem_params']['qap_data_file']):
            filepath = config['problem_params']['qap_data_file']
            problem_name = os.path.basename(config['problem_params']['qap_data_file']).rsplit('.')[0]

    if filepath is None or problem_name is None:
        filepath, problem_name = select_qap_problem()
        config['problem_params']['qap_data_file'] = filepath

    return filepath, problem_name


def load_tsp_params(config):
    data_filepath, num_nodes = None, None
    if len(config['problem_params']) != 0:
        data_filepath = config['problem_params']['tsp_data_file'] \
            if 'tsp_data_file' in config['problem_params'] and \
               os.path.exists(config['problem_params']['tsp_data_file']) \
            else data_filepath
        num_nodes = config['problem_params']['num_nodes'] if 'num_nodes' in config['problem_params'] else num_nodes

    if data_filepath is None:
        if num_nodes is None or (num_nodes <= 0 or num_nodes > 12):
            num_nodes = int(input("Insert the desired number of nodes (cities), the allowed range is [0, 11]: "))
            while num_nodes <= 0 or num_nodes > 12:
                num_nodes = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid number of nodes" + Colors.ENDC
                                      + "] Insert the desired number of nodes (cities), the allowed range is [0, 11]: ")
                                )
            config['problem_params']['num_nodes'] = num_nodes

    bruteforce, dwave, hybrid = False, False, False
    if len(config['additional_params']) != 0:
        bruteforce = config['additional_params']['bruteforce'] if 'bruteforce' in config['additional_params']\
                        else bruteforce
        dwave = config['additional_params']['dwave'] if 'dwave' in config['additional_params'] else dwave
        hybrid = config['additional_params']['hybrid'] if 'hybrid' in config['additional_params'] else hybrid

    return data_filepath, num_nodes, bruteforce, dwave, hybrid


def main(config):
    # Set the seed and create the output directory
    random.seed(config['random_seed'])
    problem_generation_seed, qals_execution_seed = random.randint(0, 1000000000), random.randint(0, 1000000000)
    other_seeds = [random.randint(0, 1000000000) for _ in range(0, 3)]

    makedirs(config['root_out_dir'], exist_ok=True)

    # Select (or load) the problem to run
    if config['problem'] is not None and config['problem'] != '':
        problem = config['problem']
        print("\t\t\t " + Colors.OKCYAN + problem + Colors.ENDC + "\n")
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
        print("[" + Colors.ERROR + "ERROR" + Colors.ENDC + "] unsupported " + Colors.BOLD + problem + Colors.ENDC
              + f" problem '{problem}'", file=sys.stderr)
        exit(0)

    # Select (or load) the problem parameters and build the QUBO matrix
    random.seed(problem_generation_seed)
    print("\t\t" + Colors.BOLD + Colors.WARNING + "  BUILDING PROBLEM..." + Colors.ENDC)
    if npp:  # NPP problem
        data_filepath, num_values, max_value = load_npp_params(config)

        if data_filepath is not None:
            S = utils.load_numbers(data_filepath)
            num_values, max_value = len(S), max(S)

        out_dir = os.path.join(config['root_out_dir'], 'NPP',
                               f'num_values_{num_values}_range_{max_value}' + ('_sim' if config['simulation'] else ''),
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        out_files_prefix = os.path.join(out_dir, f'npp_{num_values}_{max_value}')

        data_file_copy_path = f'{out_files_prefix}_data.csv'
        if data_filepath is None:
            S = utils.generate_and_save_numbers(num_values, max_value, data_file_copy_path)
        else:
            shutil.copy2(data_filepath, data_file_copy_path)

        qubo_size = num_values
        Q, c = utils.build_NPP_QUBO_problem(S)

    elif qap:  # QAP problem
        data_filepath, problem_name = load_qap_params(config)

        out_dir = os.path.join(config['root_out_dir'], 'QAP',
                               f'{problem_name}' + ('_sim' if config['simulation'] else ''),
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        out_files_prefix = os.path.join(out_dir, f'qap_{problem_name}')

        shutil.copy2(data_filepath, f'{out_files_prefix}_data.txt')

        Q, penalty, qubo_size, y = utils.build_QAP_QUBO_problem(data_filepath)

    elif tsp:  # TSP problem
        data_filepath, num_nodes, bruteforce, dwave, hybrid = load_tsp_params(config)

        if data_filepath is not None:
            nodes = tsp_utils.load_nodes(data_filepath)
            num_nodes = len(nodes)

        out_dir = os.path.join(config['root_out_dir'], 'TSP',
                               f'nodes_{num_nodes}' + ('_sim' if config['simulation'] else ''),
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        out_files_prefix = os.path.join(out_dir,  f'tsp_{num_nodes}')

        data_file_copy_path = f'{out_files_prefix}_data.csv'
        if data_filepath is None:
            nodes = tsp_utils.generate_and_save_nodes(num_nodes, data_file_copy_path)
        else:
            shutil.copy2(data_filepath, data_file_copy_path)

        qubo_size = num_nodes ** 2
        qubo_problem, Q, tsp_matrix = tsp_utils.build_TSP_QUBO_problem(nodes, qubo_size)

    # Output files
    qubo_matrix_csv_file = f'{out_files_prefix}_qubo_matrix.csv'
    adj_matrix_json_file = f'{out_files_prefix}_adj_matrix.json'
    qals_csv_log_file = f'{out_files_prefix}_qals_log.csv'
    tabu_csv_log_file = f'{out_files_prefix}_tabu_log.csv'
    solution_csv_file = f'{out_files_prefix}_solution.csv'

    # Save the experiment configuration and the QUBO matrix
    with open(os.path.join(out_dir, 'config.json'), 'w') as json_config:
        json.dump(config, json_config, ensure_ascii=False, indent=4)
    pd.DataFrame(Q).to_csv(qubo_matrix_csv_file, index=False, header=False)

    print("\t\t" + Colors.BOLD + Colors.OKGREEN + "   PROBLEM BUILT" + Colors.ENDC + "\n\n\t\t" +
          Colors.BOLD + Colors.OKGREEN + "   START ALGORITHM" + Colors.ENDC + "\n")
    if npp:
        print("[" + Colors.BOLD + Colors.OKCYAN + "S" + Colors.ENDC + f"] {S}")

    # Solve the problem using QALS
    start_time = time.time()
    qals_config = config['qals_params']
    random.seed(qals_execution_seed)
    z_star, avg_response_time = \
        qals.run(d_min=qals_config['d_min'], eta=qals_config['eta'], i_max=qals_config['i_max'],
                 k=qals_config['k'], lambda_zero=qals_config['lambda_zero'], n=qubo_size,
                 N=qals_config['N'], N_max=qals_config['N_max'], p_delta=qals_config['p_delta'],
                 q=qals_config['q'], Q=Q, topology=config['topology'], adj_matrix_json_file=adj_matrix_json_file,
                 qals_csv_log_file=qals_csv_log_file, tabu_csv_log_file=tabu_csv_log_file,
                 tabu_type=config['tabu_type'], simulation=config['simulation'])
    min_value_found = qals.function_f(Q, z_star).item()
    total_timedelta = datetime.timedelta(seconds=int(time.time() - start_time))

    # Prepare the output string and files
    print("\t\t\t" + Colors.BOLD + Colors.OKGREEN + "RESULTS" + Colors.ENDC + "\n")
    log_string = str()
    if qubo_size < 16:
        log_string += add_to_log_string("z*", z_star)
    else:
        log_string += add_to_log_string("z*", "Too long to be printed, look into " +
                                              out_dir + " for the complete result")
    log_string += add_to_log_string("f_Q value", round(min_value_found, 2))

    if npp:
        diff_squared = (c**2 + 4*min_value_found)
        log_string += add_to_log_string("c", c) + add_to_log_string("c**2", c ** 2) + \
                      add_to_log_string("diff**2", round(diff_squared, 2)) + \
                      add_to_log_string("diff", np.sqrt(diff_squared))

        utils.csv_write(csv_file=solution_csv_file, row=["c", "c**2", "diff**2", "diff", "S", "z*", "f_Q(z*)"])
        utils.csv_write(csv_file=solution_csv_file, row=[c, c ** 2, diff_squared, np.sqrt(diff_squared),
                                                         utils.numpy_vector_to_string(np.array(S)), z_star,
                                                         min_value_found])
    elif qap:
        log_string += add_to_log_string("y", y) + add_to_log_string("Penalty", penalty) + \
                      add_to_log_string("Difference", round(y + min_value_found, 2))
        utils.csv_write(csv_file=solution_csv_file, row=["problem", "penalty", "y", "f_Q(z*)",
                                                         "difference (y+f_Q(z*))", "z*"])
        utils.csv_write(csv_file=solution_csv_file, row=[problem_name, penalty, y, min_value_found, y + min_value_found,
                                                         z_star])

    elif tsp:
        output_df = pd.DataFrame(
            columns=["Solution", "Cost", "Refinement", "Avg. response time", "Total time (w/o refinement)",
                     "z*", "f_Q(z*)"],
            index=['QALS', 'Bruteforce', 'D-Wave', 'Hybrid']
        )
        qals_output, log_string = \
            tsp_utils.refine_TSP_solution_and_format_output('QALS', z_star, num_nodes, log_string, tsp_matrix,
                                                            avg_response_time, total_timedelta, min_value_found)
        tsp_utils.add_TSP_info_to_out_df(output_df, qals_output)

        tsp_utils.solve_TSP(nodes, qubo_problem, tsp_matrix, Q, output_df, other_seeds,
                            bruteforce=bruteforce, d_wave=dwave, hybrid=hybrid)

        output_df.to_csv(solution_csv_file)
    
    print(log_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running experiments on the KNN-classifier pipeline')
    parser.add_argument('config_file', metavar='config_file', type=str, nargs='?', default=None,
                        help='file (.json) containing the configuration for the experiment')
    args = parser.parse_args()

    # system('cls' if name == 'nt' else 'clear')

    with open(args.config_file) as cf:
        config = json.load(cf)

    main(config)
