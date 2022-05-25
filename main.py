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

from qals import qals_algorithm
from qals.utils import Colors, csv_write

from problems_utils.npp import load_npp_params, load_numbers, extract_range_from_filepath, generate_and_save_numbers, \
    build_NPP_QUBO_problem
from problems_utils.qap import load_qap_params, build_QAP_QUBO_problem
from problems_utils.tsp import load_tsp_params, load_nodes, generate_and_save_nodes, build_TSP_QUBO_problem, \
    refine_TSP_solution_and_format_output, add_TSP_info_to_out_df, solve_TSP


np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)


def add_to_log_string(variable, value):
    padding = 5 + max(11, len(variable)) - len(str(variable))
    return "[" + Colors.BOLD + str(variable) + Colors.ENDC + "]" + " "*padding + str(value) + "\n"


def print_dict(d, level=0, list_on_levels=False):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            print('\t'*level + k+':')
            print_dict(v, level+1, list_on_levels)
        elif isinstance(v, list) and list_on_levels:
            print('\t' * level + '{}: ['.format(k))
            for element in v:
                print('\t' * (level+1) + '{}'.format(element))
            print('\t' * level + ']')
        else:
            print('\t'*level + '{}: {}'.format(k, v))


def main(config):
    # Set the seed and create the output directory
    rng_seeds = random.Random(config['random_seed'])
    problem_generation_seed, qals_execution_seed, extra_seed = \
        rng_seeds.randint(0, 1000000000), rng_seeds.randint(0, 1000000000), rng_seeds.randint(0, 1000000000)

    os.makedirs(config['root_out_dir'], exist_ok=True)

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
    print("\n\t" + Colors.BOLD + Colors.WARNING + "         BUILDING PROBLEM..." + Colors.ENDC)
    if npp:  # NPP problem
        data_filepath, num_values, max_value = load_npp_params(config)

        if data_filepath is not None:
            S = load_numbers(data_filepath)
            num_values, max_value = len(S), max(S)
            max_value = extract_range_from_filepath(data_filepath, max_value)

        out_dir = os.path.join(config['root_out_dir'], 'NPP', f'num_values_{num_values}_range_{max_value}',
                               '{}{}'.format(config['tabu_type'], '_sim' if config['simulation'] else ''),
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        out_files_prefix = os.path.join(out_dir, f'npp_{num_values}_{max_value}')

        data_file_copy_path = f'{out_files_prefix}_data.csv'
        if data_filepath is None:
            S = generate_and_save_numbers(num_values, max_value, data_file_copy_path, problem_generation_seed)
        else:
            shutil.copy2(data_filepath, data_file_copy_path)

        qubo_size = num_values
        Q, c = build_NPP_QUBO_problem(S)

    elif qap:  # QAP problem
        data_filepath, problem_name = load_qap_params(config)

        out_dir = os.path.join(config['root_out_dir'], 'QAP', f'{problem_name}',
                               '{}{}'.format(config['tabu_type'], '_sim' if config['simulation'] else ''),
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        out_files_prefix = os.path.join(out_dir, f'qap_{problem_name}')

        shutil.copy2(data_filepath, f'{out_files_prefix}_data.txt')

        Q, qubo_size, penalty, offset = build_QAP_QUBO_problem(data_filepath)

    elif tsp:  # TSP problem
        data_filepath, num_nodes, bruteforce, dwave, hybrid = load_tsp_params(config)

        if data_filepath is not None:
            nodes = load_nodes(data_filepath)
            num_nodes = len(nodes)

        out_dir = os.path.join(config['root_out_dir'], 'TSP', f'nodes_{num_nodes}',
                               '{}{}'.format(config['tabu_type'], '_sim' if config['simulation'] else ''),
                               datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        os.makedirs(out_dir, exist_ok=True)
        out_files_prefix = os.path.join(out_dir,  f'tsp_{num_nodes}')

        data_file_copy_path = f'{out_files_prefix}_data.csv'
        if data_filepath is None:
            nodes = generate_and_save_nodes(num_nodes, data_file_copy_path, problem_generation_seed)
        else:
            shutil.copy2(data_filepath, data_file_copy_path)

        qubo_size = num_nodes ** 2
        qubo_problem_dict, Q, tsp_matrix, penalty_coefficients = build_TSP_QUBO_problem(nodes, qubo_size)

    print("\t\t" + Colors.BOLD + Colors.OKGREEN + "   PROBLEM BUILT" + Colors.ENDC + "\n\n")

    # Output files
    qubo_matrix_csv_file = f'{out_files_prefix}_qubo_matrix.csv'
    solver_info_csv_file = f'{out_files_prefix}_solver_info.csv'
    adj_matrix_json_file = f'{out_files_prefix}_adj_matrix.json'
    qals_csv_log_file = f'{out_files_prefix}_qals_log.csv'
    tabu_csv_log_file = f'{out_files_prefix}_tabu_log.csv'
    solution_csv_file = f'{out_files_prefix}_solution.csv'

    # Save the experiment configuration and the QUBO matrix
    print("\t" + Colors.BOLD + Colors.OKGREEN + "      EXPERIMENT CONFIGURATION" + Colors.ENDC)
    config['res_dir'] = out_dir
    print_dict(config)
    del config['res_dir']
    with open(os.path.join(out_dir, 'config.json'), 'w') as json_config:
        json.dump(config, json_config, ensure_ascii=False, indent=4)
    pd.DataFrame(Q).to_csv(qubo_matrix_csv_file, index=False, header=False)

    print("\n\n\t\t" + Colors.BOLD + Colors.OKGREEN + "  START ALGORITHM" + Colors.ENDC + "\n")
    # Solve the problem using QALS
    start_time = time.time()
    qals_config = config['qals_params']
    z_star, convergence, iterations_num, avg_iteration_time = \
        qals_algorithm.run(d_min=qals_config['d_min'], eta=qals_config['eta'], i_max=qals_config['i_max'],
                           k=qals_config['k'], lambda_zero=qals_config['lambda_zero'], n=qubo_size,
                           N=qals_config['N'], N_max=qals_config['N_max'], p_delta=qals_config['p_delta'],
                           q=qals_config['q'], Q=Q, topology=config['topology'], random_seed=qals_execution_seed,
                           solver_info_csv_file=solver_info_csv_file, adj_matrix_json_file=adj_matrix_json_file,
                           qals_csv_log_file=qals_csv_log_file, tabu_csv_log_file=tabu_csv_log_file,
                           tabu_type=config['tabu_type'], simulation=config['simulation'])
    min_value_found = qals_algorithm.function_f(Q, z_star)
    total_timedelta = datetime.timedelta(seconds=(time.time() - start_time))

    # Prepare the output string and files
    print("\n\t\t\t" + Colors.BOLD + Colors.OKGREEN + "RESULTS" + Colors.ENDC + "\n")
    log_string = str()
    log_string += add_to_log_string("z*", "Look into " + out_dir)
    log_string += add_to_log_string("f_Q value", round(min_value_found, 2))

    if npp:
        diff_squared = (c**2 + 4*min_value_found)
        log_string += add_to_log_string("c", c) + add_to_log_string("c**2", c ** 2) + \
                      add_to_log_string("diff**2", round(diff_squared, 2)) + \
                      add_to_log_string("diff", np.sqrt(diff_squared))

        csv_write(csv_file=solution_csv_file, row=["c", "c**2", "diff**2", "diff", "z*", "f_Q(z*)",
                                                   "convergence", "iterations", "avg. iteration time", "total time"])
        csv_write(csv_file=solution_csv_file, row=[c, c ** 2, diff_squared, np.sqrt(diff_squared),
                                                   z_star, min_value_found, convergence, iterations_num,
                                                   avg_iteration_time, total_timedelta])
    elif qap:
        log_string += add_to_log_string("penalty", penalty) + add_to_log_string("offset", offset) + \
                      add_to_log_string("obj. f val.", round(min_value_found + offset, 2))
        csv_write(csv_file=solution_csv_file, row=["problem", "penalty", "offset", "f_Q(z*)",
                                                   "objective function value (f_Q(z*) + offset)", "z*", "convergence",
                                                   "iterations", "avg. iteration time", "total time"])
        csv_write(csv_file=solution_csv_file, row=[problem_name, penalty, offset, min_value_found,
                                                   min_value_found + offset, z_star, convergence, iterations_num,
                                                   avg_iteration_time, total_timedelta])

    elif tsp:
        df_index = [
            index for index, method_selected in zip(['QALS', 'Bruteforce', 'D-Wave', 'Hybrid'],
                                                    [True, bruteforce, dwave, hybrid])
            if method_selected
        ]
        output_df = pd.DataFrame(
            columns=["solution", "cost", "refinement", "avg. iteration time", "total time (w/o refinement)",
                     "z*", "f_Q(z*)", "refined(z*)", "f_Q(refined(z*))", "convergence", "iterations", "A penalty",
                     "B penalty"],
            index=df_index
        )
        rng_refinement = random.Random(extra_seed)
        qals_output, log_string = \
            refine_TSP_solution_and_format_output('QALS', z_star, num_nodes, Q, log_string, tsp_matrix,
                                                  avg_iteration_time, total_timedelta, min_value_found, convergence,
                                                  iterations_num, penalty_coefficients, rng_refinement)
        add_TSP_info_to_out_df(output_df, qals_output)

        solve_TSP(tsp_matrix, qubo_problem_dict, Q, output_df, penalty_coefficients, rng_refinement,
                  bruteforce=bruteforce, d_wave=dwave, hybrid=hybrid)

        output_df.to_csv(solution_csv_file)
    
    print(log_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running experiments on the KNN-classifier pipeline')
    parser.add_argument('config_file', metavar='config_file', type=str, nargs='?', default=None,
                        help='file (.json) containing the configuration for the experiment')
    args = parser.parse_args()

    with open(args.config_file) as cf:
        config = json.load(cf)

    main(config)
