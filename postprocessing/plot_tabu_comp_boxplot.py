import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

from scipy.stats import wilcoxon


def print_values_and_stats(results, target_value, qals_paper_value, stats):
    max_key_length = max([len(key) for key in results.keys()] +
                         [len('Target value') if target_value is not None else 0,
                          len('QALS paper value') if qals_paper_value is not None else 0])

    print('Results:')
    for key, value in results.items():
        print('    {}:{}{}'.format(key, (max_key_length - len(key) + 1) * ' ', value))
    if target_value is not None:
        key = 'Target value'
        print('    {}:{}{}'.format(key, (max_key_length - len(key) + 1) * ' ', target_value))
    if qals_paper_value is not None:
        key = 'QALS paper value'
        print('    {}:{}{}'.format(key, (max_key_length - len(key) + 1) * ' ', qals_paper_value))

    if stats:
        print('Statistical differences (p-values):')
        if len({len(i) for i in results.values()}) == 1:
            keys_list = list(results.keys())
            keys_pairs = [(a, b) for idx, a in enumerate(keys_list) for b in keys_list[idx + 1:]]
            for key_1, key_2 in keys_pairs:
                if np.any(np.array(results[key_1]) - np.array(results[key_2])):
                    statistic, p_value = wilcoxon(results[key_1], results[key_2])
                else:
                    statistic, p_value = None, 1
                print('    {} - {}:{}{}'.format(key_1, key_2, (2 * max_key_length - len(key_1) - len(key_2) + 1) * ' ',
                                                p_value))
        else:
            print('    The number of values is not the same for all boxplots!')


def plot_boxplot(boxes_data, x_ticks_labels, target_value, qals_paper_value, x_label, y_label, title, y_limits,
                 out_file):
    matplotlib.rcParams['figure.dpi'] = 300

    width, height = 10, 10
    plt.figure(figsize=(width, height))

    plt.boxplot(boxes_data)
    plt.xticks(np.arange(1, len(boxes_data) + 1), x_ticks_labels)

    legend = False

    if target_value is not None:
        legend = True
        plt.hlines(target_value, 0.5, len(boxes_data) + 0.5, label='Target value', linestyles='--', colors='black')

    if qals_paper_value is not None:
        legend = True
        plt.hlines(qals_paper_value, 0.5, len(boxes_data) + 0.5, label='QALS paper value', linestyles='--',
                   colors='red')

    if legend:
        plt.legend(loc='upper right')

    plt.xlabel(x_label, fontsize=13, labelpad=16)
    plt.ylabel(y_label, fontsize=13, labelpad=16)
    plt.title(title, fontsize=15, pad=18)

    if y_limits is not None and len(y_limits) == 2:
        plt.ylim(y_limits[0], y_limits[1])

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main(root_res_dir, solution_key, target_value, qals_paper_value, x_label, y_label, title, y_limits, out_filename,
         verbose, stats):
    results = {}

    tabu_types_and_dirs = sorted([
        (d, os.path.join(root_res_dir, d))
        for d in os.listdir(root_res_dir)
        if os.path.isdir(os.path.join(root_res_dir, d))
    ])

    for tabu_type, tabu_dir in tabu_types_and_dirs:
        tabu_type_runs_results = []

        for run_dirname in sorted(os.listdir(tabu_dir)):
            run_dir = os.path.join(tabu_dir, run_dirname)
            if os.path.isdir(run_dir):
                solution_file = [
                   os.path.join(run_dir, f)
                   for f in os.listdir(run_dir)
                   if os.path.isfile(os.path.join(run_dir, f)) and re.match('.*_solution.csv', f)
                ]
                assert len(solution_file) == 1

                solution_df = pd.read_csv(solution_file[0])
                tabu_type_runs_results.append(solution_df.loc[0, solution_key])

        results[tabu_type] = tabu_type_runs_results

    if verbose:
        print_values_and_stats(results, target_value, qals_paper_value, stats)

    plot_boxplot(list(results.values()), list(results.keys()), target_value, qals_paper_value, x_label, y_label, title,
                 y_limits, os.path.join(root_res_dir, out_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for plotting the results of QALS experiments with different '
                                                 'tabu types.')
    parser.add_argument('--root-res-dir', metavar='root_res_dir', type=str, nargs='?', default=None, help='directory '
                        'containing the results folders related to the various tabu types (e.g. binary, spin...).')
    parser.add_argument('--solution-key', metavar='solution_key', type=str, action='store', nargs='?', default=None,
                        help='name of the property representing the solution quality inside the csv solution files.')
    parser.add_argument('--target-value', metavar='target_value', type=float, action='store', nargs='?', default=None,
                        help='target objective value.')
    parser.add_argument('--qals-paper-value', metavar='qals_paper_value', type=float, action='store', nargs='?',
                        default=None, help='objective value obtained in the run for the paper.')
    parser.add_argument('--x-label', metavar='x_label', type=str, action='store', nargs='?', default='',
                        help='label of the x axis.')
    parser.add_argument('--y-label', metavar='y_label', type=str, action='store', nargs='?', default='',
                        help='label of the y axis.')
    parser.add_argument('--title', metavar='title', type=str, action='store', nargs='?', default='',
                        help='chart title.')
    parser.add_argument('--y-limits', metavar='y_limits', type=float, action='store', nargs='+', default=None,
                        help='y limits for the plot (they must be 2, otherwise they are ignored).')
    parser.add_argument('--out-filename', metavar='out-filename', type=str, action='store', default='boxplot.pdf',
                        help='output filename (it will be saved inside the root-res-dir.')
    parser.add_argument('--verbose', dest='verbose', action='store_const', const=True, default=False,
                        help='print the boxplots values on the terminal.')
    parser.add_argument('--stats', dest='stats', action='store_const', const=True, default=False,
                        help='print the statistical differences (p-values for paired data) on the terminal; the '
                             '\'verbose\' flag must be enabled and the number of values per boxplot must be the same.')
    args = parser.parse_args()

    if args.root_res_dir is not None:
        main(args.root_res_dir, args.solution_key, args.target_value, args.qals_paper_value, args.x_label, args.y_label,
             args.title, args.y_limits, args.out_filename, args.verbose, args.stats)
