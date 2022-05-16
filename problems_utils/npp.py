import numpy as np
import os
import random
import pandas as pd

from problems_utils.utils import select_input_data
from qals.utils import Colors


def delete_extra_params_keys(config, num_values, max_value):
    if num_values is not None:
        del config['problem_params']['num_values']
    if max_value is not None:
        del config['problem_params']['max_value']

    return None, None


def load_npp_params(config):
    data_filepath, num_values, max_value = None, None, None
    if len(config['problem_params']) != 0:
        data_filepath = config['problem_params']['npp_data_file'] \
            if 'npp_data_file' in config['problem_params'] and \
               os.path.exists(config['problem_params']['npp_data_file']) \
            else data_filepath
        num_values = config['problem_params']['num_values'] if 'num_values' in config['problem_params'] else num_values
        max_value = config['problem_params']['max_value'] if 'max_value' in config['problem_params'] else max_value

    if data_filepath is not None:
        num_values, max_value = delete_extra_params_keys(config, num_values, max_value)
    else:
        if (num_values is None or num_values <= 0) or (max_value is None or max_value <= 0):
            num_values, max_value = delete_extra_params_keys(config, num_values, max_value)

            input_data_selection = input("Do you want to use an existent NPP problem file? (y/n) ")
            while input_data_selection not in ['y', 'n']:
                input_data_selection = input("Do you want to use an existent NPP problem file? (y/n) ")

            if input_data_selection == 'y':
                data_filepath, _ = select_input_data('npp')
                config['problem_params']['npp_data_file'] = data_filepath
            else:
                num_values = int(input("Insert the desired number of values: "))
                while num_values <= 0:
                    num_values = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid number of values" + Colors.ENDC
                                           + "] Insert the desired number of values: "))
                config['problem_params']['num_values'] = num_values

                max_value = int(input("Insert the upper limit of the generation interval: "))
                while max_value <= 0:
                    max_value = int(input("[" + Colors.ERROR + Colors.BOLD + "Invalid value" + Colors.ENDC
                                          + "] Insert the upper limit of the generation interval: "))
                config['problem_params']['max_value'] = max_value

    return data_filepath, num_values, max_value


def generate_and_save_numbers(num_values, max_value, data_file):
    vect = [0 for _ in range(num_values)]
    for index in range(num_values):
        vect[index] = random.randint(0, max_value)

    # Save to csv file
    pd.DataFrame([vect], dtype=int).to_csv(data_file, index=False, header=False)

    return vect


def load_numbers(data_filepath):
    df = pd.read_csv(data_filepath, header=None)

    return list(df.to_numpy()[0])


def build_NPP_QUBO_problem(numbers):
    num_values = len(numbers)
    c = 0
    for i in range(num_values):
        c += numbers[i]

    QUBO = np.zeros((num_values, num_values))
    col_max = 0
    col = 0
    for row in range(num_values):
        col_max += 1
        while col < col_max:
            if row == col:
                QUBO[row][col] = numbers[row] * (numbers[row] - c)
            else:
                QUBO[row][col] = numbers[row] * numbers[col]
                QUBO[col][row] = QUBO[row][col]
            col += 1
        col = 0

    return QUBO, c
