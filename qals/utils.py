import csv

from datetime import datetime


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def now():
    return datetime.now().strftime("%H:%M:%S")


def csv_write(csv_file, row):
    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def np_vector_to_string(vector):
    return '[' + ' '.join([str(x) for x in vector]) + ']'


def tabu_to_string(S):
    string = '[' + ' '.join(['['+' '.join([str(x) for x in row])+']' for row in S]) + ']'
    return string
