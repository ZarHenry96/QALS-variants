import os


def select_input_data(problem):
    input_data_dir = os.path.join('input_data', problem)
    input_data_files = sorted([
        f for f in os.listdir(input_data_dir) if os.path.isfile(os.path.join(input_data_dir, f))
    ])

    for i, element in enumerate(input_data_files):
        print(f"    Write {i} for the problem {element.rsplit('.')[0]}")

    problem = int(input("Which problem do you want to solve? "))
    filepath = os.path.join(input_data_dir, input_data_files[problem])

    return filepath, input_data_files[problem].rsplit('.')[0]
