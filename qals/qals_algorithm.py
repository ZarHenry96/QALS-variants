import datetime
import json
import numpy as np
import random
import time
import sys

from dimod import ising_to_qubo

from qals.solvers import get_annealing_sampler, annealing
from qals.topology import get_adj_matrix
from qals.utils import Colors, now, csv_write, np_vector_to_string, tabu_to_string


def function_f(Q, x):
    return np.matmul(np.matmul(x, Q), np.atleast_2d(x).T).item()


def make_decision(probability, rng):
    return rng.random() < probability


def shuffle_dict(dictionary, rng):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    rng.shuffle(values)

    return dict(zip(keys, values))


def g(old_perm, p, rng):
    n = len(old_perm)

    assoc_map = dict()
    for i in range(n):
        if make_decision(p, rng):
            assoc_map[i] = i
    assoc_map = shuffle_dict(assoc_map, rng)

    perm = np.zeros(n, dtype=int)
    for i in range(n):
        if i in assoc_map.keys():
            perm[i] = old_perm[assoc_map[i]]
        else:
            perm[i] = old_perm[i]

    return perm


def invert(perm):
    n = len(perm)

    inverse = np.zeros(n, dtype=int)
    for i in range(n):
        inverse[perm[i]] = i

    return inverse


def generate_weight_matrix(Q, inverse, A):
    Theta = dict()

    node_pos_dict = dict(zip(A.keys(), np.arange(len(Q))))
    for key in list(A.keys()):
        k = inverse[node_pos_dict[key]]
        Theta[key, key] = Q[k][k]
        for elem in A[key]:
            l = inverse[node_pos_dict[elem]]
            Theta[key, elem] = Q[k][l]
              
    return Theta


def map_back(z, inverse):
    n = len(z)

    z_ret = np.zeros(n, dtype=float)
    for i in range(n):
        z_ret[i] = int(z[inverse[i]])

    return z_ret


def h(vector, p, rng):
    n = len(vector)

    for i in range(n):
        if make_decision(p, rng):
            vector[i] = int((vector[i] + 1) % 2)

    return vector


def binary_vector_to_spin(z):
    return np.where(z == 0, -1, 1)


def spin_tabu_to_binary(S_spin, n):
    # Compute linear (h_values) and quadratic (J) coefficients
    h_values, J = dict(), dict()
    for i in range(S_spin.shape[0]):
        for j in range(S_spin.shape[1]):
            if i == j:
                h_values[i] = S_spin[i][i]
            else:
                J[i, j] = S_spin[i][j]

    # Convert Ising {-1,+1} formulation into QUBO {0,1}
    S_binary_dict, offset = ising_to_qubo(h_values, J)
    S_binary = np.zeros(shape=(n, n))
    for (i, j) in S_binary_dict.keys():
        S_binary[i][j] = S_binary_dict[i, j]

    return S_binary


def add_to_tabu(S, z_prime, n, tabu_type):
    if tabu_type == 'binary':
        S = S + np.outer(z_prime, z_prime) - np.identity(n, dtype=float) + np.diagflat(z_prime)
    elif tabu_type == 'spin':
        z_prime_spin = binary_vector_to_spin(z_prime)
        S = S + spin_tabu_to_binary(np.outer(z_prime_spin, z_prime_spin) - np.identity(n, dtype=float)
                                    + np.diagflat(z_prime_spin), n)
    elif tabu_type == 'binary_no_diag':
        S = S + np.outer(z_prime, z_prime) - np.diagflat(z_prime)
    elif tabu_type == 'spin_no_diag':
        z_prime_spin = binary_vector_to_spin(z_prime)
        S = S + spin_tabu_to_binary(np.outer(z_prime_spin, z_prime_spin) - np.identity(n, dtype=float), n)
    elif tabu_type == 'hopfield_like':
        z_prime_spin = binary_vector_to_spin(z_prime)
        S = S + np.outer(z_prime_spin, z_prime_spin) - np.identity(n, dtype=float)
    elif tabu_type == 'only_diag':
        z_prime_spin = binary_vector_to_spin(z_prime)
        S = S + spin_tabu_to_binary(np.diagflat(z_prime_spin), n)
    elif tabu_type == 'no_tabu':
        S = S
    else:
        print('Unknown tabu_type \'{}\'!'.format(tabu_type), file=sys.stderr)
        exit(0)

    return S


def run(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, Q, topology, random_seed, solver_info_csv_file,
        adj_matrix_json_file, qals_csv_log_file, tabu_csv_log_file, tabu_log_frequency, tabu_type, simulation):
    sampler, A, p, z_star, f_star, perm_star, S = None, None, None, None, None, None, None

    rng_seeds = random.Random(random_seed)
    rng_generic = random.Random(rng_seeds.randint(0, 1000000000))
    rng_subopt_accept = random.Random(rng_seeds.randint(0, 1000000000))
    rng_simulation = random.Random(rng_seeds.randint(0, 1000000000))

    try:
        sampler, out_string = get_annealing_sampler(simulation, topology)
        if not simulation:
            csv_write(csv_file=solver_info_csv_file, row=["name", "topology"])
            csv_write(csv_file=solver_info_csv_file, row=[sampler.solver.name, sampler.properties['topology']['type']])
        print(out_string)

        A, out_string = get_adj_matrix(simulation, topology, sampler, n)
        with open(adj_matrix_json_file, 'w') as jf:
            json.dump(A, jf, ensure_ascii=False, indent=4)
        print(out_string)

        csv_write(csv_file=qals_csv_log_file, row=["i (end)", "p", "perturb", "opt_accept", "subopt_accept",
                                                   "e", "d", "lambda", "f'", "f*", "non_perturbed_f", "z'", "z*",
                                                   "non_perturbed_z", "perm", "perm*"])
        csv_write(csv_file=tabu_csv_log_file, row=["i (end)", "S"])
        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA IN" + Colors.ENDC + "] d_min = " + str(d_min)
              + ", eta = " + str(eta) + ", i_max = " + str(i_max) + ", k = " + str(k) + ", lambda_0 = "
              + str(lambda_zero))
        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA IN" + Colors.ENDC + "] n = " + str(n) + ", N = "
              + str(N) + ", N_max = " + str(N_max) + ", p_delta = " + str(p_delta) + ", q = " + str(q) + "\n")

        p = 1

        perm_one = g(np.arange(n), p, rng_generic)
        inverse_one = invert(perm_one)
        Theta_one = generate_weight_matrix(Q, inverse_one, A)

        perm_two = g(np.arange(n), p, rng_generic)
        inverse_two = invert(perm_two)
        Theta_two = generate_weight_matrix(Q, inverse_two, A)

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC + "]  Working on z1...", end=' ')
        start_time = time.time()
        sim_seed = rng_simulation.randint(0, 1000000000) if simulation else None
        z_one = map_back(annealing(Theta_one, sampler, k, sim_seed=sim_seed), inverse_one)
        timedelta_z_one = datetime.timedelta(seconds=(time.time()-start_time))
        print("Ended in " + str(timedelta_z_one))

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC + "]  Working on z2...", end=' ')
        start_time = time.time()
        sim_seed = rng_simulation.randint(0, 1000000000) if simulation else None
        z_two = map_back(annealing(Theta_two, sampler, k, sim_seed=sim_seed), inverse_two)
        timedelta_z_two = datetime.timedelta(seconds=(time.time()-start_time))
        print("Ended in "+str(timedelta_z_two)+"\n")

        f_one = function_f(Q, z_one)
        f_two = function_f(Q, z_two)

        if f_one < f_two:
            z_star = z_one
            f_star = f_one
            perm_star = perm_one
            z_prime = z_two
            f_prime, perm = f_two, perm_two
        else:
            z_star = z_two
            f_star = f_two
            perm_star = perm_two
            z_prime = z_one
            f_prime, perm = f_one, perm_one

        S = np.zeros(shape=(n, n), dtype=float)
        if f_one != f_two:
            S = add_to_tabu(S, z_prime, n, tabu_type)

        csv_write(csv_file=qals_csv_log_file, row=[0, p, None, None, None, 0, 0, lambda_zero,
                                                   f_prime if (z_one != z_two).any() else None, f_star, None,
                                                   z_prime if (z_one != z_two).any() else None, z_star, None,
                                                   np_vector_to_string(perm), np_vector_to_string(perm_star)])
        csv_write(csv_file=tabu_csv_log_file, row=[0, tabu_to_string(S)])
    except KeyboardInterrupt:
        exit("\n\n[" + Colors.BOLD + Colors.OKGREEN + "KeyboardInterrupt" + Colors.ENDC + "] Exiting...")

    e = 0
    d = 0
    i = 0
    lambda_value = lambda_zero
    iterations_time = 0

    convergence = False
    try:
        while i != i_max and not ((e + d >= N_max) and (d < d_min)):
            print("-" * 116)
            iteration_start_time = time.time()
            if i != 0:
                string = str(datetime.timedelta(seconds=((iterations_time/i) * (i_max - i))))
            else:
                string = "Not yet available"
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "PRG" + Colors.ENDC +
                  f"]  Cycle {i+1}/{i_max} -- {round(((i / i_max) * 100), 2)}% done -- ETA {string}")

            Q_prime = np.add(Q, (np.multiply(lambda_value, S)))
            
            if i % N == 0:
                p = p - ((p - p_delta)*eta)

            perm = g(perm_star, p, rng_generic)
            inverse = invert(perm)
            Theta_prime = generate_weight_matrix(Q_prime, inverse, A)
            
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC + "]  Working on z'...", end=' ')
            start_time = time.time()
            sim_seed = rng_simulation.randint(0, 1000000000) if simulation else None
            z_prime = map_back(annealing(Theta_prime, sampler, k, sim_seed=sim_seed), inverse)
            timedelta_z_prime = datetime.timedelta(seconds=(time.time()-start_time))
            print("Ended in "+str(timedelta_z_prime))

            perturbation, non_perturbed_z, non_perturbed_f, optimal_acceptance, suboptimal_acceptance = \
                False, None, None, False, False

            if make_decision(q, rng_generic):
                perturbation = True
                non_perturbed_z = np.copy(z_prime)
                non_perturbed_f = function_f(Q, z_prime)
                z_prime = h(z_prime, p, rng_generic)

            if (z_prime != z_star).any():
                f_prime = function_f(Q, z_prime)
                
                if f_prime < f_star:
                    optimal_acceptance = True
                    z_prime, z_star = z_star, z_prime
                    f_star = f_prime
                    perm_star = perm
                    e = 0
                    d = 0
                    S = add_to_tabu(S, z_prime, n, tabu_type)
                else:
                    d = d + 1
                    if make_decision((p-p_delta)**(f_prime-f_star), rng_subopt_accept):
                        suboptimal_acceptance = True
                        z_prime, z_star = z_star, z_prime
                        f_star = f_prime
                        perm_star = perm
                        e = 0
                lambda_value = min(lambda_zero, (lambda_zero/(2+i-e)))
            else:
                f_prime = None
                e = e + 1

            i += 1

            iteration_timedelta = datetime.timedelta(seconds=(time.time() - iteration_start_time))
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC
                  + f"] f_star = {round(f_star, 2)}, p = {p}, lambda = {round(lambda_value, 5)}, e = {e}, and d = {d}\n"
                  + now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC + f"] "
                  + f"Took {iteration_timedelta} in total")
            csv_write(csv_file=qals_csv_log_file, row=[i, p, perturbation, optimal_acceptance, suboptimal_acceptance,
                                                       e, d, lambda_value, f_prime, f_star, non_perturbed_f,
                                                       z_prime if f_prime else None, z_star, non_perturbed_z,
                                                       np_vector_to_string(perm), np_vector_to_string(perm_star)])
            if i % tabu_log_frequency == 0:
                csv_write(csv_file=tabu_csv_log_file, row=[i, tabu_to_string(S)])

            iterations_time = iterations_time + (time.time() - iteration_start_time)
            print("-" * 116 + "\n")

        if i % tabu_log_frequency != 0:
            csv_write(csv_file=tabu_csv_log_file, row=[i, tabu_to_string(S)])

        convergence = (i != i_max)
        print(now() + " [" + Colors.BOLD + (Colors.OKGREEN if convergence else Colors.OKBLUE) + " END" +
              Colors.ENDC + "] Exited at cycle " + str(i) + "/" + str(i_max) +
              (" due to convergence." if convergence else " -- 100% done\n"))
    except KeyboardInterrupt:
        pass

    iterations_timedelta = datetime.timedelta(seconds=iterations_time)
    if i != 0:
        avg_iteration_time = datetime.timedelta(seconds=(iterations_time/i))
    else:
        avg_iteration_time = datetime.timedelta(seconds=iterations_time)
    
    print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "TIME" + Colors.ENDC + "] Average iteration time: "
          + str(avg_iteration_time) + "\n" + now() + " [" + Colors.BOLD + Colors.OKGREEN + "TIME" + Colors.ENDC
          + "] Total iterations time: " + str(iterations_timedelta) + "\n")

    return z_star, convergence, i, avg_iteration_time
