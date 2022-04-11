#!/usr/bin/env python3
import time
import numpy as np
from qals.utils import generate_chimera_topology, generate_pegasus_topology
from qals.solvers import annealer, hybrid, stub_annealer
from dwave.system.samplers import DWaveSampler
import datetime
import neal
import sys
import csv
import random
from qals.colors import Colors
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod import ising_to_qubo

np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)


def function_f(Q, x):
    return np.matmul(np.matmul(x, Q), np.atleast_2d(x).T)


def get_active(sampler, n):
    nodes = dict()
    tmp = list(sampler.nodelist)
    nodelist = list()
    for i in range(n):
        try:
            nodelist.append(tmp[i])
        except IndexError:
            input(f"Error when reaching {i}-th element of tmp {len(tmp)}")

    for i in nodelist:
        nodes[i] = list()

    for node_1, node_2 in sampler.edgelist:
        if node_1 in nodelist and node_2 in nodelist:
            nodes[node_1].append(node_2)
            nodes[node_2].append(node_1)

    if (len(nodes) != n):
        i = 1
        while (len(nodes) != n):
            nodes[tmp[n+i]] = list()

    return nodes


def make_decision(probability):
    return random.random() < probability


def random_shuffle(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    random.shuffle(values)

    return dict(zip(keys, values))


def fill(m, perm, _n):
    n = len(perm)
    if (n != _n):
        n = _n

    filled = np.zeros(n, dtype=int)
    for i in range(n):
        if i in m.keys():
            filled[i] = perm[m[i]]
        else:
            filled[i] = perm[i]

    return filled


def invert(perm, _n):
    n = len(perm)
    if(n != _n):
        n = _n

    inverse = np.zeros(n, dtype=int)
    for i in range(n):
        inverse[perm[i]] = i

    return inverse


def g(Q, A, oldperm, p, simulation):
    n = len(Q)

    m = dict()
    for i in range(n):
        if make_decision(p):
            m[i] = i

    m = random_shuffle(m)
    perm = fill(m, oldperm, n)
    inverse = invert(perm, n)
    
    Theta = dict()
    if (simulation):
        for row, col in A:
            k = inverse[row]
            l = inverse[col]
            Theta[row, col] = Q[k][l]
    else:
        support = dict(zip(A.keys(), np.arange(n))) 
        for key in list(A.keys()):
            k = inverse[support[key]]
            Theta[key, key] = Q[k][k]
            for elem in A[key]:
                l = inverse[support[elem]]
                Theta[key, elem] = Q[k][l]
              
    return Theta, perm


def map_back(z, perm):
    n = len(z)
    inverted = invert(perm, n)

    z_ret = np.zeros(n, dtype=int)

    for i in range(n):
        z_ret[i] = int(z[inverted[i]])

    return z_ret


def h(vector, p):
    n = len(vector)

    for i in range(n):
        if make_decision(p):
            vector[i] = int((vector[i] + 1) % 2)

    return vector


def to_ising(z):
    return np.where(z == 0, -1, 1)


def add_to_tabu(S, z_prime, n, tabu_type):
    if tabu_type == 'binary':
        S = S + np.outer(z_prime, z_prime) - np.identity(n) + np.diagflat(z_prime)
    elif tabu_type == 'spin':
        z_prime_spin = to_ising(z_prime)
        S = S + np.outer(z_prime_spin, z_prime_spin) - np.identity(n) + np.diagflat(z_prime_spin)
    elif tabu_type == 'binary_no_diag':
        S = S + np.outer(z_prime, z_prime) - np.identity(n)
    elif tabu_type == 'spin_no_diag':
        z_prime_spin = to_ising(z_prime)
        S = S + np.outer(z_prime_spin, z_prime_spin) - np.identity(n)
    elif tabu_type == 'hopfield_like':
        z_prime_spin = to_ising(z_prime)
        S = S + np.outer(z_prime_spin, z_prime_spin) - np.identity(n)

    return S


def sum_Q_and_tabu(Q, S, lambda_value, n, tabu_type):
    Q_prime = None
    if tabu_type in ['binary', 'binary_no_diag', 'hopfield_like']:
        Q_prime = np.add(Q, (np.multiply(lambda_value, S)))
    elif tabu_type in ['spin', 'spin_no_diag']:
        # Compute linear (h) and quadratic (J) coefficients
        bqm = BinaryQuadraticModel.from_qubo(S)
        h, J = bqm.linear, bqm.quadratic

        # Convert Ising {-1,+1} formulation into QUBO {0,1}
        S_binary_dict, offset = ising_to_qubo(h, J)
        S_binary = np.zeros(shape=(n, n))
        for (i, j) in S_binary_dict.keys():
            S_binary[i][j] = S_binary_dict[i, j]

        # Sum as usual
        Q_prime = np.add(Q, (np.multiply(lambda_value, S_binary)))
    else:
        print('Execution modality not supported!', file=sys.stderr)
        exit(0)

    return Q_prime


def csv_write(csv_file, row):
    with open(csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def run(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, Q, topology, csv_log_file, tabu_type, simulation):
    try:
        if not simulation:
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER
                  + "Started Algorithm in Quantum Mode" + Colors.ENDC)

            sampler = DWaveSampler({'topology__type': topology})
            csv_log_file.replace("TSP_", "TSP_QA_")

            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER
                  + f"Using {topology} Topology \n" + Colors.ENDC)

            A = get_active(sampler, n)
        else:
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.OKCYAN
                  + "Started Algorithm in Simulation Mode (simulated annealing sampler on embedded QUBO matrix)"
                  + Colors.ENDC)

            sampler = neal.SimulatedAnnealingSampler()
            csv_log_file.replace("TSP_", "TSP_SA_")

            if topology == 'chimera':
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.OKCYAN
                      + "Using Chimera Topology \n" + Colors.ENDC)

                if n <= 2048:
                    try:
                        A = generate_chimera_topology(n)
                    except:
                        print(now() + " [" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC
                              + "] " + f"the required topology could not be generated",
                              file=sys.stderr)
                        exit(0)
                else:
                    print(now() + " [" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC
                          + "] " + f"the number of QUBO variables ({n}) is larger than the topology size (2048)",
                          file=sys.stderr)
                    exit(0)
            else:
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER
                      + "Using Pegasus Topology \n" + Colors.ENDC)

                A = generate_pegasus_topology(n)

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA IN" + Colors.ENDC + "] d_min = " + str(d_min)
              + " - eta = " + str(eta) + " - i_max = " + str(i_max) + " - k = " + str(k) + " - lambda_0 = "
              + str(lambda_zero) + " - n = " + str(n) + " - N = " + str(N) + " - N_max = " + str(N_max)
              + " - p_delta = " + str(p_delta) + " - q = " + str(q) + "\n")

        p = 1
        Theta_one, m_one = g(Q, A, np.arange(n), p, simulation)
        Theta_two, m_two = g(Q, A, np.arange(n), p, simulation)

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC + "] Working on z1...", end=' ')
        start_time = time.time()
        z_one = map_back(annealer(Theta_one, sampler, k), m_one)
        timedelta_z_one = datetime.timedelta(seconds=(time.time()-start_time))
        print("Ended in " + str(timedelta_z_one) + "\n" + now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN"
              + Colors.ENDC + "] Working on z2...", end=' ')
        start_time = time.time()
        z_two = map_back(annealer(Theta_two, sampler, k), m_two)
        timedelta_z_two = datetime.timedelta(seconds=(time.time()-start_time))
        print("Ended in "+str(timedelta_z_two)+"\n")

        f_one = function_f(Q, z_one).item()
        f_two = function_f(Q, z_two).item()

        if (f_one < f_two):
            z_star = z_one
            f_star = f_one
            m_star = m_one
            z_prime = z_two
        else:
            z_star = z_two
            f_star = f_two
            m_star = m_two
            z_prime = z_one

        S = np.zeros(shape=(n, n))
        if (f_one != f_two):
            S = add_to_tabu(S, z_prime, n, tabu_type)
    except KeyboardInterrupt:
        exit("\n\n[" + Colors.BOLD + Colors.OKGREEN + "KeyboardInterrupt" + Colors.ENDC + "] Closing program...")

    e = 0
    d = 0
    i = 1
    lamda_value = lambda_zero
    total_time = 0
    
    while True:
        print(f"-------------------------------------------------------------------------------------------------------"
              f"--------")
        if total_time:
            string = str(datetime.timedelta(seconds=((total_time/i) * (i_max - i))))
        else:
            string = "Not yet available"
        
        print(now() +" [" + Colors.BOLD + Colors.OKGREEN + "PRG" + Colors.ENDC +
              f"] Cycle {i}/{i_max} -- {round((((i - 1) / i_max) * 100), 2)}% -- ETA {string}")

        try:
            Q_prime = sum_Q_and_tabu(Q, S, lamda_value, n, tabu_type)
            
            if (i % N == 0):
                p = p - ((p - p_delta)*eta)

            Theta_prime, m = g(Q_prime, A, m_star, p, simulation)
            
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC + "] Working on z'...", end=' ')
            start_time = time.time()
            z_prime = map_back(annealer(Theta_prime, sampler, k), m)
            timedelta_z_prime = datetime.timedelta(seconds=(time.time()-start_time))
            print("Ended in "+str(timedelta_z_prime))

            if make_decision(q):
                z_prime = h(z_prime, p)

            if (z_prime != z_star).any():
                f_prime = function_f(Q, z_prime).item()
                
                if (f_prime < f_star):
                    z_prime, z_star = z_star, z_prime
                    f_star = f_prime
                    m_star = m
                    e = 0
                    d = 0
                    S = add_to_tabu(S, z_prime, n, tabu_type)
                else:
                    d = d + 1
                    if make_decision((p-p_delta)**(f_prime-f_star)):
                        z_prime, z_star = z_star, z_prime
                        f_star = f_prime
                        m_star = m
                        e = 0
                lamda_value = min(lambda_zero, (lambda_zero/(2+(i-1)-e)))
            else:
                e = e + 1
            
            iteration_timedelta = datetime.timedelta(seconds=(time.time()-start_time))

            try:
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC
                      + f"] f_prime = {round(f_prime, 2)}, f_star = {round(f_star, 2)}, p = {p}, e = {e}, d = {d} "
                        f"and lambda = {round(lamda_value, 5)}\n" + now() + " [" + Colors.BOLD + Colors.OKGREEN
                      + "DATA" + Colors.ENDC + f"] Took {iteration_timedelta} in total")
                csv_write(csv_file=csv_log_file, row=[i, f_prime, f_star, p, e, d, lamda_value, z_prime, z_star])
            except UnboundLocalError:
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC +
                      f" No variations on f and z. p = {p}, e = {e}, d = {d} and lambda = {round(lamda_value, 5)}\n"
                      + now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC
                      + f"] Took {iteration_timedelta} in total")
                csv_write(csv_file=csv_log_file, row=[i, "null", f_star, p, e, d, lamda_value, "null", z_star])
            
            total_time = total_time + (time.time() - start_time)

            print(f"---------------------------------------------------------------------------------------------------"
                  f"------------\n")
            if (i == i_max) or ((e + d >= N_max) and (d < d_min)):
                if i != i_max:
                    print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] Exited at cycle "
                          + str(i) + "/" + str(i_max) + " due to convergence.")
                else:
                    print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "END" + Colors.ENDC + "] Exited at cycle "
                          + str(i) + "/" + str(i_max) + "\n")
                break
            
            i = i + 1
        except KeyboardInterrupt:
            break

    total_timedelta = datetime.timedelta(seconds=total_time)
    if i != 1:
        avg_time = datetime.timedelta(seconds=int(total_time/(i-1)))
    else:
        avg_time = datetime.timedelta(seconds=int(total_time))
    
    print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "TIME" + Colors.ENDC + "] Average time for iteration: "
          + str(avg_time) + "\n" + now() + " [" + Colors.BOLD + Colors.OKGREEN + "TIME" + Colors.ENDC + "] Total time: "
          + str(total_timedelta) + "\n")

    return np.atleast_2d(np.atleast_2d(z_star).T).T[0], avg_time

