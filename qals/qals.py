#!/usr/bin/env python3
import time
import numpy as np
from qals.utils import generate_chimera_topology, generate_pegasus_topology
from qals.solvers import annealer
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

    if(len(nodes) != n):
        i = 1
        while(len(nodes) != n):
            nodes[tmp[n+i]] = list()

    return nodes


def make_decision(probability):
    return random.random() < probability


def random_shuffle(a):
    keys = list(a.keys())
    values = list(a.values())
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


def inverse(perm, _n):
    n = len(perm)
    if(n != _n):
        n = _n
    inverted = np.zeros(n, dtype=int)
    for i in range(n):
        inverted[perm[i]] = i

    return inverted


def g(Q, A, oldperm, p, sim):
    n = len(Q)
    m = dict()
    for i in range(n):
        if make_decision(p):
            m[i] = i

    m = random_shuffle(m)
    perm = fill(m, oldperm, n)
    inversed = inverse(perm, n)
    
    Theta = dict()
    if (sim):
        for row, col in A:
            k = inversed[row]
            l = inversed[col]
            Theta[row, col] = Q[k][l]
    else:
        support = dict(zip(A.keys(), np.arange(n))) 
        for key in list(A.keys()):
            k = inversed[support[key]]
            Theta[key, key] = Q[k][k]
            for elem in A[key]:
                l = inversed[support[elem]]
                Theta[key, elem] = Q[k][l]
              
    return Theta, perm


def map_back(z, perm):
    n = len(z)
    inverted = inverse(perm, n)

    z_ret = np.zeros(n, dtype=int)

    for i in range(n):
        z_ret[i] = int(z[inverted[i]])

    return z_ret


def h(vect, pr):
    n = len(vect)

    for i in range(n):
        if make_decision(pr):
            vect[i] = int((vect[i]+1) % 2)

    return vect


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


def csv_write(DIR, l):
    with open(DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(l)


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def run(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, Q, topology, log_DIR, tabu_type, sim):
    try:
        if (not sim):
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER
                  + "Started Algorithm in Quantum Mode" + Colors.ENDC)
            sampler = DWaveSampler({'topology__type': topology})
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER
                  + "Using Pegasus Topology \n" + Colors.ENDC)
            A = get_active(sampler, n)
            log_DIR.replace("TSP_", "TSP_QA_")
        else:
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.OKCYAN
                  + "Started Algorithm in Simulation Mode (simulated annealing sampler)" + Colors.ENDC)
            sampler = neal.SimulatedAnnealingSampler()
            log_DIR.replace("TSP_", "TSP_SA_")
            if (topology == 'chimera'):
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.OKCYAN
                      + "Using Chimera Topology \n" + Colors.ENDC)
                if (n > 2048):
                    n = int(input(
                        now() + " [" + Colors.WARNING + Colors.BOLD + "WARNING" + Colors.ENDC
                        + f"] {n} inserted value is bigger than max topology size (2048), please insert a valid n "
                          f"or press any key to exit: "))
                try:
                    A = generate_chimera_topology(n)
                except:
                    exit()
            else:
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER
                      + "Using Pegasus Topology \n" + Colors.ENDC)
                A = generate_pegasus_topology(n)

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA IN" + Colors.ENDC + "] d_min = " + str(d_min)
              + " - eta = " + str(eta) + " - i_max = " + str(i_max) + " - k = " + str(k) + " - lambda_0 = "
              + str(lambda_zero) + " - n = " + str(n) + " - N = " + str(N) + " - N_max = " + str(N_max)
              + " - p_delta = " + str(p_delta) + " - q = " + str(q) + "\n")

        p = 1
        Theta_one, m_one = g(Q, A, np.arange(n), p, sim)
        Theta_two, m_two = g(Q, A, np.arange(n), p, sim)

        print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC + "] Working on z1...", end=' ')
        start = time.time()
        z_one = map_back(annealer(Theta_one, sampler, k), m_one)
        convert_1 = datetime.timedelta(seconds=(time.time()-start))
        print("Ended in " + str(convert_1) +"\n" + now() +" [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC
              + "] Working on z2...", end=' ')
        start = time.time()
        z_two = map_back(annealer(Theta_two, sampler, k), m_two)
        convert_2 = datetime.timedelta(seconds=(time.time()-start))
        print("Ended in "+str(convert_2)+"\n")

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
    lam = lambda_zero
    sum_time = 0
    
    while True:
        print(f"-------------------------------------------------------------------------------------------------------"
              f"--------")
        start_time = time.time()
        if sum_time:
            string = str(datetime.timedelta(seconds=((sum_time/i) * (i_max - i))))
        else:
            string = "Not yet available"
        
        print(now() +" [" + Colors.BOLD + Colors.OKGREEN + "PRG" + Colors.ENDC +
              f"] Cycle {i}/{i_max} -- {round((((i - 1) / i_max) * 100), 2)}% -- ETA {string}")

        try:
            Q_prime = sum_Q_and_tabu(Q, S, lam, n, tabu_type)
            
            if (i % N == 0):
                p = p - ((p - p_delta)*eta)

            Theta_prime, m = g(Q_prime, A, m_star, p, sim)
            
            print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "ANN" + Colors.ENDC + "] Working on z'...", end=' ')
            start = time.time()
            z_prime = map_back(annealer(Theta_prime, sampler, k), m)
            convert_z = datetime.timedelta(seconds=(time.time()-start))
            print("Ended in "+str(convert_z))

            if make_decision(q):
                z_prime = h(z_prime, p)

            if (z_prime != z_star).any() :
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
                lam = min(lambda_zero, (lambda_zero/(2+(i-1)-e)))
            else:
                e = e + 1
            
            converted = datetime.timedelta(seconds=(time.time()-start_time))

            try:
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC
                      + f"] f_prime = {round(f_prime, 2)}, f_star = {round(f_star, 2)}, p = {p}, e = {e}, d = {d} "
                        f"and lambda = {round(lam, 5)}\n" + now() + " [" + Colors.BOLD + Colors.OKGREEN
                      + "DATA" + Colors.ENDC + f"] Took {converted} in total")
                csv_write(DIR=log_DIR,l=[i, f_prime, f_star, p, e, d, lam, z_prime, z_star])
            except UnboundLocalError:
                print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC +
                      f" No variations on f and z. p = {p}, e = {e}, d = {d} and lambda = {round(lam, 5)}\n"
                      + now() + " [" + Colors.BOLD + Colors.OKGREEN + "DATA" + Colors.ENDC
                      + f"] Took {converted} in total")
                csv_write(DIR=log_DIR,l=[i, "null", f_star, p, e, d, lam, "null", z_star])
            
            sum_time = sum_time + (time.time() - start_time)

            print(f"---------------------------------------------------------------------------------------------------"
                  f"------------\n")
            if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
                if(i != i_max):
                    print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "END" + Colors.ENDC + "] Exited at cycle "
                          + str(i) + "/" + str(i_max) + " thanks to convergence.")
                else:
                    print(now() + " [" + Colors.BOLD + Colors.OKBLUE + "END" + Colors.ENDC + "] Exited at cycle "
                          + str(i) + "/" + str(i_max) + "\n")
                break
            
            i = i + 1
        except KeyboardInterrupt:
            break

    converted = datetime.timedelta(seconds=sum_time)
    if i != 1:
        conv = datetime.timedelta(seconds=int(sum_time/(i-1)))
    else:
        conv = datetime.timedelta(seconds=int(sum_time))
    
    print(now() + " [" + Colors.BOLD + Colors.OKGREEN + "TIME" + Colors.ENDC + "] Average time for iteration: "
          + str(conv) + "\n" + now() + " [" + Colors.BOLD + Colors.OKGREEN + "TIME" + Colors.ENDC + "] Total time: "
          + str(converted) + "\n")

    return np.atleast_2d(np.atleast_2d(z_star).T).T[0], conv

