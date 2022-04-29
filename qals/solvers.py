import neal
import random
import time

from dwave.system.samplers import DWaveSampler

from qals.colors import Colors
from qals.utils import now


def get_annealing_sampler(simulation, topology):
    if simulation:
        sampler = neal.SimulatedAnnealingSampler()
        string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.OKCYAN \
                 + "Algorithm Started in Simulation Modality (ideal annealer topology & simulated annealing sampler)" \
                 + Colors.ENDC
    else:
        sampler = DWaveSampler({'topology__type': topology})
        string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER \
                 + "Algorithm Started in Quantum Modality" + Colors.ENDC

    return sampler, string


def annealing(theta, annealing_sampler, k, print_time=False):
    start_time = None
    if print_time:
        start_time = time.time()
        
    response = annealing_sampler.sample_qubo(theta, num_reads=k)
    
    if print_time:
        print(f"Time: {time.time()-start_time}")
    
    return list(response.first.sample.values())


def hybrid(theta, hybrid_sampler):
    response = hybrid_sampler.sample_qubo(theta)

    return list(response.first.sample.values())


def stub_solver(solution_length):
    return [random.randint(0, 1) for _ in range(0, solution_length)]
