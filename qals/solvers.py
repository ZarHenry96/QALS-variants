import neal

from dwave.system.samplers import DWaveSampler

from qals.utils import Colors, now


def get_annealing_sampler(simulation, topology):
    if simulation:
        sampler = neal.SimulatedAnnealingSampler()
        string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.OKCYAN \
                 + "Algorithm Started in Simulation Modality (ideal annealer topology & simulated annealing sampler)" \
                 + Colors.ENDC
    else:
        sampler = DWaveSampler({'topology__type': topology})
        string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER \
                 + "Algorithm Started in Quantum Modality" + Colors.ENDC

    return sampler, string


def annealing(theta, annealing_sampler, k):
    response = annealing_sampler.sample_qubo(theta, num_reads=k)
    
    return list(response.first.sample.values())


def hybrid(theta, hybrid_sampler):
    response = hybrid_sampler.sample_qubo(theta)

    return list(response.first.sample.values())


def stub_solver(solution_length, rng):
    return [rng.randint(0, 1) for _ in range(0, solution_length)]
