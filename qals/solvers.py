from dwave.system.samplers import DWaveSampler
from neal import SimulatedAnnealingSampler

from qals.utils import Colors, now


def get_annealing_sampler(simulation, topology):
    if simulation:
        sampler = SimulatedAnnealingSampler()
        string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.OKCYAN \
                 + "Algorithm Started in Simulation Modality (ideal topology & simulated annealing sampler)" \
                 + Colors.ENDC
    else:
        sampler = DWaveSampler({'topology__type': topology})
        string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER \
                 + "Algorithm Started in Quantum Modality" + Colors.ENDC

    return sampler, string


def annealing(theta, annealing_sampler, k, sim_seed=None):
    response = None
    if isinstance(annealing_sampler, DWaveSampler):
        response = annealing_sampler.sample_qubo(theta, num_reads=k)
    elif isinstance(annealing_sampler, SimulatedAnnealingSampler):
        response = annealing_sampler.sample_qubo(theta, num_reads=k, seed=sim_seed)
    else:
        print('The annealing sampler belongs to an unknown class')
        exit(0)
    
    return list(response.first.sample.values())


def hybrid(theta, hybrid_sampler):
    response = hybrid_sampler.sample_qubo(theta)

    return list(response.first.sample.values())


def stub_solver(solution_length, rng):
    return [rng.randint(0, 1) for _ in range(0, solution_length)]
