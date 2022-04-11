import random
import time


def annealer(theta, sampler, k, print_time=False):
    start_time = None
    if print_time:
        start_time = time.time()
        
    response = sampler.sample_qubo(theta, num_reads=k) 
    
    if print_time:
        print(f"Time: {time.time()-start_time}")
    
    return list(response.first.sample.values())


def hybrid(theta, sampler):
    response = sampler.sample_qubo(theta)

    return list(response.first.sample.values())


def stub_solver(n):
    return [random.randint(0, 1) for _ in range(0, n)]
