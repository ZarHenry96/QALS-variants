import time

def annealer(theta, sampler, k, print_time=False):
    if print_time:
        start = time.time()
        
    response = sampler.sample_qubo(theta, num_reads=k) 
    
    if print_time:
        print(f"Time: {time.time()-start}")
    
    return list(response.first.sample.values())


def hybrid(theta, sampler):
    response = sampler.sample_qubo(theta)

    return list(response.first.sample.values())
