def annealer(theta, sampler, k, time=False):
    if time:
        start = time.time()
        
    response = sampler.sample_qubo(theta, num_reads=k) 
    
    if time:
        print(f"Time: {time.time()-start}")
    
    return list(response.first.sample.values())


def hybrid(theta, sampler):
    response = sampler.sample_qubo(theta)

    return list(response.first.sample.values())
