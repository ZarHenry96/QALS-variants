import dwave_networkx as dnx
import math
import networkx as nx
import numpy as np
import sys

from qals.utils import Colors, now


def build_A_for_simulation(graph_adj_matrix, nodes_considered):
    A = dict()
    for node1 in nodes_considered:
        A[node1] = list()
        for node2 in graph_adj_matrix[node1]:
            if node2 in nodes_considered:
                A[node1].append(node2)

    return A


def generate_chimera_topology_adj_matrix(qubits_num):
    if qubits_num > 2048:
        print(now() + " [" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC
              + "] " + f"the number of QUBO variables ({qubits_num}) is larger than the topology size (2048)",
              file=sys.stderr)
        exit(0)

    G = dnx.chimera_graph(16)
    graph_adj_matrix = nx.to_dict_of_lists(G)
    nodes_considered = sorted(graph_adj_matrix.keys())[0:qubits_num]

    return build_A_for_simulation(graph_adj_matrix, nodes_considered)


def get_pegasus_lw_up_indices(sampler_nodes, qubits_num):
    center_node = (min(sampler_nodes) + max(sampler_nodes)) / 2
    nearest_node_index = np.argmin([abs(node - center_node) for node in sampler_nodes])

    ideal_min_index = nearest_node_index - math.ceil((qubits_num - 1) / 2)
    ideal_max_index = nearest_node_index + math.floor((qubits_num - 1) / 2) + 1

    lw_index = max(ideal_min_index - max(0, ideal_max_index - len(sampler_nodes)), 0)
    up_index = min(ideal_max_index + max(0, -ideal_min_index), len(sampler_nodes))

    return lw_index, up_index


def generate_pegasus_topology_adj_matrix(qubits_num):
    if qubits_num > 5640:
        print(now() + " [" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC
              + "] " + f"the number of QUBO variables ({qubits_num}) is larger than the topology size (5640)",
              file=sys.stderr)
        exit(0)

    G = dnx.pegasus_graph(16)
    graph_adj_matrix = nx.to_dict_of_lists(G)
    sampler_nodes = sorted(graph_adj_matrix.keys())
    lw_index, up_index = get_pegasus_lw_up_indices(sampler_nodes, qubits_num)
    nodes_considered = sampler_nodes[lw_index:up_index]

    return build_A_for_simulation(graph_adj_matrix, nodes_considered)


def get_active_adj_matrix(topology, sampler, qubits_num):
    A = dict()

    sampler_nodes = sorted(list(sampler.nodelist))
    if len(sampler_nodes) < qubits_num:
        print(f"There are not enough active nodes in the {topology.capitalize()} topology to solve the problem provided"
              f" ({len(sampler_nodes)} available, {qubits_num} required).", file=sys.stderr)
        exit(0)

    if topology.lower() == 'pegasus':
        lw_index, up_index = get_pegasus_lw_up_indices(sampler_nodes, qubits_num)
    else:
        lw_index, up_index = 0, qubits_num
    nodes_considered = sampler_nodes[lw_index:up_index]

    for node in nodes_considered:
        A[node] = list()

    for node_1, node_2 in sorted(sampler.edgelist):
        if node_1 in nodes_considered and node_2 in nodes_considered:
            A[node_1].append(node_2)
            A[node_2].append(node_1)

    return A


def get_adj_matrix(simulation, topology, sampler, qubits_num):
    A = None

    if simulation:
        if topology.lower() == 'chimera':
            A = generate_chimera_topology_adj_matrix(qubits_num)
        elif topology.lower() == 'pegasus':
            A = generate_pegasus_topology_adj_matrix(qubits_num)
        else:
            print(f"Unknown annealer topology '{topology}'!")
            exit(0)
    else:
        A = get_active_adj_matrix(topology, sampler, qubits_num)

    string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER \
             + f"Using {topology.capitalize()} Topology \n" + Colors.ENDC

    return A, string
