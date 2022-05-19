import dwave_networkx as dnx
import networkx as nx
import sys

from qals.utils import Colors, now


def get_topology_active_adj_matrix(sampler, n):
    A = dict()

    sampler_nodes = list(sampler.nodelist)
    nodes_considered = list()
    for i in range(n):
        try:
            nodes_considered.append(sampler_nodes[i])
        except IndexError:
            print(f"Error when reaching the {i}-th element of the sampler nodes list (len = {len(sampler_nodes)})\n"
                  f"There are not enough active nodes to solve the problem provided.",
                  file=sys.stderr)
            exit(0)

    for node in nodes_considered:
        A[node] = list()

    for node_1, node_2 in sampler.edgelist:
        if node_1 in nodes_considered and node_2 in nodes_considered:
            A[node_1].append(node_2)
            A[node_2].append(node_1)

    return A


def generate_chimera_topology_adj_matrix(qubits_num):
    G = dnx.chimera_graph(16)
    tmp = nx.to_dict_of_lists(G)

    rows = []
    cols = []
    for i in range(qubits_num):
        rows.append(i)
        cols.append(i)
        for j in tmp[i]:
            if j < qubits_num:
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))


def generate_pegasus_topology_adj_matrix(qubits_num):
    G = dnx.pegasus_graph(16)
    tmp = nx.to_numpy_matrix(G)

    rows = []
    cols = []
    for i in range(qubits_num):
        rows.append(i)
        cols.append(i)
        for j in range(qubits_num):
            if tmp.item(i, j):
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))


def get_adj_matrix(simulation, topology, sampler, n):
    A, string = None, ''

    if simulation:
        if topology.lower() == 'chimera':
            if n <= 2048:
                A = generate_chimera_topology_adj_matrix(n)
                string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " \
                         + Colors.OKCYAN + "Using Chimera Topology \n" + Colors.ENDC
            else:
                print(now() + " [" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC
                      + "] " + f"the number of QUBO variables ({n}) is larger than the topology size (2048)",
                      file=sys.stderr)
                exit(0)
        else:
            if n <= 5640:
                A = generate_pegasus_topology_adj_matrix(n)
                string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER \
                         + "Using Pegasus Topology \n" + Colors.ENDC
            else:
                print(now() + " [" + Colors.BOLD + Colors.ERROR + "ERROR" + Colors.ENDC
                      + "] " + f"the number of QUBO variables ({n}) is larger than the topology size (5640)",
                      file=sys.stderr)
                exit(0)

    else:
        A = get_topology_active_adj_matrix(sampler, n)
        string = now() + " [" + Colors.BOLD + Colors.OKGREEN + "LOG" + Colors.ENDC + "] " + Colors.HEADER \
                 + f"Using {topology} Topology \n" + Colors.ENDC

    return A, string
