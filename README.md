# Quantum Annealing for solving QUBO Problems

Python implementation of the algorithm presented in "Pastorello, D. and Blanzieri, E., 2019. Quantum annealing learning search for solving QUBO problems. *Quantum Information Processing*, 18(10), p.303".

To execute the program, run the `main.py` script providing a configuration file as a parameter (e.g. `config_files/default.json`).

The tabu types supported (`tabu_type` parameter) are the following:
- `binary`
- `spin`
- `binary_no_diag`
- `spin_no_diag`
- `hopfield_like`
- `only_diag`
- `no_tabu`
