# Quantum Annealing Learning Search Variants

This is a forked and modified version of the repository https://github.com/bonom/Quantum-Annealing-for-solving-QUBO-Problems. In detail, this repository provides a refactored version, with bug fixes and minor improvements, of the original Python implementation of QALS. Additionally, the code provided here supports different tabu types:
- `binary`: the tabu matrix is defined in the {0, 1} domain;
- `spin`: the tabu matrix is defined in the {-1, 1} domain and then converted to the {0, 1} domain;
- `binary_no_diag`: like the `binary` tabu type, but with the tabu matrix diagonal set to 0;
- `spin_no_diag`: like the `spin` tabu type, but with the tabu matrix diagonal set to 0 before the conversion;
- `hopfield_like`: inspired to Hopfield networks, with problem variables defined in the {0, 1} domain and the tabu matrix defined in the {-1, 1} domain. As in Hopfield networks, the diagonal of the tabu matrix is set to 0;
- `only_diag`: like the `spin` tabu type, but with the out-of-diagonal elements of the tabu matrix set to 0 before the conversion;
- `no_tabu`: the tabu matrix is null.

## 1. Prerequisites
In order to run the code, you need to have Python 3 installed. If you do not have Python 3 on your machine, we suggest to install Anaconda Python.

You may also want to create a virtual environment before performing the setup step. If you are using Anaconda Python, the shell command (for Python 3.8.10) is the following:
```shell
conda create -n "venv" python=3.8.10
```

Eventually, to use the quantum annealers provided by D-Wave, you must have an account on [D-Wave Leap](https://cloud.dwavesys.com/leap/login/?next=/leap/).

## 2. Setup
Once you have met the prerequisites, download the repository and move inside the project root folder. From the command line, this is the sequence of commands:
```shell
git clone https://github.com/ZarHenry96/QALS-variants.git
cd QALS-variants
```

Then, activate the virtual environment, if you are using it (for Anaconda, the command is `conda activate venv`), and install the required modules by runnning the following command:
```shell
pip install -r requirements.txt
```

Finally, you need to configure the access to the D-Wave solvers. To do this, follow the instructions provided [here](https://docs.ocean.dwavesys.com/en/stable/overview/install.html#set-up-your-environment).

## 3. Execution
To execute QALS, you have to run the `main.py` script providing a configuration file as a parameter. You can find an example of configuration file in the `config_files` folder (`default.json`).

If you want to test all tabu types on a specific problem instance, you can use the `run_exp.sh` bash script. The script in question takes as input a template configuration file (some examples are provided in the `config_files` folder), a dataset file (look into the `input_data` folder), and a seed for the RNG. Additionally, you can specify the maximum number of QALS iterations (2000 by default) as a fourth parameter.

## 4. Visualization
If you want to visually compare the results obtained by different tabu types on a certain problem, you can use the `plot_tabu_comp_boxplot.py` script (located in the `postprocessing` folder). An example of how to use it is provided in the `plot_npp_boxplots_example.sh` script.
