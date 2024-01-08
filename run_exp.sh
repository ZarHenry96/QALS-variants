#!/usr/bin/bash

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "Illegal number of parameters!"
    echo "Correct usage: ./run_exp.sh file.template dataset_file seed [i_max]"
    echo "    file.template = .template configuration file"
    echo "    dataset_file = dataset file"
    echo "    seed = seed for the random numbers generator"
    echo "    i_max = maximum number of QALS iterations"
    exit 0
fi

# Parameters
TEMPLATE_CONFIG_FILE="$1"

DATASET_FILE="$2"
SEED="$3"
declare -a TABU_TYPES=(
    "binary"
    "spin"
    "binary_no_diag"
    "spin_no_diag"
    "hopfield_like"
    "only_diag"
    "no_tabu"
)
if [ "$#" -eq 4 ]; then
    I_MAX=$(($4))
else
    I_MAX=2000
fi

SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Create a temporary directory to store experiment configuration files
TMP_CONFIG_DIR="${SCRIPT_DIR}/tmp"
mkdir -p ${TMP_CONFIG_DIR}

# Iterate over tabu types
for tabu_type in "${TABU_TYPES[@]}"; do
    dataset_filename="${DATASET_FILE##*/}"
    exp_config_file="${TMP_CONFIG_DIR}/${dataset_filename%.*}_${tabu_type}.json"

    # Create the experiment configuration file starting from the template
    sed -e "s@\${data_file}@${DATASET_FILE}@" -e "s@\${seed}@${SEED}@" -e "s@\${tabu_type}@${tabu_type}@" \
        -e "s@\${i_max}@${I_MAX}@" "${TEMPLATE_CONFIG_FILE}" > "${exp_config_file}"

    # Run the experiment
    python main.py "${exp_config_file}"
    printf "\n\n\n"
done


# Delete the temporary directory
rm -rf "${TMP_CONFIG_DIR}"