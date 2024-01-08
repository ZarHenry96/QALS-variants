#!/usr/bin/bash

declare -a PROBLEM_SIZES=(500 1200 2500 5436)
declare -a TARGET_VALUES=(0 1 1 0)
declare -a QALS_PAPER_VALUES=(2640 6337 11681 528)

printf "\n"
last_problem_size_index=$((${#PROBLEM_SIZES[@]} - 1))
for i in $(seq 0 ${last_problem_size_index}); do
    problem_size="${PROBLEM_SIZES[i]}"
    target_value="${TARGET_VALUES[i]}"
    qals_paper_value="${QALS_PAPER_VALUES[i]}"

    echo "results/NPP/num_values_${problem_size}_range_10000"
    python postprocessing/plot_tabu_comp_boxplot.py \
           --root-res-dir "results/NPP/num_values_${problem_size}_range_10000" \
           --solution-key diff --target-value "${target_value}" \
           --qals-paper-value "${qals_paper_value}" \
           --x-label "Tabu Type" --y-label "Sets Difference" \
           --title "NPP, n=${problem_size}, range=10000, i_max=2000" \
           --out-filename "npp_n_${problem_size}_r_10000_boxplot.pdf" \
           --verbose
    printf "\n"
done
