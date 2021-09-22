#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "script is running in: $SCRIPTPATH"
PPATH="$( cd ../../src/ "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "setting PYTHONPATH to src: $PPATH"

export PYTHONPATH=$PPATH

folder=$(date +"%m-%d-%y_%T")

echo "output will be to : simulation_correction_$folder"

single=0.3
pair=0.3
window=4

python3 $PPATH/simulate/simulate.py -p example.fam -s fake_snps.map -w $window -l $single -m $pair -o simulation_correction_$folder/errors_st_${single}_pt_${pair}_snpwindow_${window}

python3 $PPATH/plotting/plot_rank_statistics.py -s simulation_correction_$folder/errors_st_${single}_pt_${pair}/emat/empirical.txt -o simulation_correction_$folder/errors_st_${single}_pt_${pair}_snpwindow_${window}
