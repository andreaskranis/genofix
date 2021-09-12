#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "script is running in: $SCRIPTPATH"
PPATH="$( cd ../../src/ "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "setting PYTHONPATH to src: $PPATH"

export PYTHONPATH=$PPATH

folder=$(date +"%m-%d-%y_%T")

echo "output will be to : simulation_correction_$folder"

python3 $PPATH/simulate/simulate.py -p example.fam -s fake_snps.map  -l 0.7 -m 0.5 -o simulation_correction_$folder
