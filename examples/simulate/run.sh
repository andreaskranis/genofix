#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "script is running in: $SCRIPTPATH"
PPATH="$( cd ../../src/ "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "setting PYTHONPATH to src: $PPATH"

export PYTHONPATH=$PPATH

folder=$(date +"%m-%d-%y_%T")

echo "output will be to : simulation_correction_$folder"

for tiethreshold in 0.2 0.1 0.05 0.01
do 

for single in 0.6 0.7 0.8 0.9
do

for pair in 0.6 0.7 0.8 0.9
do

echo "tiethreshold = $tiethreshold single = $single pair = $pair"
window=4

python3 $PPATH/simulate/simulate.py -p example.fam -s fake_snps.map -w $window -l $single -m $pair -t $tiethreshold -o simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}

mkdir -p simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}/pair_stats
python3 $PPATH/plotting/plot_rank_statistics.py -s simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}/pair_stats.txt -o simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}/pair_stats

mkdir -p simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}/single_stats
python3 $PPATH/plotting/plot_rank_statistics.py -s simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}/single_stats.txt -o simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}/single_stats

cat simulation_correction_$folder/*/statistics.tsv > simulation_correction_$folder/statistics.tsv

done
done
done
