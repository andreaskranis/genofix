#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "script is running in: $SCRIPTPATH"
PPATH="$( cd ../../src/ "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "setting PYTHONPATH to src: $PPATH"

export PYTHONPATH=$PPATH

folder=$(date +"%m-%d-%y_%T")

echo "output will be to : simulation_correction_$folder"

for window in 4 2
do

for lddist in global none local
do 

for tiethreshold in 0.4 0.3 0.2
do 

for single in 0.5 0.6
do

for pair in 0.5 0.6
do

echo "tiethreshold = $tiethreshold single = $single pair = $pair"

python3 ${PPATH}/simulate/simulate.py -p example.fam -s fake_snps.map -w ${window} -l ${single} -m ${pair} -t ${tiethreshold} -d ${lddist} -n 100 -o simulation_correction_${folder}/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}_ld_${lddist} 

mkdir -p simulation_correction_${folder}/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}_ld_${lddist}/pair_stats
python3 ${PPATH}/plotting/plot_rank_statistics.py -s simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}_ld_${lddist}/pair_stats.txt.gz -o simulation_correction_${folder}/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}_ld_${lddist}/pair_stats

mkdir -p simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}_ld_${lddist}/single_stats
python3 ${PPATH}/plotting/plot_rank_statistics.py -s simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}_ld_${lddist}/single_stats.txt.gz -o simulation_correction_$folder/errors_st_${single}_pt_${pair}_t_${tiethreshold}_snpwindow_${window}_ld_${lddist}/single_stats

cat simulation_correction_${folder}/*/statistics.tsv > simulation_correction_${folder}/statistics.tsv

done
done
done
done
done
