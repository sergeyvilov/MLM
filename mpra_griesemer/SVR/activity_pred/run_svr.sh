#!/bin/bash

#################################################################
#
#Run SVR on Griesemer MPRA data
#
#
#sbatch --array=0-29%10 run_svr.sh
#################################################################

#SBATCH -J mpra_svr
#SBATCH -c 16
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.e

source ~/.bashrc; conda activate svilov-mlm

svr_dir='/s/project/mll/sergey/effect_prediction/MLM/griesemer/SVR_LeaveGroupOut/'

c=0

for cell_type in HMEC HEK293FT HEPG2 K562 GM12878 SKNSH; do

    for model in MLM 4mers 5mers word2vec griesemer; do
    
        if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then
        
            params="--cell_type $cell_type --model $model --output_dir $svr_dir \
            --N_trials 1000 --keep_first --N_splits 1000 --N_CVsplits 5 --seed 1"

            python run_svr.py ${params} > ${svr_dir}/${cell_type}-${model}.log 2>${svr_dir}/${cell_type}-${model}.err 
        fi
        
        c=$((c+1))
        
    done
done
        

        
