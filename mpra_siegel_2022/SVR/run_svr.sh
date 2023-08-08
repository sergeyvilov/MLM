#!/bin/bash

#################################################################
#
#Run SVR on Griesemer MPRA data
#
#
#sbatch --array=0-23%10 run_svr.sh
#################################################################

#SBATCH -J mpra_svr
#SBATCH -c 16
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/griesemer/slurm_logs/%a.e

source ~/.bashrc; conda activate svilov-mlm

svr_dir='/s/project/mll/sergey/effect_prediction/MLM/siegel_2022/SVR/'

c=0

for response in stability steady_state; do

    for cell_type in Jurkat Beas2B; do

        for model in MLM 4mers 5mers 6mers word2vec effective_length; do

            if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

                output_dir="$svr_dir/$response/$cell_type/$model"

                params="--cell_type $cell_type --model $model --output_dir $output_dir \
                --N_trials 1000 --keep_first --response $response --N_splits 1000 --N_CVsplits 5 --seed 1"

                mkdir -p $output_dir

                python run_svr.py ${params} > ${output_dir}/log 2>${output_dir}/err 
            fi

            c=$((c+1))
        done
    done
done
        

        
