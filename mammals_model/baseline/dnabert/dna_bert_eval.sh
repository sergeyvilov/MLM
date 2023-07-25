#!/bin/bash

#################################################################
#
#DNA BERT inference
#
#sbatch --array=0-12%10 dna_bert_eval.sh
#################################################################

#SBATCH -J DNABERT
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/baseline/dnabert/slurm_logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/baseline/dnabert/slurm_logs/%a.e

source ~/.bashrc; conda activate svilov-mlm

model=default

dataset_len=1500
total_seq=18200

#fasta='/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/species/Homo_sapiens.fa'

output_dir="/s/project/mll/sergey/effect_prediction/MLM/baseline/dnabert/$model/eval"

mkdir -p $output_dir

c=0

for dataset_start in `seq 0 $dataset_len $total_seq`; do

    if [ ${SLURM_ARRAY_TASK_ID} -eq $c ]; then

        python -u dna_bert_eval.py  $output_dir $dataset_start $dataset_len > ${output_dir}/log 2>${output_dir}/err 

    fi

    c=$((c+1))

done
