#!/bin/bash

#################################################################
#
#Default run of species-aware model
#
#sbatch --array=0-0%1 default.sh
#################################################################

#SBATCH -J MLM_default
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/nnc_logs/slurm_logs/%a.o
#SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/nnc_logs/slurm_logs/%a.e

source ~/.bashrc; conda activate svilov-spade

test_name=seq_len_5000

fasta='/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/240_mammals.shuffled.fa'

species_list='/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/240_species.txt'

logs_dir='/s/project/mll/sergey/effect_prediction/MLM/nnc_logs/'

script_dir='/data/ouga/home/ag_gagneur/l_vilov/workspace/species-aware-DNA-LM/mammals_model/'

cd $script_dir

output_dir="$logs_dir/$test_name/"

echo "Output dir: $output_dir"

NN_PARAMETERS="${COMMON_NN_PARAMETERS}  \
--fasta $fasta  --species_list $species_list --output_dir ${output_dir} \
--save_at 2:11:3 --validate_every 1  \
--train_splits 4 --tot_epochs 11 --n_layers 4 --batch_size 128 --weight_decay 0 --seq_len 5000"

echo "output dir = ${output_dir}"
echo "NN parameters = ${NN_PARAMETERS}"

mkdir -p $output_dir

python main.py ${NN_PARAMETERS} > ${output_dir}/log 2>${output_dir}/err 
