#!/bin/bash

# ################################################################
#
# Default run of species-aware model
#
# sbatch --array=0-0%1 default.sh
# ################################################################

# SBATCH -J MLM_default
# SBATCH --gres=gpu:1
# SBATCH -c 16
# SBATCH --mem=80G
# SBATCH -o /s/project/mll/sergey/effect_prediction/MLM/nnc_logs/slurm_logs/test_human.o
# SBATCH -e /s/project/mll/sergey/effect_prediction/MLM/nnc_logs/slurm_logs/test_himan.e

source ~/.bashrc; conda activate svilov-spade

test_name=motif_predictions/species_aware

fasta='/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/species/Homo_sapiens.fa'

species_list='/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/240_species.txt'

logs_dir='/s/project/mll/sergey/effect_prediction/MLM/nnc_logs/'

script_dir='/data/ouga/home/ag_gagneur/l_vilov/workspace/species-aware-DNA-LM/mammals_model/'

cd $script_dir

output_dir="$logs_dir/$test_name/"

echo "Output dir: $output_dir"

NN_PARAMETERS="${COMMON_NN_PARAMETERS}  \
--test --get_motif_acc --batch_size 1 --fasta $fasta  --species_list $species_list --output_dir ${output_dir}  \
--model_weight /s/project/mll/sergey/effect_prediction/MLM/nnc_logs/seq_len_5000/weights/epoch_11_weights_model"

echo "output dir = ${output_dir}"
echo "NN parameters = ${NN_PARAMETERS}"

mkdir -p $output_dir

python main.py ${NN_PARAMETERS} > ${output_dir}/log 2>${output_dir}/err 
