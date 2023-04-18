#!/bin/bash

#SBATCH -o extract_train.o
#SBATCH -e extract_train.e


##########################################################
# Extract 3'UTR regions from HAL archive to a FASTA file
# Don't take Homo Sapiens regions
#
# For regions with negative MAF (positive human genes) 
# or positive MAF (negative human genes) take reverse complement
#
##########################################################

LINE_WIDTH=80

data_dir=/s/project/mll/sergey/MLM

hal_file="$data_dir/600_way/241-mammalian-2020v2.hal"
utr_table="$data_dir/UTR_coords/GRCh38_3_prime_UTR_all_species.tsv" 

output_dir="$data_dir/fasta/"

mkdir -p $output_dir

output_fasta="$output_dir/240_mammals.fa" #241-1 (Homo Sapiens)

true > $output_fasta

while read -r HGNC_Symbol human_UTR_ID human_UTR_len human_transcript_strand species contig UTR_start UTR_end MAF_strand; do

#add stop codon at the beginning, for debugging
#if ([ "$human_transcript_strand" = "+" ] && [ "$MAF_strand" = "+" ]) ||  ([ "$human_transcript_strand" = "-" ] #&& [ "$MAF_strand" = "-" ]) then
#UTR_start=$((UTR_start-3))
#else
#UTR_end=$((UTR_end+3))
#fi

UTR_len=$((UTR_end-UTR_start))

echo  ">$human_UTR_ID:$species:$contig:$UTR_len" >> $output_fasta #sequence header

hal2fasta $hal_file $species --sequence $contig --start $UTR_start --length $UTR_len  --lineWidth $LINE_WIDTH| tail -n +2 > tmp.fa

if ([ "$human_transcript_strand" = "+" ] && [ "$MAF_strand" = "+" ]) ||  ([ "$human_transcript_strand" = "-" ] && [ "$MAF_strand" = "-" ]) then
    cat tmp.fa >> $output_fasta
else    
    #take reverse complement
    cat tmp.fa|tr -d '\n'|tr 'ACTGactg' 'TGACtgac'|rev|fold -w $LINE_WIDTH >> $output_fasta
    echo '' >> $output_fasta
fi

done < <(grep -v 'Homo_sapiens' $utr_table|tail -n +2)

rm tmp.fa