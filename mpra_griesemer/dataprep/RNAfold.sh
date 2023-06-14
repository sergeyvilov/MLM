#!/bin/bash

#extract  minimal free energy for each seqeunce in FASTA file using RNAfold

RNA_fold_command='/s/project/mll/sergey/effect_prediction/tools/ViennaRNA/bin/RNAfold  --noPS --noLP  '

workdir='/s/project/mll/sergey/effect_prediction/MLM/griesemer/fasta/'

fasta=$workdir'GRCh38_UTR_variants.fa'
output_RNAfold=$workdir'GRCh38_UTR_variants.free_energy.tsv'


cat $fasta|sed '1d'|sed 's/>.*/ /'|tr -d '\n'|tr ' ' '\n'|$RNA_fold_command \
|sed -n '1~2!p'|sed 's/.* //'|sed 's/[\(\)]//g' > $output_RNAfold
