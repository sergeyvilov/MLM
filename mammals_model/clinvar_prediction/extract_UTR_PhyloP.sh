#!/bin/bash

UTR_dir='/s/project/mll/sergey/effect_prediction/MLM/UTR_coords/'

PhyloP_dir='/s/project/mll/sergey/effect_prediction/tools/PhyloP/'



while read -r chr start end _;do

  tabix ${PhyloP_dir}241-mammalian-2020v2.tsv.gz "$chr:$start-$end" >> ${UTR_dir}PhyloP241_intersect.tsv
  tabix ${PhyloP_dir}hg38.phyloP100way.tsv.gz  "$chr:$start-$end" >> ${UTR_dir}PhyloP100_intersect.tsv

  #echo $chr $start $end

done <  ${UTR_dir}GRCh38_3_prime_UTR_clean.bed
