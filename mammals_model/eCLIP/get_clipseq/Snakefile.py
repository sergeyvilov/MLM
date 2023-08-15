import numpy as np
import os
import pandas as pd


progress_dir = '/s/project/mll/sergey/effect_prediction/MLM/eCLIP/data/' #output dir

liftover_dir = '/s/project/mll/sergey/effect_prediction/tools/liftOver/'

table_3utr = '/s/project/mll/sergey/effect_prediction/MLM/UTR_coords/GRCh38_3_prime_UTR_clean.bed'

PhyloP_tsv = '/s/project/mll/sergey/effect_prediction/tools/PhyloP/hg38.phyloP100way.tsv.gz'

rule all:
    input:
        progress_dir + 'eCLIP.3utr.pos.PhyloP.bed',
        progress_dir + 'eCLIP.3utr.neg.PhyloP.bed',

rule get_tss_bed:
    input:
        meta_tsv = progress_dir + 'ENCSR456FVU_metadata.tsv' #https://www.encodeproject.org/publication-data/ENCSR456FVU/
    output:
        bed = progress_dir + 'eCLIP.hg19.bed6'
    shell:
        r'''
        > {output.bed}
        while read download_url; do
            wget -q $download_url -O - |bgzip -d >> {output.bed}
        done < <(cat {input.meta_tsv} |grep narrowPeak|cut -f21)
        '''

rule liftover:
    input:
        bed = progress_dir + 'eCLIP.hg19.bed6',
        chain_file = liftover_dir + 'hg19ToHg38.over.chain.gz' #chain file to convert positions see https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/
    output:
        bed = progress_dir + 'eCLIP.hg38.liftover.bed',
        umap = progress_dir + 'eCLIP_hg19.umap'
    log:
        progress_dir + 'logs/liftover.log'
    shell:
        r'''
        {liftover_dir}/liftOver -bedPlus=6 {input.bed}  {input.chain_file} {output.bed}  {output.umap}  > {log} 2>&1
        '''


rule extend_50:
    #extend by 50bp in 5' direction and sort
    input:
        bed = progress_dir + 'eCLIP.hg38.liftover.bed',
    output:
        bed = progress_dir + 'eCLIP.hg38.extended.bed',
    shell:
        r'''
        cat {input.bed}|awk 'BEGIN{{OFS="\t"}}{{if ($6=="+") {{$2-=50}} else {{$3+=50}};print}}'|sort -k1,1 -k2,2n > {output.bed}
        '''

rule limit_3utr:
    input:
        bed = table_3utr,
    output:
        bed = progress_dir + 'GRCh38.3utr_5Klimited.bed',
    shell:
        r'''
        cat {input.bed}|awk 'BEGIN{{OFS="\t"}}{{if ($6=="+") {{$3=($3<$2+5000)?$3:$2+5000}} else {{$2=($3-5000>$2)?$3-5000:$2}};print}}' > {output.bed}
        '''

rule split_cell_type:
    #split IDR  (storng peaks) peaks into cell types
    input:
        eclip_bed = progress_dir + 'eCLIP.hg38.extended.bed',
    output:
        eclip_bed = progress_dir + 'eCLIP.hg38.IDR_{cell_type}.bed',
    shell:
        r'''
        grep "_IDR" {input.eclip_bed}|grep  "{wildcards.cell_type}" > {output.eclip_bed}
        '''

# rule get_positive_merged_set:
#     #positive set: intersection of IDR (storng peaks) with 3'UTR coordinates
#     input:
#         HepG2_bed = progress_dir + 'eCLIP.hg38.IDR_HepG2.bed',
#         K562_bed = progress_dir + 'eCLIP.hg38.IDR_K562.bed',
#         utr_bed =  progress_dir + 'GRCh38.3utr_5Klimited.bed',
#     output:
#         bed = progress_dir + 'eCLIP.3utr.pos_merged.bed',
#     shell:
#         r'''
#         bedtools intersect -s -a {input.HepG2_bed} -b {input.K562_bed} | bedtools intersect -s -a stdin -b {input.utr_bed} \
#         |sort -k1,1 -k2,2n|bedtools merge -i - | awk 'BEGIN{{OFS="\t"}} {{if ($3-$2>50) {{print}} }}' | \
#         bedtools intersect -a stdin -b {input.utr_bed} -f 1 -wo  > {output.bed}
#         '''

rule get_positive_set:
    #positive set: intersection of IDR (storng peaks) with 3'UTR coordinates
    input:
        eclip_bed = progress_dir + 'eCLIP.hg38.extended.bed',
        utr_bed =  progress_dir + 'GRCh38.3utr_5Klimited.bed',
    output:
        bed = progress_dir + 'eCLIP.3utr.pos.bed',
    shell:
        r'''
        grep "_IDR" {input.eclip_bed} | bedtools intersect -s -a stdin -b {input.utr_bed} -wo -f 1 | sort -k1,1 -k2,2n \
        |awk 'BEGIN{{OFS="\t"}} {{if ($3-$2>50) {{print}} }}' > {output.bed}
        '''

rule get_negative_set:
    #negative set: all eCLIP (2 replicas+IDR) subtracted from 3'UTR coordinates
    input:
        eclip_bed = progress_dir + 'eCLIP.hg38.extended.bed',
        utr_bed = progress_dir + 'GRCh38.3utr_5Klimited.bed'
    output:
        bed = progress_dir + 'eCLIP.3utr.neg.bed',
    shell:
        r'''
        bedtools subtract -a {input.utr_bed} -b {input.eclip_bed} | sort -k1,1 -k2,2n \
        |awk 'BEGIN{{OFS="\t"}} {{if ($3-$2>50) {{print}} }}' > {output.bed}
        '''

rule annotate_PhyloP:
    input:
        bed = progress_dir + 'eCLIP.3utr.{subset}.bed',
        phylop_tsv = PhyloP_tsv
    output:
        bed = progress_dir + 'eCLIP.3utr.{subset}.PhyloP.bed',
    resources:
        mem = "16g"
    shell:
        r'''
            zcat {input.phylop_tsv}|awk 'BEGIN{{OFS="\t"}}{{print $1,$2-1,$2,$3}}' \
            | bedtools intersect -sorted -b stdin -a {input.bed} -F 1 -wo \
            |awk 'BEGIN{{OFS="\t"}}{{print $1,$2,$3,$4,$14,$25}}' \
            |awk '{{for (idx=1;idx<NF-1;idx++) {{printf $idx":"}};printf $(NF-1)"\t"$NF"\n"}}' \
            |awk '{{arr[$1]+=$2;count[$1]+=1}}END{{for (a in arr) {{print a"\t"arr[a] / count[a]}} }}'|sed 's/:/\t/g' > {output.bed}
        '''
