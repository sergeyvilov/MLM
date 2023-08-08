import pandas as pd

progress_dir = f'/s/project/mll/sergey/effect_prediction/MLM/clinvar/' #output dir

clinvar_vcf = '/s/project/mll/sergey/effect_prediction/tools/ClinVar/clinvar_20230722.vcf.gz'

utr3_bed = '/s/project/mll/sergey/effect_prediction/MLM/UTR_coords/GRCh38_3_prime_UTR_clean-sorted.bed'

PhyloP100_tsv='/s/project/mll/sergey/effect_prediction/tools/PhyloP/hg38.phyloP100way.tsv.gz'

PhyloP241_tsv='/s/project/mll/sergey/effect_prediction/tools/PhyloP/241-mammalian-2020v2.tsv.gz'

rule all:
    input:
        progress_dir + 'clinvar.3utr.tsv',

rule filter_calls:
    #use only mutations in promoter regions
    input:
        vcf = clinvar_vcf,
    output:
        vcf = temp(progress_dir + 'clinvar.filtered.vcf.gz'),
        tbi = temp(progress_dir + 'clinvar.filtered.vcf.gz.tbi'),
    shell:
        r'''
        bcftools view  -e 'CLNREVSTAT~"no_assertion" | CLNREVSTAT~"no_interpretation" | (CLNSIG!~"benign/i" & CLNSIG!~"pathogenic/i") | CLNSIG~"Conflicting"' {input.vcf} -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''

rule replace_chroms:
    input:
        vcf = progress_dir + 'clinvar.filtered.vcf.gz',
        tbi = progress_dir + 'clinvar.filtered.vcf.gz.tbi',
        chrom_conv = 'chrom_conv.txt',
    output:
        vcf = temp(progress_dir + 'clinvar.new_chroms.vcf.gz'),
        tbi = temp(progress_dir + 'clinvar.new_chroms.vcf.gz.tbi'),
    shell:
        r'''
        bcftools annotate --threads 4 \
        --rename-chrs {input.chrom_conv} \
        {input.vcf} \
        -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''



rule annotate_regions:
    #add gene annotations using promoter bed file
    #each promoter mutations is annotated with the corresponding gene(s)
    input:
        vcf = progress_dir + 'clinvar.new_chroms.vcf.gz',
        tbi = progress_dir + 'clinvar.new_chroms.vcf.gz.tbi',
        header = 'headers/utr3_header.txt',
        bed = utr3_bed,
    output:
        vcf = progress_dir + 'clinvar.utr.vcf.gz',
        tbi = progress_dir + 'clinvar.utr.vcf.gz.tbi',
    shell:
        r'''
        bcftools annotate --threads 4 \
        -h {input.header} \
        -c 'CHROM,FROM,TO,=UTR3' \
        -a {input.bed} \
        {input.vcf} \
        -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''

rule annotate_PhyloP100:
    input:
        vcf = progress_dir + 'clinvar.utr.vcf.gz',
        tbi = progress_dir + 'clinvar.utr.vcf.gz.tbi',
        header = 'headers/PhyloP100_header.txt',
        tsv = PhyloP100_tsv,
    output:
        vcf = progress_dir + 'clinvar.PhyloP100.vcf.gz',
        tbi = progress_dir + 'clinvar.PhyloP100.vcf.gz.tbi',
    shell:
        r'''
        bcftools annotate --threads 4 \
        -h {input.header} \
        -c 'CHROM,POS,PhyloP100' \
        -a {input.tsv} \
        {input.vcf} \
        -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''

rule annotate_PhyloP241:
    input:
        vcf = progress_dir + 'clinvar.PhyloP100.vcf.gz',
        tbi = progress_dir + 'clinvar.PhyloP100.vcf.gz.tbi',
        header = 'headers/PhyloP241_header.txt',
        tsv = PhyloP241_tsv,
    output:
        vcf = progress_dir + 'clinvar.PhyloP241.vcf.gz',
        tbi = progress_dir + 'clinvar.PhyloP241.vcf.gz.tbi',
    shell:
        r'''
        bcftools annotate --threads 4 \
        -h {input.header} \
        -c 'CHROM,POS,PhyloP241' \
        -a {input.tsv} \
        {input.vcf} \
        -Oz -o {output.vcf}
        tabix -f {output.vcf}
        '''

rule extract_data:
    input:
        vcf = progress_dir + 'clinvar.PhyloP241.vcf.gz',
        tbi = progress_dir + 'clinvar.PhyloP241.vcf.gz.tbi',
    output:
        tsv = progress_dir + 'clinvar.3utr.tsv',
    shell:
        r'''
        bcftools query -i 'UTR3!="."' -f "%CHROM\t%POS\t%ID\t%REF\t%ALT\t%UTR3\t%CLNSIG\t%PhyloP100\t%PhyloP241\n" {input.vcf} > {output.tsv}
        '''
