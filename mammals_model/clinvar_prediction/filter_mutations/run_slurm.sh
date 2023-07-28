#!/bin/bash
#exclude servers with no AVX support (otherwise calling takes ages)!
#mutect2 log should have the line "Using CPU-supported AVX-512 instructions"
#if the line is absent, the node should also be excluded as AVX works somehow more slowly on these machines

source /data/ouga/home/ag_gagneur/l_vilov/.bashrc
conda activate bio
rm -f slurm_logs/*
mkdir slurm_logs
snakemake -s Snakefile.py -k --use-conda --latency-wait 180 --cluster-config cluster.yaml --cluster 'sbatch \
--mem={cluster.mem}  --threads-per-core={cluster.threads_per_core} -c {cluster.cores} -o slurm_logs/%j.out' -j 100
#-x ibis216-010-0[68-71] \
