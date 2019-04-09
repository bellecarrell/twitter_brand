#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N sp
#$ -S /bin/bash
#$ -o out.txt
#$ -e err.txt

source activate /home/hltcoe/acarrell/.conda/envs/py3

python sp.py --input_dir /exp/acarrell/twitter_brand/promoting_users/ --output_prefix /exp/acarrell/twitter_brand/analysis/final