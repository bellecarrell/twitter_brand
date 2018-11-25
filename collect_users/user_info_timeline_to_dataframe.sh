#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N user_info_to_dataframe
#$ -S /bin/bash
#$ -o out.txt
#$ -e err.txt

source activate /home/hltcoe/acarrell/.conda/envs/py3

python user_info_timeline_to_dataframe.py /exp/abenton/twitter_brand_data/ /exp/acarrell/twitter_brand/promoting_users