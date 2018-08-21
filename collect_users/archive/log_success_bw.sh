#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=48:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N log_success_bw
#$ -o out.txt
#$ -e err.txt
#$ -S /bin/bash

ROLE=$1

source activate /home/hltcoe/acarrell/.conda/envs/py2

python log_success_bw.py /exp/acarrell/twitter_brand/artist_users/${ROLE} /exp/acarrell/twitter_brand/artist_users/tmp_log_success/${ROLE}/