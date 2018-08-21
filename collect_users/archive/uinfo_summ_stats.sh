#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=48:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N uinfo_summ_stats
#$ -o out.txt
#$ -e err.txt
#$ -S /bin/bash

ROLE=$1

source /home/hltcoe/acarrell/.bashrc


python uinfo_summ_stats.py /exp/acarrell/twitter_brand/artist_users/${ROLE} /exp/acarrell/twitter_brand/artist_users/stats/${ROLE}/