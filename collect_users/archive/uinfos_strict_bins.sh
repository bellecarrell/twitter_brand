#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=48:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N uinfos_strict_bins
#$ -o out.txt
#$ -e err.txt
#$ -S /bin/bash

ROLE=$1

source /home/hltcoe/acarrell/.bashrc

python uinfos_strict_bins.py /exp/acarrell/twitter_brand/artist_users/${ROLE} /exp/acarrell/twitter_brand/artist_users/strict/${ROLE}/