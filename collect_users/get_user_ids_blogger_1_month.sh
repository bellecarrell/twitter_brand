#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N get_user_ids
#$ -S /bin/bash
#$ -o out.txt
#$ -e err.txt

source activate /home/hltcoe/acarrell/.conda/envs/py3

TWITTER_PATH=/twitter/current/sample/2018/07/
OUT_DIR=/exp/acarrell/twitter_brand/blogger_1_month/

python get_user_ids.py ${TWITTER_PATH} ${OUT_DIR} --stats --om BLOGGER_ONLY
