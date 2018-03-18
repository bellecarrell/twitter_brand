#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N test
#$ -S /bin/bash
#$ -o out.txt
#$ -e err.txt

TWITTER_PATH=$in
OUT_DIR=$out

./iama_roles.py ${TWITTER_PATH} ${OUT_DIR}
