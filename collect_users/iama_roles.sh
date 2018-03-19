#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N test
#$ -S /bin/bash
#$ -o $out/out.txt
#$ -e $out/err.txt

TWITTER_PATH=$one
OUT_DIR=$two

./iama_roles.py ${TWITTER_PATH} ${OUT_DIR}
