#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N desc_role_file
#$ -S /bin/bash
#$ -o out.txt
#$ -e err.txt

TWITTER_PATH=/twitter/current/sample/2017/01/2017_01_01_00_00_00.gz
OUT_DIR=/exp/acarrell/

desc_roles_file.py ${TWITTER_PATH} ${OUT_DIR}
