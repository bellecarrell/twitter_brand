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

IAMA_OUT=/exp/acarrell/twitter_brand/collect_users
AGG_OUT=/exp/acarrell/twitter_brand/aggregate_out/iama_agg_out.txt

./collect_iama_counts.py ${IAMA_OUT} ${AGG_OUT}
