#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N collect_desc_roles
#$ -S /bin/bash
#$ -o out.txt
#$ -e err.txt

IAMA_OUT=/exp/acarrell/twitter_brand/artist_roles/
AGG_OUT=/exp/acarrell/twitter_brand/aggregate_out/

collect_desc_roles.py ${IAMA_OUT} ${AGG_OUT}
