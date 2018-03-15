#!/bin/sh
#$ -cwd
#$ -l mem_free=1g
#$ -l h_rt=96:00:00
#$ -l h_vmem=1g
#$ -l num_proc=1
#$ -N getStatuses
#$ -S /bin/bash

KEY=$1
IDX=$2

python twitter_search.py --statuses --keypath=keys/${KEY}_keys.txt --accesstoken=access_tokens/access_token_${KEY}.txt --kwpath=lists_backfill/backfilled_ids.txt${IDX} --outdir=../full_statuses_backfill_4-1-2014/${IDX}/
