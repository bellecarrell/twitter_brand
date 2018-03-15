#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=168:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N gn.nodup
#$ -S /bin/bash

KEY=$1
IDX=$2

python twitter_search.py --netsample --keypath=keys/${KEY}_keys.txt --accesstoken=access_tokens/access_token_${KEY}.txt --kwpath=/export/a07/abenton/FOR-MARK/vaccine_friends/user_ids.vaccine.sorted.${IDX}.tsv --outdir=/export/a07/abenton/FOR-MARK/vaccine_friends/networks/
