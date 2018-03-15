#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=72:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N getCandNets
#$ -S /bin/bash

KEY=$1
IDX=$2

python twitter_search.py --netsample --keypath=keys/${KEY}_keys.txt --accesstoken=access_tokens/access_token_${KEY}.txt --kwpath=/export/a07/abenton/FOR-MARK/vaccine_friends/candidates.${IDX}.txt --outdir=/export/a07/abenton/FOR-MARK/vaccine_friends/candidate_networks/
