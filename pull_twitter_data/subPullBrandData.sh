#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=48:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N getInfos
#$ -o out.txt
#$ -e err.txt
#$ -S /bin/bash

KEY=$1
INDEX=$2

source /home/hltcoe/acarrell/.bashrc

python twitter_search.py --info --keypath=keys/${KEY}_keys.txt --accesstoken=access_tokens/access_token_${KEY}.txt --kwpath=/exp/acarrell/twitter_brand/aggregate_out/${INDEX}.txt --outdir=/exp/acarrell/twitter_brand/artist_users/${INDEX}