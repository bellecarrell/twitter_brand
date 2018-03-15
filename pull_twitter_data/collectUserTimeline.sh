#!/bin/sh
#$ -cwd
#$ -l mem_free=20g
#$ -l h_rt=168:00:00
#$ -l h_vmem=20g
#$ -l num_proc=1
#$ -N sihua.user.timeline
#$ -S /bin/bash

KEY=$1
IDX=$2

python twitter_search.py --pasttweets --keypath=keys/${KEY}_keys.txt --accesstoken=access_tokens/access_token_${KEY}.txt --kwpath=../uids_for_sihua/uids_for_sihua.${IDX}.txt --outdir=../past_user_tweets_sihua/ --numtocache=200
