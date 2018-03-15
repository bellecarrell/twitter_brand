#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=168:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N gutime.nodup
#$ -S /bin/bash

KEY=$1
IDX=$2

python twitter_search.py --pasttweets --keypath=keys/${KEY}_keys.txt --accesstoken=access_tokens/access_token_${KEY}.txt --kwpath=/export/a07/abenton/FOR-MARK/demographics_recent_tweets/unique_user_ids.txt --outdir=/export/a07/abenton/FOR-MARK/demographics_recent_tweets/user_past_tweets/ --numtocache=200
