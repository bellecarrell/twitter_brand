#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=48:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N getPlaceNames
#$ -S /bin/bash

KEY=$1
IDX=$2

python twitter_search.py --geosearch --keypath=/home/hltcoe/abenton/e-cigs/twitter_collection/keys/${KEY}_keys.txt --accesstoken=/home/hltcoe/abenton/e-cigs/twitter_collection/access_tokens/access_token_${KEY}.txt --kwpath=/home/hltcoe/abenton/e-cigs/search_twitter_locations/placeNames${IDX}.txt  --outdir /home/hltcoe/abenton/e-cigs/search_twitter_locations/placeNames