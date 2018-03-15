#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=48:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N getInfos
#$ -S /bin/bash

python twitter_search.py --info --keypath=keys/work_keys.txt --accesstoken=access_tokens/access_token_work.txt --kwpath=../uinfo_files/missingUsers.txt --outdir=../user_infos
