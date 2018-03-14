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

HOME='/home/hltcoe/acarrell/twitter_brand'
TWITTER_PATH='/twitter/current/sample/2018/01/2018_01_01_00_00_00.gz'

rm _user.txt

echo 'before script'

${HOME}/base_counts.py ${TWITTER_PATH} ${HOME}

echo 'after'
