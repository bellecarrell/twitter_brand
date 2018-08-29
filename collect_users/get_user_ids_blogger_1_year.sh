#!/bin/sh
# Gets iama counts for each subdirectory in a top-level directory of files containing Twitter stream data
# in JSON format.

TWITTER_YEAR_DIR=/twitter/current/sample/2018
OUT_DIR=/exp/acarrell/twitter_brand/blogger_2018

for i in 01 02 03 04 05 06 07
do
       mkdir -p $OUT_DIR/$i
       qsub -q all.q -cwd -v one=$TWITTER_YEAR_DIR/$i,two=$OUT_DIR/$i/ ./get_user_ids.sh
done