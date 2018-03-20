#!/bin/sh
# Gets iama counts for each subdirectory in a top-level directory of files containing Twitter stream data
# in JSON format.

TWITTER_YEAR_DIR=$1
OUT_DIR=$2

for i in 7 8 9
do
        mkdir -p $OUT_DIR/$i
        qsub -q all.q -cwd -v one=$TWITTER_YEAR_DIR/$i,two=$OUT_DIR/$i ./iama_roles.sh
done