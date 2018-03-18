#!/bin/sh
# Gets iama counts for each subdirectory in a top-level directory of files containing Twitter stream data
# in JSON format.

TWITTER_YEAR_DIR=$1
OUT_DIR=$2

for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
    mkdir -p $OUT_DIR/$i
    qsub -v in=$TWITTER_YEAR_DIR/$i out=$OUT_DIR/$i/./iama_roles.sh
done