#!/bin/sh
# Gets iama counts for each subdirectory in a top-level directory of files containing Twitter stream data
# in JSON format.

TWITTER_YEAR_DIR=$1
OUT_DIR=$2

for i in 01 02 03 04 05 06 07 08 09 10 11 12
do
       mkdir -p $OUT_DIR/$i
       qsub -q all.q -cwd -v one=$TWITTER_YEAR_DIR/$i,two=$OUT_DIR/$i/ ./desc_roles.sh
done