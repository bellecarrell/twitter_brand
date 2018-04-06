#!/bin/bash

ROLE=(designer artist poet rapper writer singer musician actor dancer freelance photographer journalist reporter);

for i in {1..14};
  do let "IDX = ${i} - 1";
  mkdir -p /exp/acarrell/twitter_brand/artist_users/plots/${ROLE[$IDX]}
  echo "qsub -q all.q -cwd hist_by_feat.sh ${ROLE[$IDX]}";
  qsub -q all.q -cwd hist_by_feat.sh ${ROLE[$IDX]};
  sleep 5;
done;
