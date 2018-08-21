#!/bin/bash

ROLE=(designer artist poet rapper writer singer musician actor dancer freelance photographer journalist reporter);

for i in {1..14};
  do let "IDX = ${i} - 1";
  mkdir -p /exp/acarrell/twitter_brand/artist_users/stats/${ROLE[$IDX]}
  echo "qsub -q all.q -cwd uinfo_summ_stats.sh ${ROLE[$IDX]}";
  qsub -q all.q -cwd uinfo_summ_stats.sh ${ROLE[$IDX]};
  sleep 5;
done;

source /home/hltcoe/acarrell/.bashrc

python uinfo_summ_stats.py /exp/acarrell/twitter_brand/artist_users/ /exp/acarrell/twitter_brand/artist_users/stats/