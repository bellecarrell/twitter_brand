#!/bin/bash

KEYS=(ac cr2 dl1 dl2 dl3 dl4 home t1 t2 t3 t4 t5 t6);
ARTS=(designer poet rapper writer singer musician actor artist dancer freelance photographer journalist reporter);

for i in {1..14};
  do let "IDX = ${i} - 1";
  mkdir -p /exp/acarrell/twitter_brand/artist_users/${ARTS[$IDX]}
  echo "qsub -q all.q -cwd collectArtistInfos.sh ${KEYS[$IDX]} ${ARTS[$IDX]}";
  qsub -q all.q -cwd collectArtistInfos.sh ${KEYS[$IDX]} ${ARTS[$IDX]};
  sleep 5;
done;
