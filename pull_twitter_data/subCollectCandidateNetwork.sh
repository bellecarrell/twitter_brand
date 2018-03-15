#!/bin/bash

KEYS=(ac cr2 dl1 dl2 dl3 dl4 home t1 t2 t3 t4 t5 t6 t7 t8 tv2 tv3 tv4 tv5 tv6 work);

for i in {1..11};
  do let "IDX = ${i} - 1";
  echo "qsub -q all.q -cwd collectCandidateNets.sh ${KEYS[$IDX]} ${IDX};";
  qsub -q all.q -cwd collectCandidateNets.sh ${KEYS[$IDX]} ${IDX};
  sleep 5;
done;
