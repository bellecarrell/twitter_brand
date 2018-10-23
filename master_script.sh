#!/bin/sh
#$ -cwd
#$ -l mem_free=2g
#$ -l h_rt=24:00:00
#$ -l h_vmem=2g
#$ -l num_proc=1
#$ -N get_user_ids
#$ -S /bin/bash
#$ -o out.txt
#$ -e err.txt

source activate /home/hltcoe/acarrell/.conda/envs/py3

TWITTER_PATH=$one
USERS_BY_MONTH_DIR=$two
AGGREGATE_USERS_DIR=$three
USER_SAMPLE_DIR=$four
PRESTUDY_DIR=$six
FIRST_DIR=$seven
SECOND_DIR=$eight
THIRD_DIR=$nine

#Data collection
/home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/get_user_ids_blogger_1_year.sh ${TWITTER_PATH} ${USERS_BY_MONTH_DIR}
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/unique_users_by_id.py ${USERS_BY_MONTH_DIR} ${AGGREGATE_USERS_DIR}
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/sample_users_by_percentile_range.py ${AGGREGATE_USERS_DIR} ${USER_SAMPLE_DIR}

#Statistics
# todo:add follower_count_histogram.py once actually used

#HITs

#Pre-study (110)
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/generate_self_promoting_HIT_CSV.py ${USER_SAMPLE_DIR} ${USER_SAMPLE_DIR} 100
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/prestudy_hit_statistics.py ${USER_SAMPLE_DIR}/100_size_HIT_self_promoting.csv ${PRESTUDY_DIR}

#First 440
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/generate_self_promoting_HIT_CSV.py ${USER_SAMPLE_DIR} ${USER_SAMPLE_DIR} 400
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/prestudy_hit_statistics.py ${USER_SAMPLE_DIR}/400_size_HIT_self_promoting_1.csv ${FIRST_DIR}


#Second 440
#NOTE: had to copy first 400 sample into other dir before running
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/generate_self_promoting_HIT_CSV.py ${USER_SAMPLE_DIR} ${USER_SAMPLE_DIR} 400
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/prestudy_hit_statistics.py ${USER_SAMPLE_DIR}/400_size_HIT_self_promoting_2.csv ${SECOND_DIR}

#660
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/generate_self_promoting_HIT_CSV.py ${USER_SAMPLE_DIR} ${USER_SAMPLE_DIR} 600
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/collect_users/prestudy_hit_statistics.py ${USER_SAMPLE_DIR}/600_size_HIT_self_promoting.csv ${THIRD_DIR}
