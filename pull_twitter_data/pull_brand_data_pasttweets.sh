#!/bin/bash

cd /exp/abenton/twitter_brand/pull_twitter_data/
export PATH=/opt/anaconda2/bin/:${PATH}
source activate py3k
#source activate /home/hltcoe/acarrell/.conda/envs/py3
python /exp/abenton/twitter_brand/pull_twitter_data/pull_brand_data.py pasttweets
cd -
