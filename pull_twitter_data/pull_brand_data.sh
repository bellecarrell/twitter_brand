cd /home/hltcoe/acarrell/PycharmProjects/twitter_brand/pull_twitter_data/
export PATH=/opt/anaconda2/bin/:${PATH}
source activate py3k
python /home/hltcoe/acarrell/PycharmProjects/twitter_brand/pull_twitter_data/pull_brand_data.py ${1}
cd -
