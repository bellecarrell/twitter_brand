import argparse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
from twokenize import twokenize
import csv
import gzip
import os
from collections import defaultdict
import datetime
import copy
import pandas as pd
import json


def posted_recently(collection_date,user_date):
    return datetime.datetime.fromtimestamp(collection_date) - datetime.timedelta(days=60) <= datetime.datetime.fromtimestamp(user_date)


def dated_tweets_by_user(f, users):
    """
    Get tweets from the timeline compressed file and place in dict sorted by user.
    Filtered to include only tweets within window
    :param in_dir: top-level directory containing promoting user dfs
    :return: dict of tweets by user ID
    """
    reader = csv.reader(f)
    dates_tweets = defaultdict(list)
    dates = defaultdict(int)
    most_recent = 0
    count = 0

    print('Reading rows from timeline file to filter users')
    for row in reader:
        if count == 0:
            count += 1
            continue
        tweet_id, created_at, text, user_id = row
        created_at, user_id = int(created_at), int(user_id)
        if user_id in users:
            dates_tweets[user_id].append((created_at, text))
            if created_at > dates[user_id]:
                dates[user_id] = created_at
            if created_at > most_recent:
                most_recent = created_at
        count += 1
        if count % 100000 == 0:
            print('Read {} rows'.format(count))

    print('Filtering users with {} most recent'.format(most_recent))
    recent_users_removed = 0
    all_users = copy.deepcopy(list(dates_tweets.keys()))
    for user in all_users:
        u_date = dates[user]
        if not posted_recently(most_recent, u_date):
            del dates_tweets[user]
            del dates[user]
            recent_users_removed += 1
    print('{} users removed'.format(recent_users_removed))

    return dates_tweets.keys(), dates_tweets



def all_tweets_by_user(users,dates_tweets):
    return (' '.join([dt[1] for dt in dates_tweets[u]]) for u in users)

def to_json_file(data,out_dir, fname):

    json_str = json.dumps(data) + "\n"  # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')  # 3. bytes (i.e. UTF-8)

    with gzip.GzipFile(os.path.join(out_dir,'{}.json.gz'.format(fname)), 'w') as fout:  # 4. gzip
        fout.write(json_bytes)

def main(in_dir, out_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    tweet_f = gzip.open(os.path.join(in_dir, 'timeline/tweet_texts.csv.gz'), 'rt')
    timeline = gzip.open(os.path.join(in_dir, 'timeline/user_tweets.noduplicates.tsv.gz'), 'rt')
    users = static_info.loc[static_info['classify_account-mace_label'] == 'promoting'][
        'user_id'].dropna().unique().tolist()
    filtered_users, dates_tweets = dated_tweets_by_user(timeline, users)
    corpus = all_tweets_by_user(filtered_users,dates_tweets)

    #todo: remove "usher names" and numbers and lemmatize
    sw = set(stopwords.words('english'))
    sw.union({c for c in list(string.punctuation) if c is not "#" and c is not "@"})

    vectorizer = CountVectorizer(tokenizer=twokenize.tokenizeRawTweetText,stop_words=sw,ngram_range=(1,2),min_df=10,max_df=0.8)
    vectorizer.fit(corpus)

    to_json_file(vectorizer.vocabulary_,os.path.join(out_dir),'vocab')
    to_json_file(dates_tweets,os.path.join(out_dir),'dates_tweets')

if __name__ == '__main__':
    """
    Run randomized logistic regression to see features selected for different specializations.
    May add different class of Ys in the future for user impact (so, change in follower count or
    a related metric)
    """

    parser = argparse.ArgumentParser(
        description='calculate initial statistics for "shotgun" analysis of user data.'
    )
    parser.add_argument('--input_dir', required=True,
                        dest='input_dir', metavar='INPUT_DIR',
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs. should have vocab')
    parser.add_argument('--output_prefix', required=True,
                        dest='output_prefix', metavar='OUTPUT_PREFIX',
                        help='prefix to write out final labels, ' +
                             'descriptive statistics, and plots')
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_prefix

    main(in_dir, out_dir)