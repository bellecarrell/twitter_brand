import argparse
import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
from analysis.rlr import RandomizedRegression as RR
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import gzip
import logging
import csv
from collections import defaultdict
import datetime
from twokenize import twokenize


logging.basicConfig(level=logging.INFO)
import copy
import time
import datetime
import random
import numpy as np
import json
from scipy.sparse import csr_matrix
import multiprocessing as mp
from nltk.corpus import stopwords
nltk.download('stopwords')

BATCH_START_WINDOW = datetime.datetime(2018, 4, 1)
BATCH_END_WINDOW = datetime.datetime(2018, 7, 31)
SEED = 12345
VERBOSE = True


# https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates
def randomDate(start, end, prop):
    stime, etime = start, end
    ptime = stime + prop * (etime - stime)
    return ptime


def days_between(d1, d2):
    return abs((d2 - d1).days)


def time_window(date, window_size):
    tw_delta = datetime.timedelta(days=window_size)
    stop = date + tw_delta
    return (date, stop)


def posted_recently(collection_date, user_date):
    return datetime.datetime.fromtimestamp(collection_date) - datetime.timedelta(
        days=60) <= datetime.datetime.fromtimestamp(user_date)


def filter_inactive_users(df, users):
    most_recent_collection_date = max(df['created_at'].values)
    return [u for u in users if
            posted_recently(most_recent_collection_date, max(df.loc[df['user_id'] == u]['created_at'].values))]


def tweets(f):
    """
    Creates generator of all tweets in file.
    :param f: input file, rows formatted ['tweet_id', 'created_at', 'text', 'user_id']
    :return: generator of tweets in corpus
    """
    return (row[2] for row in csv.reader(f))


def dated_tweets_by_user(f, users):
    """
    Get tweets from the timeline compressed file and place in dict sorted by user.
    Filtered to include only tweets within window
    :param in_dir: top-level directory containing promoting user dfs
    :return: dict of tweets by user ID
    """
    reader = csv.reader(f)
    tweets = defaultdict(list)
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
            tweets[user_id].append((created_at, text))
            if created_at > dates[user_id]:
                dates[user_id] = created_at
            if created_at > most_recent:
                most_recent = created_at
        count += 1
        if count % 100000 == 0:
            print('Read {} rows'.format(count))

    print('Filtering users with {} most recent'.format(most_recent))
    recent_users_removed = 0
    all_users = copy.deepcopy(list(tweets.keys()))
    for user in all_users:
        u_date = dates[user]
        if not posted_recently(most_recent, u_date):
            del tweets[user]
            del dates[user]
            recent_users_removed += 1
    print('{} users removed'.format(recent_users_removed))

    return tweets.keys(), tweets


def filter_by_tw_and_specialization(static_info, dates_tweets, tw):
    start, stop = tw
    tweets = defaultdict(list)
    for user in dates_tweets.keys():
        specialization = static_info.loc[static_info['user_id'] == user]['category_most_index-mace_label'].values[0]
        for created_at, text in dates_tweets[user]:
            if start <= datetime.datetime.fromtimestamp(created_at) <= stop:
                if len(text) > 0 and type(specialization) is not float:
                    tweets[user].append(text)
    return tweets.keys(), tweets

def _X_us_for_spec(static_info, s, X, us):
    us = list(us)
    filt_us = []
    for u in us:
        specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]
        if s == specialization:
            filt_us.append(u)

    filt_X = np.zeros(shape=(len(filt_us),20000))
    for i, u in enumerate(filt_us):
        orig_i = us.index(u)
        filt_X[i] = X[orig_i]

    return X, us

def _generate_batch(static_info, dates_tweets, time_window, vectorizer):
    print('generating batch for {} time window'.format(time_window))
    filtered_users, tweets_by_user = filter_by_tw_and_specialization(static_info, dates_tweets, time_window)

    X = np.zeros(shape=(len(filtered_users), len(vectorizer.vocabulary.keys())))
    for i, u in enumerate(filtered_users):
        u_tweets = [' '.join(tweets_by_user[u])]
        # todo: relative frequency
        Xi_unorm = vectorizer.transform(u_tweets).toarray()
        # sum = float(Xi_unorm.sum(axis=1))
        # Xi_unorm = np.array([i/sum for i in Xi_unorm])
        # row_sums = np.array(Xi_unorm.sum(axis=1),dtype=float)[:, 0]
        # row_indices, col_indices = Xi_unorm.nonzero()
        # Xi_unorm.data /= row_sums[row_indices]
        X[i] = Xi_unorm

    print('batch generated, {} users'.format(len(filtered_users)))

    return X, filtered_users


def _generate_batches(static_info, timeline, vectorizer, n_batches=100, window_size=30):
    random.seed(SEED)
    end_date = BATCH_END_WINDOW - datetime.timedelta(days=window_size)
    time_windows = [time_window(randomDate(BATCH_START_WINDOW, end_date, random.random()), window_size) for n in
                    range(n_batches)]
    users = static_info.loc[static_info['classify_account-mace_label'] == 'promoting'][
        'user_id'].dropna().unique().tolist()
    filtered_users, dates_tweets = dated_tweets_by_user(timeline, users)

    return (_generate_batch(static_info, dates_tweets, tw, vectorizer) for tw in time_windows)


def _generate_y(us, mode, static_info, cutoff=70):
    y = np.array([0 for u in us])
    for i, u in enumerate(us):
        percentile = int(static_info.loc[static_info['user_id'] == u]['percentile'].values[0])
        if mode == 'low' and percentile < cutoff:
            y[i] = 1
        if mode == 'high' and percentile >= cutoff:
            y[i] = 1
    return y

def fit_batches(rr, Xs_users, n_batches, s, static_info, l1_range=[0.0, 1.0]):
    np.random.seed(SEED)

    if rr.log_l1_range:
        l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
    else:
        l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

    start = time.time()

    # fit each batch
    for b, (X_us, l1) in enumerate(zip(Xs_users, l1s)):
        X, us = X_us
        y = _generate_y(us, s, static_info)

        print('{} size of y'.format(y.shape))
        rr.fit_batch(X, y, l1, b)
        if VERBOSE:
            print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))


def fit_batches_all_spec(Xs_users, n_batches, static_info, cutoff=70, l1_range=[0.0, 1.0]):
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    np.random.seed(SEED)

    rr = RR(is_continuous=False, model_dir='./log_reg_models', log_l1_range=True)

    if rr.log_l1_range:
        l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
    else:
        l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

    start = time.time()

    rrs_for_spec = [(s, RR(is_continuous=False, model_dir='./log_reg_models', log_l1_range=True), RR(is_continuous=False, model_dir='./log_reg_models', log_l1_range=True)) for s in
                    specializations]

    # fit each batch
    for b, (X_us, l1) in enumerate(zip(Xs_users, l1s)):
        X, us = X_us

        for s_rrs in rrs_for_spec:
            s, low, high = s_rrs
            X, us = _X_us_for_spec(static_info,s,X,us)
            y_l = _generate_y(us, 'low', static_info)
            low.fit_batch(X, y_l, l1, b)
            y_h = _generate_y(us, 'high', static_info)
            high.fit_batch(X, y_h, l1, b)
        if VERBOSE:
            print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))

    return rrs_for_spec


def main(in_dir, out_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    timeline = gzip.open(os.path.join(in_dir, 'timeline/user_tweets.noduplicates.tsv.gz'), 'rt')

    v_file = gzip.GzipFile(os.path.join(in_dir, 'vocab.json.gz'), 'r')
    v_json = v_file.read()
    v_file.close()
    v_json = v_json.decode('utf-8')
    vocab = json.loads(v_json)
    n_batches = 100


    sw = set(stopwords.words('english'))
    sw.union({c for c in list(string.punctuation) if c is not "#" and c is not "@"})

    vectorizer = CountVectorizer(tokenizer=twokenize.tokenizeRawTweetText, max_features=20000, vocabulary=vocab,stop_words=sw)

    Xs_users = _generate_batches(static_info, timeline, vectorizer, n_batches=n_batches)
    timeline.close()

    with open(os.path.join(out_dir, 'rlr_selected_features_low_high.txt'), 'w+') as f:
        spec_fitted = fit_batches_all_spec(Xs_users, n_batches, static_info, l1_range=[0.0, 10.0])

        for s in spec_fitted:
            name, low, high = s
            low_salient_features = low.get_salient_features(dict((v, k) for k, v in vocab.items()), {0: 'target'})
            high_salient_features = high.get_salient_features(dict((v, k) for k, v in vocab.items()), {0: 'target'})
            f.write('Specialization: {} Range: {} Salient features: {}\n'.format(s, 'low', low_salient_features))
            f.write('\n')
            f.write('Specialization: {} Range: {} Salient features: {}\n'.format(s, 'high', high_salient_features))
            f.write('\n')

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
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs')
    parser.add_argument('--output_prefix', required=True,
                        dest='output_prefix', metavar='OUTPUT_PREFIX',
                        help='prefix to write out final labels, ' +
                             'descriptive statistics, and plots')
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_prefix

    main(in_dir, out_dir)