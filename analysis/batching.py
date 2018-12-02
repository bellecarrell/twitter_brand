"""
Script with helper methods for time window batching.
Includes batch generation, fitting batches with rlr.
"""

import sys

import pandas as pd
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
import time
import random
import numpy as np
import nltk
from analysis.datetime_util import *
import scipy.sparse
nltk.download('stopwords')

BATCH_START_WINDOW = datetime.datetime(2018, 4, 1)
BATCH_END_WINDOW = datetime.datetime(2018, 7, 31)
BATCH_END_FOLLOWER_TS = datetime.datetime(2018, 10, 31)
SEED = 12345
VERBOSE = True

np.seterr(all='raise')


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


def filter_by_tw_and_specialization_precomputed(static_info, feature_df, tw):
    start, stop = tw
    
    not_nan = ~static_info['category_most_index-mace_label'].isna()
    promoting_users = set(static_info.loc[(static_info['classify_account-mace_label'] == 'promoting') &
                                          not_nan, 'user_id'])
    
    feature_df = feature_df[feature_df['user_id'].isin(promoting_users)]  # restrict to promoting users
    feature_df = feature_df[feature_df['created_at'].map(
            lambda x: start <= datetime.datetime.utcfromtimestamp(x) <= stop)
    ]  # only keep tweets from preset time range
    
    return feature_df['user_id'].unique(), feature_df


def generate_batch(static_info, dates_tweets, time_window, vectorizer,ret_tw=False):
    print('generating batch for {} time window'.format(time_window))
    filtered_users, tweets_by_user = filter_by_tw_and_specialization(static_info, dates_tweets, time_window)

    # filter out users with zero feature vectors
    filtered_users_zero_fv = []
    for i, u in enumerate(filtered_users):
        u_tweets = [' '.join(tweets_by_user[u])]
        Xi_unorm = vectorizer.transform(u_tweets).toarray()
        # todo: relative frequency
        sum = float(Xi_unorm.sum(axis=1))
        if sum != 0.0:
            filtered_users_zero_fv.append(u)

    X = np.zeros(shape=(len(filtered_users_zero_fv), len(vectorizer.vocabulary.keys())))
    for i, u in enumerate(filtered_users_zero_fv):
        u_tweets = [' '.join(tweets_by_user[u])]
        Xi_unorm = vectorizer.transform(u_tweets).toarray()
        # todo: relative frequency
        sum = float(Xi_unorm.sum(axis=1))
        X[i] = Xi_unorm / sum

    print('batch generated, {} users'.format(len(filtered_users_zero_fv)))

    if not ret_tw:
        return X, filtered_users_zero_fv
    else:
        return X, filtered_users_zero_fv, time_window


def generate_batch_precomputed_features(static_info, tweet_feature_df, time_window, vocab_key, ret_tw=False):
    print('generating batch for {} time window'.format(time_window))
    filtered_users, tweet_feature_df = filter_by_tw_and_specialization_precomputed(static_info,
                                                                                   tweet_feature_df,
                                                                                   time_window)
    
    # join features from several tweets
    def _join_features(feats):
        uni_counts = {}
        bi_counts = {}
        
        c1 = 0
        c2 = 0
        
        for fs in eval(feats):
            for f in fs:
                
                try:
                    f = eval(f)
                except Exception as mistake:
                    import pdb; pdb.set_trace()
                
                if len(f) == 1:
                    if f not in uni_counts:
                        uni_counts[f] = 0
                    uni_counts[f] += 1
                    c1 += 1
                elif len(f) == 2:
                    if f not in bi_counts:
                        bi_counts[f] = 0
                    bi_counts[f] += 1
                    c2 += 1
                else:
                    raise Exception('Problem reading feature: {}'.format(f))
        
        feats = dict([(k, v/c1) for k, v in uni_counts.items()] + [(k, v/c2) for k, v in bi_counts.items()])
        
        return feats
    
    # collect all non-zero indices for each user
    user_feature_df = tweet_feature_df.groupby('user_id')['extracted_features'].agg(_join_features)
    
    filtered_users_zero_fv = user_feature_df.loc[
        user_feature_df['extracted_features'].map(lambda x: len(x) == 0), 'user_id'
    ].tolist()
    user_feature_df = user_feature_df[user_feature_df['extracted_features'].map(lambda x: len(x) > 0)]
    
    max_col = len(vocab_key)
    
    rcv = [(r, c, v) for r, feats in enumerate(user_feature_df['extracted_features']) for c, v in feats.items()]
    
    X = scipy.sparse.csr_matrix(([x[2] for x in rcv], ([x[0] for x in rcv], [x[1] for x in rcv])),
                                shape=(user_feature_df.shape[0], max_col))
    
    print('batch generated, {} users'.format(len(filtered_users_zero_fv)))

    if not ret_tw:
        return X, filtered_users_zero_fv
    else:
        return X, filtered_users_zero_fv, time_window


def generate_batches_precomputed_features(static_info, feature_df, rev_vocab_key, n_batches=100, window_size=30, ret_tw=False, full_time_range=(BATCH_START_WINDOW, BATCH_END_WINDOW)):
    random.seed(SEED)
    end_date = full_time_range[1] - datetime.timedelta(days=window_size)
    time_windows = [time_window(randomDate(full_time_range[0],
                                           end_date,
                                           random.random()),
                                window_size) for _ in
                    range(n_batches)]
    
    return (generate_batch_precomputed_features(static_info,
                                                feature_df,
                                                tw,
                                                rev_vocab_key,
                                                ret_tw) for tw in time_windows)


def generate_batches(static_info, dates_tweets, vectorizer, n_batches=100, window_size=30, ret_tw=False):
    random.seed(SEED)
    end_date = BATCH_END_WINDOW - datetime.timedelta(days=window_size)
    time_windows = [time_window(randomDate(BATCH_START_WINDOW, end_date, random.random()), window_size) for n in
                    range(n_batches)]

    return (generate_batch(static_info, dates_tweets, tw, vectorizer,ret_tw) for tw in time_windows)


def generate_y_binary(us, s, static_info):
    y = np.array([0 for u in us])
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    s_to_y_val = dict((s, i) for i, s in enumerate(specializations))
    s_i = s_to_y_val[s]
    for i, u in enumerate(us):
        specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]
        u_i = s_to_y_val[specialization]
        if u_i == s_i:
            y[i] = 1

    return y


def generate_y_multi(us, static_info):
    y = np.array([0 for u in us])
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    s_to_y_val = dict((s, i) for i, s in enumerate(specializations))
    for i, u in enumerate(us):
        specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]
        y[i] = s_to_y_val[specialization]

    return y


def fit_batches(rr, Xs_users, n_batches, static_info, y_func, l1_range=[0.0, 1.0]):
    np.random.seed(SEED)

    if rr.log_l1_range:
        l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
    else:
        l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

    start = time.time()

    # fit each batch
    for b, (X_us, l1) in enumerate(zip(Xs_users, l1s)):
        X, us = X_us
        y = y_func(us, static_info)

        print('{} size of y'.format(y.shape))
        rr.fit_batch(X, y, l1, b)
        if VERBOSE:
            print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))
