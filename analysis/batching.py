"""
Script with helper methods for time window batching.
Includes batch generation, fitting batches with rlr.
"""

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
import nltk
from nltk.corpus import stopwords
from analysis.datetime_util import *
nltk.download('stopwords')
import string
import sklearn.model_selection

BATCH_START_WINDOW = datetime.datetime(2018, 4, 1)
BATCH_END_WINDOW = datetime.datetime(2018, 7, 31)
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
