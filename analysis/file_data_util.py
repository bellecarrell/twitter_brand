"""
Utility method for file and data handling for consistency across analysis/ scripts.
"""

import random
import gzip
import json
import os
import pandas as pd
SEED = 12345
random.seed(SEED)
import math
import numpy as np
np.random.seed(SEED)

def load_dates_tweets(in_dir,fname):
    dt = from_gz(in_dir,fname)
    return clean_dates_tweets(dt)

def from_gz(in_dir, fname):
    v_file = gzip.GzipFile(os.path.join(in_dir,'{}.json.gz'.format(fname)),'r')
    v_json = v_file.read()
    v_file.close()
    v_json = v_json.decode('utf-8')
    return json.loads(v_json)


def clean_dates_tweets(d):
    return dict((int(k),v) for k,v in d.items())


def split_list_3_way(l,frac1,frac2):
    def _split_no_q(l,n,frac1,frac2):
        frac0 = 1 - frac1 - frac2
        s0, s1 = int(n * frac0), int(n * (frac0 + frac1))
        return l[:s0], l[s0:s1], l[s1:]

    n = len(l)
    random.shuffle(l)
    if n % (1/frac1) == 0:
        return _split_no_q(l,n,frac1,frac2)
    else:
        m = int(math.floor(n*frac1)/frac1)
        l0,l1,l2 = _split_no_q(l[:m],m,frac1,frac2)
        r = l[m:]
        for el in r:
            x = np.random.choice([0,1,2],1,p=[(1-frac1-frac2),frac1,frac2])
            if x == 0:
                l0.append(el)
            elif x == 1:
                l1.append(el)
            else:
                l2.append(el)
        return l0,l1,l2


def train_dev_test(in_dir, users, dev_frac, test_frac):
    """

    :param in_dir:
    :param users:
    :param dev_frac:
    :param test_frac:
    :return:
    """
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    percentiles = static_info['percentile'].dropna().unique().tolist()
    train, dev, test = [], [], []

    for s in specializations:
        for p in percentiles:
            s_p_users = static_info.loc[(static_info['percentile'] == p) & (static_info['category_most_index-mace_label'] == s)]['user_id'].values.tolist()
            s_p_users = [u for u in s_p_users if u in users]
            sp_train, sp_dev, sp_test = split_list_3_way(s_p_users,dev_frac,test_frac)
            train.extend(sp_train)
            dev.extend(sp_dev)
            test.extend(sp_test)
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


