"""
Utility method for file and data handling for consistency across analysis/ scripts.
"""
import random
import gzip
import json
import os
SEED = 12345

def from_gz(in_dir,fname):
    v_file = gzip.GzipFile(os.path.join(in_dir,'{}.json.gz'.format(fname)),'r')
    v_json = v_file.read()
    v_file.close()
    v_json = v_json.decode('utf-8')
    return json.loads(v_json)

def clean_dates_tweets(d):
    return dict((int(k),v) for k,v in d.items())

def train_dev_test(users,dev_frac, test_frac):
    random(SEED)
    n_users = len(users)
    random.shuffle(users)
    dev_u = n_users*dev_frac
    test_frac = n_users*test_frac

