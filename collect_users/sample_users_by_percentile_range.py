#!/usr/bin/env python
import sys

import numpy as np

sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
from configs.config import *
import re
import math
import matplotlib as plt
import random
from utils.file_io import *
import requests
from utils.twitter_user_util import *

follower_count_p = re.compile(r'"followers_count":.*?(\d+)', re.S)
id_p = re.compile(r'"id_str":.*?"(.+?)"', re.S)

def read_user_json_file(in_dir):
    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if filename.endswith("_json.txt"):
                    with open(os.path.join(dirpath, filename), 'rt') as f:
                        return f.readlines()

def collect_follower_counts(user_json):
    user_ids = set()
    follower_counts = []

    for user in user_json:
        follower_count = follower_count_p.findall(user)
        if follower_count:
            follower_count = int(follower_count[0])
            follower_counts.append(follower_count)

    return follower_counts

def calculate_log_follower_percentiles(follower_counts, percentile_max):
    log_follower_counts = [math.log(count, 10) if count > 0 else 0 for count in follower_counts]

    percentiles = [np.percentile(log_follower_counts, per) for per in percentile_max]

    return percentiles

def save_percentiles(out_dir, percentiles):
    fname = out_dir + "percentiles.txt"

    with open(fname, 'w+') as f:
        for val in percentiles:
            f.write("log_follower: {} follower: {}\n".format(val, 10**val))

def is_active_id(id):
    response = requests.get('https://twitter.com/intent/user?user_id=' + id)
    if response.status_code == 200:
        return True
    return False

def is_active(user):
    id = id_p.findall(user)
    active_user = False

    if id:
        id = id[0]
        active_user = is_active_id(id)

    return active_user

def get_sample_users_by_percentile(user_json, percentiles, percentile_max, num_users_per_interval):
    sample_users_by_percentile = defaultdict(set)
    all_users_by_percentile = defaultdict(set)

    random.shuffle(user_json)

    for user in user_json:
        follower_count = follower_count_p.findall(user)

        if follower_count:
            log_follower_count = math.log(int(follower_count[0]), 10) if int(follower_count[0]) > 0 else 0
            for i, percentile in enumerate(percentile_max):
                if i > 0:
                    if percentiles[i-1] <= log_follower_count <= percentiles[i] and is_active(user):
                        all_users_by_percentile[percentile].add(user)
                        if len(sample_users_by_percentile[percentile]) < num_users_per_interval:
                            sample_users_by_percentile[percentile].add(user)
                        break
                elif i == 0:
                    if log_follower_count <= percentiles[i] and is_active(user):
                        all_users_by_percentile[percentile].add(user)
                        if len(sample_users_by_percentile[percentile]) < num_users_per_interval:
                            sample_users_by_percentile[percentile].add(user)
                        break

    return sample_users_by_percentile, all_users_by_percentile

def write_to_file(out_dir, sample_users):

    for range in sample_users.keys():
        jsons = sample_users_by_percentile[range]
        ids = [field_from_json('id_str', json) for json in jsons]
        mturk_fields = [fields_for_mturk(json) for json in jsons]

        json_fname = out_dir + "json/{}.txt".format(str(range))
        ids_fname = out_dir + "ids/{}.users.txt".format(str(range))
        mturk_fname = out_dir + "mturk/{}.csv".format(str(range))

        write_list_to_file(json_fname, jsons, mode='w+')
        write_list_to_file(ids_fname, ids, mode='w+')
        write_rows_to_csv(mturk_fname, mturk_fields)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output
    percentile_max = [20,30,40,50,60,70,80,90,95,99,100]
    num_users_per_interval = 400

    user_json = read_user_json_file(in_dir)
    for user in user_json:
        if is_active(user) is False:
            pass

    follower_counts = collect_follower_counts(user_json)
    log_follower_percentiles = calculate_log_follower_percentiles(follower_counts, percentile_max)
    save_percentiles(out_dir, log_follower_percentiles)
    sample_users_by_percentile, all_users_by_percentile = get_sample_users_by_percentile(user_json, log_follower_percentiles, percentile_max, num_users_per_interval)
    write_to_file(out_dir, all_users_by_percentile)

    out_dir = out_dir + '/first_400_sample/'
    write_to_file(out_dir, sample_users_by_percentile)
