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

follower_count_p = re.compile(r'"followers_count":.*?(\d+)', re.S)

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

def get_sample_users_by_percentile(user_json, percentiles, percentile_max, num_users_per_interval):
    sample_users_by_percentile = defaultdict(set)

    random.shuffle(user_json)

    for user in user_json:
        follower_count = follower_count_p.findall(user)
        if follower_count:
            log_follower_count = math.log(int(follower_count[0]), 10) if int(follower_count[0]) > 0 else 0
            for i, percentile in enumerate(percentile_max):
                if i > 0:
                    if percentiles[i-1] <= log_follower_count <= percentiles[i] and len(sample_users_by_percentile[percentile]) < num_users_per_interval:
                        sample_users_by_percentile[percentile].add(user)
                        break
                elif i == 0:
                    if log_follower_count <= percentiles[i] and len(sample_users_by_percentile[percentile]) < num_users_per_interval:
                        sample_users_by_percentile[percentile].add(user)
                        break

    return sample_users_by_percentile

def write_to_file(out_dir, sample_users):
    for range in sample_users.keys():
        fname = out_dir + "sample_users_for_{}th_percentile.txt".format(str(range))
        write_list_to_file(fname, sample_users_by_percentile[range], mode='w+')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output
    percentile_max = [20,30,40,50,60,70,80,90,95,99,100]
    num_users_per_interval = 200

    user_json = read_user_json_file(in_dir)
    follower_counts = collect_follower_counts(user_json)
    log_follower_percentiles = calculate_log_follower_percentiles(follower_counts, percentile_max)
    save_percentiles(out_dir, log_follower_percentiles)
    sample_users_by_percentile = get_sample_users_by_percentile(user_json, log_follower_percentiles, percentile_max, num_users_per_interval)
    write_to_file(out_dir, sample_users_by_percentile)

