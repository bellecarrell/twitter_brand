import argparse
import pandas as pd
import os
import datetime
import copy
from analysis.file_data_util import *
from collections import defaultdict
from utils import file_io
from analysis.file_data_util import *
import csv
from collections import defaultdict

def calculate_percent_change(static_info, in_dir, us):
    info = gzip.open(os.path.join(in_dir, 'info/user_info_dynamic.tsv.gz'), 'rt')
    pcs = {}
    for u in us:
        start = static_info.loc[static_info['user_id'] == int(u)]['followers_count'].values[0]
        max = 0
        fc = 0
        pcs[u] = [start,max,fc]


    count = 0
    reader = csv.reader(info)
    for row in reader:
        if count % 50000 == 0:
            print('{} rows read from dynamic info file'.format(count))
        if count == 0:
            count += 1
            continue
        count += 1
        row = row[0].split('\t')
        timestamp, user_id, followers_count, friends_count, listed_count, statuses_count = row
        timestamp, user_id, followers_count = int(timestamp), int(user_id), int(followers_count)
        if user_id in us:
            start, max, fc = pcs[u]
            if timestamp > max:
                pcs[u] = [start, timestamp, followers_count]
                print('user: {} data: {}'.format(u, pcs[u]))


    for u in pcs.keys():
        start, max, fc = pcs[u]
        pcs[u] = (fc-start)/start


    return pcs

def main(in_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    static_info['percent_change'] = float('nan')
    users = static_info.loc[static_info['classify_account-mace_label'] == 'promoting'][
        'user_id'].dropna().unique().tolist()

    print('Calculating percent change and adding to df')
    pcs = calculate_percent_change(static_info, in_dir,users)

    for u, pc in pcs.items():
        static_info.loc[static_info['user_id'] == int(u), 'percent_change'] = pc

    print('saving df')
    static_info.to_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))

if __name__ == '__main__':
    """
    add 'percent_change' column to static_info    
    """

    parser = argparse.ArgumentParser(
        description='calculate initial statistics for "shotgun" analysis of user data.'
    )
    parser.add_argument('--input_dir', required=True,
                        dest='input_dir', metavar='INPUT_DIR',
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs. should have vocab')
    args = parser.parse_args()

    in_dir = args.input_dir
    main(in_dir)