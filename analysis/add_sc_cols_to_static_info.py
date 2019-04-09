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

def calculate_statuses_change(static_info, in_dir, us):
    df = pd.read_csv(os.path.join(in_dir, 'info/user_info_dynamic.tsv.gz'),sep='\t',compression='gzip')
    pcs = {}
    for u in us:
        start = static_info.loc[static_info['user_id'] == int(u)]['statuses_count'].values[0]
        tss = df.loc[df['user_id']==u]['timestamp'].values
        timestamp = max(tss)
        statuses_count = df.loc[(df['user_id']==u) & (df['timestamp']==timestamp)]['statuses_count'].values[0]
        pcs[u] = [start, timestamp, statuses_count]
    for u in pcs.keys():
        start, mx, sc = pcs[u]
        pcs[u] = (sc-start, sc)


    return pcs

def main(in_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    #static_info['percent_change'] = float('nan')

    users = static_info.loc[static_info['classify_account-mace_label'] == 'promoting'][
        'user_id'].dropna().unique().tolist()

    print('Calculating statuses in collection window and adding to df')
    pcs = calculate_statuses_change(static_info, in_dir, users)

    for u, (sc_tw, sc) in pcs.items():

        static_info.loc[static_info['user_id'] == int(u), 'statuses_tw'] = sc_tw
        static_info.loc[static_info['user_id'] == int(u), 'end_statuses'] = sc

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