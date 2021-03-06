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
    df = pd.read_csv(os.path.join(in_dir, 'info/user_info_dynamic.tsv.gz'),sep='\t',compression='gzip')
    pcs = {}
    for u in us:
        start = static_info.loc[static_info['user_id'] == int(u)]['followers_count'].values[0]
        tss = df.loc[df['user_id']==u]['timestamp'].values
        timestamp = max(tss)
        followers_count = df.loc[(df['user_id']==u) & (df['timestamp']==timestamp)]['followers_count'].values[0]
        pcs[u] = [start, timestamp, followers_count]
    for u in pcs.keys():
        start, mx, fc = pcs[u]
        pcs[u] = (fc-start)/start


    return pcs

def main(in_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    #static_info['percent_change'] = float('nan')
    static_info['pc_percentile'] = float('nan')
    users = static_info.loc[static_info['classify_account-mace_label'] == 'promoting'][
        'user_id'].dropna().unique().tolist()

    print('Calculating percent change and adding to df')
    pcs = calculate_percent_change(static_info, in_dir,users)

    percentiles = [20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    percentile_vals = [0.00000000e+00, 9.95256917e-02, 2.22222222e-01, 4.28571429e-01,
     6.79021879e-01, 1.09976409e+00, 1.81931818e+00, 4.20000000e+00,
     7.84366197e+00, 2.96772269e+01, 4.88000000e+02]
    for u, pc in pcs.items():
        static_info.loc[static_info['user_id'] == int(u), 'percent_change'] = pc
        for p, v in zip(percentiles, percentile_vals):
            if pc <= v:
                static_info.loc[static_info['user_id'] == int(u), 'pc_percentile'] = p
                break

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