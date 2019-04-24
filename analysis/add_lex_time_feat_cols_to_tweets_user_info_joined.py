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
import datetime

def on_friday(dt):
    day = dt.weekday()
    if day == 4:
        return True
    else:
        return False

def main(in_file):
    df = pd.read_csv(in_file,sep='\t',compression=gzip)

    df['on_friday'] = 0
    df['friday_9_10'] = 0
    df['10_noon'] = 0
    df['RT'] = 0
    df['@_mention'] = 0
    df['url'] = 0

    for index, row in df.itterrows():
        dt = datetime.datetime.fromtimestamp(row['created_at'])
        if on_friday(dt):
            row['on_friday'] = 1
            if 9 <= dt.hour <= 10:
                row['on_friday'] = 1
        if 10 <= dt.hour <= 12:
            row['10_noon'] = 1

        t = df['text']
        if t.startswith('RT'):
            df['RT'] = 1
        if '@' in t:
            df['@'] = 1
        if 'http' in t:
            df['url'] = 1

    print('saving df')
    df.to_csv('/exp/abenton/twitter_brand_workspace_20190417/promoting_user_tweets.merged_with_user_info.noduplicates.added_lex_feat.tsv.gz')

if __name__ == '__main__':
    """
    add 'percent_change' column to static_info    
    """

    parser = argparse.ArgumentParser(
        description='calculate initial statistics for "shotgun" analysis of user data.'
    )
    parser.add_argument('--input_file', required=True,
                        dest='input_file', metavar='INPUT_FILE',
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs. should have vocab')
    args = parser.parse_args()

    in_file = args.input_file
    main(in_file)