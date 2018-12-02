import argparse
import pandas as pd
import os
import datetime
import copy
from analysis.file_data_util import *
from collections import defaultdict
from utils import file_io
from analysis.file_data_util import *

def posted_recently(collection_date,user_date):
    return datetime.datetime.fromtimestamp(collection_date) - datetime.timedelta(days=60) <= datetime.datetime.fromtimestamp(user_date)




def filter_inactive_users(dates_tweets):
    users = dates_tweets.keys()
    dates = defaultdict(int)
    most_recent = 0

    print('Assembling dates dict to filter users with')

    for u in users:
        for dt in dates_tweets[u]:
            created_at, text = dt
            u_date = dates[u]
            if created_at > u_date:
                dates[u] = created_at
            if created_at > most_recent:
                most_recent = created_at

    recent_users = 0

    print('Filtering users with {} most recent'.format(most_recent))
    all_users = copy.deepcopy(list(dates_tweets.keys()))
    for user in all_users:
        u_date = dates[user]
        if posted_recently(most_recent, u_date):
            del dates[user]
            recent_users += 1
    print('{} users after filtering'.format(recent_users))

    return dates.keys()

def main(in_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    static_info['fold'] = float('nan')
    static_info['posted_recently'] = 'y'

    json_dir = os.path.join(in_dir,'json')
    dates_tweets = load_dates_tweets(json_dir,'dates_tweets')

    print('writing train dev test data to df')

    train, dev, test = train_dev_test(in_dir,dates_tweets.keys(),0.2,0.2)
    for u in train:
        static_info.loc[static_info['user_id'] == int(u), 'fold'] = 'train'
    for u in dev:
        static_info.loc[static_info['user_id'] == int(u), 'fold'] = 'dev'
    for u in test:
        static_info.loc[static_info['user_id'] == int(u), 'fold'] = 'test'

    print('filtering inactive users')
    inactive_users = filter_inactive_users(dates_tweets)

    print('writing inactive users to df')
    for u in inactive_users:
        static_info.loc[static_info['user_id'] == int(u),'posted_recently'] = 'n'

    print('saving df')
    static_info.to_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))

if __name__ == '__main__':
    """
    add 'fold' and 'posted_recently' columns to static_info    
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