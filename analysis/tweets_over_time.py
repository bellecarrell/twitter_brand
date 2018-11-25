'''
Count # of tweets made over time.

Adrian Benton
11/25/2018
'''

import argparse
import datetime
import pandas as pd


def main(tweet_path, user_label_path):
    tweet_df = pd.read_table(tweet_path, sep=',')
    user_df = pd.read_table(user_label_path)
    
    prom_user_df = user_df[user_df['classify_account-mace_label']=='promoting']
    promoting_users = set(prom_user_df['user_id'].tolist())

    main_spec_df = prom_user_df[['user_id', 'classify_account-mace_label']]
    
    filt_tweet_df = tweet_df[tweet_df['user_id'].isin(promoting_users)]
    filt_tweet_df = filt_tweet_df.join(main_spec_df, on='user_id',
                                       how='left', lsuffix='', rsuffix='',
                                       sort=True)
    
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='count tweets made over time'
    )
    parser.add_argument('--tweet_path', required=True,
                        help='path with all tweets')
    parser.add_argument('--user_label_path', required=True,
                        help='path to MTurk-labeled users')
    args = parser.parse_args()
    
    main(args.tweet_path, args.user_label_path)
