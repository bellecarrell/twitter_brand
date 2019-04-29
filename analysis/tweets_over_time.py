xr'''
Count # of tweets made over time.

Adrian Benton
11/25/2018
'''

import argparse
import datetime
import pandas as pd


def main(tweet_path, user_label_path, out_path):
    tweet_df = pd.read_table(tweet_path, sep=',')
    user_df = pd.read_table(user_label_path)

    # restrict to self-promoting
    prom_user_df = user_df[user_df['classify_account-mace_label']=='promoting']
    promoting_users = set(prom_user_df['user_id'].tolist())
    
    main_spec_df = prom_user_df[['user_id', 'category_most_index-mace_label']]
    u_to_lab = {u: lab for u, lab
            in zip(main_spec_df['user_id'].tolist(),
                   main_spec_df['category_most_index-mace_label'].tolist())}
    
    filt_tweet_df = tweet_df[tweet_df['user_id'].isin(promoting_users)]
    filt_tweet_df['main_specialization'] = [u_to_lab[u]
                                            for u
                                            in filt_tweet_df['user_id']]

    # map unix timestamps to datetimes
    filt_tweet_df['datetime'] = filt_tweet_df['created_at'].map(
        lambda x: datetime.datetime.fromtimestamp(x)
    )
    filt_tweet_df['month'] = filt_tweet_df['datetime'].map(
        lambda x: (x.year, x.month)
    )
    
    print('Tweets collected from "{}" to "{}"'.format(
        filt_tweet_df['datetime'].min(), filt_tweet_df['datetime'].max())
    )

    # Total tweets per month
    t_per_mon = filt_tweet_df.groupby('month')['user_id'].count()
    
    # Unique users per month
    u_per_mon = filt_tweet_df.groupby('month')['user_id'].unique().map(len)

    all_mon_idx = u_per_mon.index

    # tweets per user per month
    counts_per_user = {}
    for u in filt_tweet_df['user_id'].unique():
        u_tweet_subset = filt_tweet_df[filt_tweet_df['user_id']==u]
        counts_per_user['USER_{}'.format(u)] = \
            u_tweet_subset.groupby('month')['user_id'].count().reindex(
                u_per_mon.index, fill_value=0
            )

    # users per specialization per month
    counts_per_spec = {}
    for s in filt_tweet_df['main_specialization'].unique():
        s_tweet_subset = filt_tweet_df[filt_tweet_df['main_specialization']==s]
        counts_per_spec['SPEC_{}'.format(s)] = \
            s_tweet_subset.groupby('month')['user_id'].unique().map(len).reindex(
                u_per_mon.index, fill_value=0
            )
    
    df = pd.DataFrame(dict(list(counts_per_spec.items()) +
                           list(counts_per_user.items()) +
                           [('UNIQUE_USERS', u_per_mon),
                            ('NUM_TWEETS', t_per_mon)]))
    
    df.to_csv(out_path, sep='\t', index=True, header=True)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='count tweets made over time'
    )
    parser.add_argument('--tweet_path', required=True,
                        help='path with all tweets')
    parser.add_argument('--user_label_path', required=True,
                        help='path to MTurk-labeled users')
    parser.add_argument('--out_path', required=True,
                        help='table of counts per month')
    args = parser.parse_args()
    
    main(args.tweet_path, args.user_label_path, args.out_path)
