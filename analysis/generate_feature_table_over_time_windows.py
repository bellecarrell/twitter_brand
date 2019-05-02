import pandas as pd
import os
import argparse
import datetime
import pytz
import numpy as np
import time
import gzip
import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
from analysis.category_binary_words import *


def tweets_by_user(f, users):
    """
    Get tweets from the timeline compressed file and place in dict sorted by user.
    :param in_dir: top-level directory containing promoting user dfs
    :return: dict of tweets by user ID
    """
    reader = csv.reader(f)
    tweets = defaultdict(list)
    count = 0

    print('Reading rows from timeline file to filter users')
    for row in reader:
        if count == 0:
            count += 1
            continue
        created_at, user_id = int(created_at), int(user_id)
        if user_id in users:
            tweets[user_id].append(row)

        count += 1
        if count % 100000 == 0:
            print('Read {} rows'.format(count))

    return tweets.keys(), tweets


### Checklist of features to extract.  "| +" indicates ###
### we have extracted this feature, else it is still TODO ###

### Controls to extract ###
'''
- current follower count (raw + log-scaled) | +
- user impact score (log[ (num_lists + 1) * (num_followers + 1)^(2) / (num_friends + 1) ]) | +
- mean # followers gained/day in recent past (momentum over X previous days) | +
- geo_enabled (binary)
- main specialization domain (categorical)
'''

### Hypotheses to extract ###
'''
- Tweet was posted on the most recent Friday (binary, null if Friday not in preceding window)
- % messages occurring on Friday ($\in [0,1]$, null if no Friday in preceding window)
- % last Fridays with at least one tweet ($\in [0,1]$, null if no Friday)
- % messages posted between 9-12 ET ([0,1])
- % messages posted between 9-12 local time (TODO: how to find local time)
- % messages that are RTs ([0,1])
- % days with >= 1 message ([0,1])
- mean # tweets/day (float)
- message count/day distribution (list[float])
- message count/hour distribution (list[float])
- entropy of messages/day distribution, add-1 smoothed (float)
- entropy of messages/hour distribution, add-0.1 smoothed (float)
- maximum messages/hour (int)
- mean # new friends/day (float) | +
- mean # RTs/day (float)
- % messages that are replies ([0,1])
- mean # replies/day (float)
- mean @-mentions per tweet (float)
- % messages containing user mention ([0,1])
- % messages with pointer to URL
- % messages with URL to user's blog
- Interactivity -- user falls in top 50% of for # of tweets/day, % replies,
  mean # user mentions/tweets (binary)
- % messages with positive sentiment ([0,1])
- mean sentiment (float)
- median sentiment (float)
- variance sentiment (float)
- topic counts (list[int])
- topic distribution, add-1 smoothed (list[float])
- entropy of topic distribution, add-1 smoothed (float)
- % messages tagged with plurality topic, add-1 smoother ([0,1])
'''


### DVs to extract ###
'''
- % change in follower count | +
- future follower count | +
  + computed from 12pm - 12pm, take nearest sample above current time (within 12 hours, else value is null):
    days in the future {1, 2, 3, 4, 5, 6, 7, 14, 21, 28}
'''


def collect_dvs_from_user_info_table(dynamic_user_info_path, tracked_uids,
                                     tws=[1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
                                     min_timestamp=datetime.datetime(2018, 10, 14, 12),
                                     max_timestamp=datetime.datetime(2019, 4, 5, 12)):
    '''
    Process dynamic user info file to compute:
    - past {1, 2, 3, 4, 5, 6, 7, 14, 21, 28}
      + mean # friends/day
      + mean # followers/day
    - current
      + raw follower count
      + log follower count
      + raw friend count
      + log friend count
      + number of lists
      + user impact score
    - future {1, 2, 3, 4, 5, 6, 7, 14, 21, 28}
      + raw follower count
      + % change follower count
      + log follower count
      + user impact score
    '''
    
    COLUMNS  = ['user_id', 'sampled_datetime', 'history_agg_window']
    COLUMNS += ['past-' + k for k in ['mean_friendsPerDay', 'mean_followersPerDay']]
    COLUMNS += ['current-' + k for k in ['follower_count', 'log_follower_count',
                                         'friend_count', 'log_friend_count',
                                         'list_count', 'user_impact_score']]
    COLUMNS += ['future-horizon{}-'.format(tw) + k for tw in tws
                for k in ['follower_count', 'log_follower_count', 'pct_change_follower_count',
                          'user_impact_score', 'pct_change_user_impact_score']]
    
    df = pd.read_table(dynamic_user_info_path)
    df = df[df['user_id'].isin(tracked_uids)]  # restrict to users in our sample
    df['curr_datetime'] = df['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x))
    df['curr_day'] = df['curr_datetime'].map(lambda x: (x.year, x.month, x.day))
    df = df.set_index(df['curr_day'].map(str))
    
    # list days to build samples for
    sampled_dts = []
    curr_dt = min_timestamp
    one_day = datetime.timedelta(days=1)
    while curr_dt <= max_timestamp:
        sampled_dts.append(curr_dt)
        curr_dt += one_day
    
    all_feature_rows = []
    
    uid_uniq = df['user_id'].unique()
    for uid_idx, uid in enumerate(uid_uniq):
        feature_rows = []
        user_df = df[df['user_id']==uid]  # restrict to a single user and extract samples for this one person
        
        for curr_dt in sampled_dts:
            # extract current day features
            curr_day_idx = str((curr_dt.year, curr_dt.month, curr_dt.day))
            
            try:
                curr_df = user_df.loc[curr_day_idx]
            except Exception as ex:
                # user has no user information sampled on this day, skip
                continue
            
            # pick value closest to 12pm
            curr_df['distfrom12'] = (curr_df['curr_datetime'] - curr_dt).map(lambda x: abs(x.total_seconds()))
            min_row = curr_df.iloc[curr_df['distfrom12'].values.argmin()]
            
            follower_count = min_row['followers_count']
            friend_count = min_row['friends_count']
            user_impact = np.log( (1. + min_row['listed_count']) *
                                  (1. + follower_count)**2. /
                                  (1. + friend_count) )
            curr_vals = [follower_count, np.log(1. + follower_count),
                         friend_count, np.log(1. + friend_count),
                         min_row['listed_count'], user_impact]
            
            # extract future features
            future_vals = []
            for horizon in tws:
                future_idx_dts = [curr_dt + datetime.timedelta(days=delta) for delta in range(1, horizon+1)]
                future_idxes   = [str((d.year, d.month, d.day)) for d in future_idx_dts]
                
                for fut_idx, fut_idx_dt in zip(future_idxes, future_idx_dts):
                    try:
                        fut_df = user_df.loc[fut_idx]
                    except Exception as ex:
                        # we are missing samples for this day, insert null values
                        future_vals += [None, None, None, None, None]
                        continue
                    
                    # pick value closest to 12pm
                    fut_df['distfrom12'] = (fut_df['curr_datetime'] -
                                            fut_idx_dt).map(lambda x: abs(x.total_seconds()) )
                    min_row = fut_df.iloc[fut_df['distfrom12'].values.argmin()]
                    follower_count = min_row['followers_count']
                    friend_count = min_row['friends_count']
                    user_impact = np.log( (1. + min_row['listed_count']) *
                                          (1. + follower_count)**2. /
                                          (1. + friend_count) )
                    
                    # add small value to avoid inf if user had zero followers previously
                    pct_follower_change = 100.* ((follower_count + 0.01) / (curr_vals[0] + 0.01) - 1.)
                    
                    future_vals += [follower_count, pct_follower_change,
                                    np.log(1. + follower_count), user_impact]
            
            # extract past features
            for agg_window in tws:
                past_idx_dts = [curr_dt-datetime.timedelta(days=delta) for delta in range(1, agg_window+1)]
                past_idxes = [str((d.year, d.month, d.day)) for d in past_idx_dts]
                tmp_row  = [uid, curr_dt, agg_window] + curr_vals + future_vals

                for past_idx, past_idx_dt in zip(past_idxes, past_idx_dts):
                    try:
                        past_df = user_df.loc[past_idx]
                    except Exception as ex:
                        # we are missing samples for this day, insert null
                        feature_rows.append( tmp_row + [None, None] )
                        continue
                    
                    # pick value closest to 12pm
                    past_df['distfrom12'] = (past_df['curr_datetime'] - past_idx_dt).map(
                            lambda x: abs(x.total_seconds())
                    )

                    min_row = past_df.iloc[past_df['distfrom12'].values.argmin()]
                    
                    follower_count = min_row['followers_count']
                    friend_count = min_row['friends_count']
                    
                    mean_followers_per_day = (curr_vals[0] - follower_count) / float(agg_window)
                    mean_friends_per_day = (curr_vals[2] - friend_count) / float(agg_window)
                    
                    # add another sample to the table
                    feature_rows.append( tmp_row + [mean_followers_per_day, mean_friends_per_day] )
        
        all_feature_rows += feature_rows
        print('Extracted total of {} samples for user {}/{}'.format(len(feature_rows),
                                                                    uid_idx,
                                                                    len(uid_uniq)))
    
    extracted_features = pd.DataFrame(all_feature_rows, columns=COLUMNS)
    
    return extracted_features


def main(in_dir, out_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    promoting_users = static_info.loc[
        static_info['classify_account-mace_label'] == 'promoting'
        ]['user_id'].dropna().unique().tolist()
    
    # extract network features for each user and sample date
    net_tbl = collect_dvs_from_user_info_table(os.path.join(in_dir, 'info/user_info_dynamic.tsv.gz'),
                                               set(promoting_users),
                                               tws=[1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
                                               min_timestamp=datetime.datetime(2018, 10, 14, 12),
                                               max_timestamp=datetime.datetime(2019, 4, 5, 12))
    
    import pdb; pdb.set_trace()
    
    info = pd.read_table(os.path.join(in_dir, 'info/user_info_dynamic.tsv.gz'))
    t = gzip.open(os.path.join(in_dir, 'timeline/user_tweets.noduplicates.tsv.gz'), 'rt')

    promoting_users = static_info.loc[
        static_info['classify_account-mace_label'] == 'promoting'
    ]['user_id'].dropna().unique().tolist()
    promoting_users = [promoting_users[0]]
    tweets = tweets_by_user(t,promoting_users)

    tws = [datetime.timedelta(days=d) for d in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    dv_types = [('delta', 'followers_count'), ('percent', 'followers_count')]
    iv_types = [('rate','tweets_day'),('rate','mentions_tweet'),
                ('bool_percent', 'rt'), ('bool_percent', 'mention'), ('bool_percent','url'), ('bool_percent','reply')]
    
    #EST = datetime.datetime.astimezone(datetime.timedelta(hours=-5))
    EST = pytz.timezone('US/Eastern')

    rows = []
    
    # AB: Refactored to write one row per <user_id, window_size_days, eval_date> triple.
    # Easier to fit models with this format.
    COLUMNS = ['user_id', 'window_size_days', 'window_start', 'window_stop', 'eval_date']
    COLUMNS += ['IV_{}-{}'.format(iv_agg_name, iv_name) for iv_agg_name, iv_name in iv_types]
    COLUMNS += ['DV_horizon{}_{}-{}'.format(horizon_width.days, dv_agg_name, dv_name)
                for horizon_width in tws
                for dv_agg_name, dv_name in dv_types]
    tweet_cols = ['tweet_id', 'created_at', 'text', 'user_id', 'mention', 'mention_count', 'url','rt','reply']

    info['datetime'] = datetime.datetime.fromtimestamp(time.time())
    info['date'] = datetime.date(1990, 12, 1)
    for user in promoting_users:
        u_infos = info.loc[info['user_id'] == user]
        print(len(u_infos.index))
        #u_infos['datetime'] = u_infos['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x, tz=EST))
        #u_infos['date'] = u_infos['datetime'].map(lambda x: datetime.date(x.year, x.month, x.day))

        for index, row in u_infos.iterrows():
            row['datetime'] = datetime.datetime.fromtimestamp(row['timestamp'],tz=EST)
            row['date'] = datetime.date(row['datetime'].year, row['datetime'].month,row['datetime'].day)


        # AB: need to iterate over all dates that the user was active, not just the days on which they tweeted
        #tweet_dates = timeline.loc[timeline['user_id']==user]['created_at'].unique().tolist()
        tweet_dates = [t[1] for t in tweets[user]]
        print(len(tweet_dates))
        min_tweet_ts = datetime.datetime.fromtimestamp(min(tweet_dates), tz=EST)
        max_tweet_ts = datetime.datetime.fromtimestamp(max(tweet_dates), tz=EST)
        
        tweet_dates = []
        
        # compute features from 12pm - 12pm each day, Eastern time zone,
        # since that is when we sampled the follower counts
        curr_ts = datetime.datetime(min_tweet_ts.year,
                                    min_tweet_ts.month,
                                    min_tweet_ts.day,
                                    12, 0, 0, 0,
                                    EST)
        while curr_ts < max_tweet_ts:
            tweet_dates.append(curr_ts)
            curr_ts += datetime.timedelta(days=1)
        
        info_dates = info.loc[info['user_id']==user]['timestamp'].unique().tolist()
        max_info_date = max(info_dates)
        
        for date_idx, date in enumerate(tweet_dates):
            for i, window in enumerate(tws):
                end = date + window
                if end <= max(tweet_dates):
                    row = [user, window.days, date, end, end]
                    window_tweets = [t for t in tweets[user] if date <= t[1] <= end]
                    n_tweets = len(window_tweets)

                    #todo: finish # days with tweets
                    #n_days_w_tweet = 0
                    #for d in range(window.days):

                    # compute independent var values
                    for compute, column in iv_types:
                        if column == 'tweets_day':
                            row.append(n_tweets/window.days)
                        if column == 'mentions_tweet':
                            mention_counts = [t[-4] for t in tweets[user]]
                            row.append(sum(mention_counts)/n_tweets)
                        else:
                            iv_vals = [t[tweet_cols.index(column)] for f in tweets[user]]
                            num_tweets = len(iv_vals)
                            if compute == 'bool_percent':
                                iv_val = sum(iv_vals)/len(iv_vals)
                                row.append(iv_val)
                            else:
                                raise Exception('Do not recognize IV aggregation type: "{}"'.format(compute))
                    
                    # compute dependent vars for each horizon
                    for j, horizon in enumerate(tws):
                        h_start_day = datetime.date(end.year, end.month, end.day)
                        h_end_day = h_start_day + horizon
                        if datetime.datetime.timestamp(h_end_day) <= max_info_date:
                            for compute, column in dv_types:
                                c_start = u_infos.loc[u_infos['date'] == h_start_day][column]
                                c_end = u_infos.loc[u_infos['date'] == h_end_day][column]
                                
                                # we sample 2x/day, take last sample as the change
                                c_start = c_start.tolist()[-1]
                                c_end = c_end.tolist()[-1]
                                if compute == 'delta':
                                    dv_val = c_end - c_start
                                elif compute == 'percent':
                                    dv_val = (c_end - c_start)/c_start
                                else:
                                    raise Exception('Do not recognize DV compute: "{}"'.format(compute))
                                row.append(dv_val)
                    
                    rows.append(row)

    ft = pd.DataFrame(rows, columns=COLUMNS)
    ft.to_csv(os.path.join(out_dir, 'feature_table.csv.gz'), compression='gzip')


if __name__ == '__main__':
    """
    Build table of features calculated from user activity (independent variables) 
    and corresponding future change in influence such as follower count (dependent variables)
    """

    parser = argparse.ArgumentParser(
    )
    parser.add_argument('--input_dir', required=True,
                        dest='input_dir', metavar='INPUT_DIR',
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs. should have vocab')
    parser.add_argument('--out_dir', required=True,
                        dest='out_dir', metavar='OUTPUT_PREFIX',
                        help='output directory')
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.out_dir
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    main(in_dir, out_dir)