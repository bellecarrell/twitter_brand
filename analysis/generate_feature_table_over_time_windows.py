import argparse
from collections import defaultdict
import csv
import datetime
import gzip
import json
import multiprocessing as mp
import numpy as np
import os
import pytz
import time
import pandas as pd


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
- Tweet was posted on the most recent Friday (binary, null if Friday not in preceding window) | +
- % messages occurring on Friday ($\in [0,1]$, null if no Friday in preceding window) | +
- % last Fridays with at least one tweet ($\in [0,1]$, null if no Friday) | +
- % messages posted between 9-12 ET ([0,1]) | +
- % messages posted between 9-12 local time (TODO: how to find local time) | +
- % messages that are RTs ([0,1]) | +
- % days with >= 1 message ([0,1]) | +
- mean # tweets/day (float) | +
- message count/day distribution (list[float]) | +
- message count/hour distribution (list[float]) | +
- entropy of messages/day distribution, add-1 smoothed (float) | +
- entropy of messages/hour distribution, add-0.1 smoothed (float) | +
- maximum messages/hour (int) | +
- mean # new friends/day (float) | +
- mean # RTs/day (float) | +
- % messages that are replies ([0,1]) | +
- mean # replies/day (float) | +
- mean @-mentions per tweet (float) | +
- % messages containing user mention ([0,1]) | +
- % messages with pointer to URL | +
- % messages with URL to user's blog | +
- Interactivity -- user falls in top 50% of for # of tweets/day, % replies,
  mean # user mentions/tweets (binary)
- % messages with positive sentiment ([0,1]) | +
- mean sentiment (float) | +
- median sentiment (float) | +
- variance sentiment (float) | +
- topic counts (list[int]) | +
- topic distribution, add-1 smoothed (list[float]) | +
- entropy of topic distribution, add-1 smoothed (float) | +
- % messages tagged with plurality topic, add-1 smoothed ([0,1]) | +
'''


### DVs to extract ###
'''
- % change in follower count | +
- future follower count | +
  + computed from 12pm - 12pm, take nearest sample above current time (within 12 hours, else value is null):
    days in the future {1, 2, 3, 4, 5, 6, 7, 14, 21, 28}
'''


def extract_tweet_level_features_from_row(tweet_row):
    curr_dt = tweet_row['curr_datetime']
    
    # Friday features
    is_friday = (curr_dt.weekday() == 4)
    
    # time of posting
    is_9to12_utc = 9  <= curr_dt.hour <= 12
    is_9to12_est = 13 <= curr_dt.hour <= 16
    is_9to12_local = -1  # TODO: how to identify local time
    
    # RTs
    is_rt = 1. * tweet_row['rt']
    
    # replies
    is_reply = 1. * tweet_row['reply']
    
    # mentions
    num_mentions = 1. * tweet_row['mention_count']
    has_mention = 1. if tweet_row['mention_count'] > 0 else 0.
    
    # urls
    num_urls = 1. * tweet_row['url']
    num_personal_urls = -1  # TODO: identify when linking to personal website
    
    # diff granularities of time
    norm_day = datetime.datetime(curr_dt.year, curr_dt.month, curr_dt.day, 12)
    if curr_dt.hour < 12:
        norm_day += datetime.timedelta(days=1)
    
    curr_hour = (curr_dt.year, curr_dt.month, curr_dt.day, curr_dt.hour)
    
    # topics
    topic_wts = json.loads(tweet_row['topics_per_tweet'])
    if max(topic_wts):
        max_topic = None
    else:
        max_topic = np.argmax(topic_wts)
    
    # sentiment
    is_positive_sentiment = 1. * (tweet_row['tweet_sentiment_score'] > 0.)

    DERIVED_FEATURES = [is_friday, is_9to12_utc, is_9to12_est, is_9to12_local,
                        is_rt, is_reply, num_mentions, has_mention, num_urls,
                        num_personal_urls, norm_day, curr_hour, max_topic,
                        is_positive_sentiment]
    
    return DERIVED_FEATURES


def get_num_past_fridays(day, tw):
    curr_weekday = day.weekday()
    
    num_fridays = tw // 7
    remainder = tw % 7
    
    if ((curr_weekday > 4) and ((curr_weekday - remainder) <= 4)) or \
            ((tw > curr_weekday) and (((curr_weekday - tw) % 7) < 4)):
        num_fridays += 1
    last_friday = day - datetime.timedelta(days=(curr_weekday - 4) % 7)
    
    return num_fridays, last_friday


def entropy(bin_counts, num_bins, delta=0.0):
    num_nz_bins = len(bin_counts)
    num_z_bins = num_bins - num_nz_bins
    denom = sum(bin_counts) + delta * num_bins
    
    entropy = 0.0
    for count in bin_counts:
        p = (count + delta) / denom
        entropy -= p * np.log(p)
    
    # cover the zero-count bins
    p_z = delta / denom
    entropy -= num_z_bins * p_z * np.log(p_z)
    
    return entropy


def aggregate_tweet_level_features(subset_df, day, tw):
    ''' Compute hypotheses/controls for a single <user, day, window> triple. '''

    pre = 'DerivedFeature-'

    num_msgs = subset_df.shape[0]
    
    # timing

    ## friday features -- null if no preceding Friday
    has_tweet_last_friday = None
    pct_msgs_on_friday = None
    pct_fridays_with_tweet = None

    num_fridays, last_friday = get_num_past_fridays(day, tw)
    if num_fridays > 0:  # was there a Friday in this window?
        has_tweet_last_friday = 1. * (subset_df[pre + 'NORM_DAY'] == last_friday).any()
        pct_msgs_on_friday = subset_df[subset_df[pre + 'IS_FRIDAY']].shape[0] / float(num_msgs)
        pct_fridays_with_tweet = len(subset_df.loc[subset_df[pre + 'IS_FRIDAY'],
                                                   pre + 'NORM_DAY'].unique()) / float(num_fridays)

    pct_msgs_9to12_utc = subset_df[pre + 'IS_9TO12_UTC'].sum() / float(num_msgs)
    pct_msgs_9to12_et = subset_df[pre + 'IS_9TO12_EST'].sum() / float(num_msgs)
    pct_msgs_9to12_local = subset_df[pre + 'IS_9TO12_LOCAL'].sum() / float(num_msgs)
    pct_days_with_some_msg = len(subset_df[pre + 'NORM_DAY'].unique()) / float(tw)
    mean_tweets_per_day = num_msgs / float(tw)
    
    count_per_day_df = subset_df.groupby('NORM_DAY')['tweet_id'].count()
    msg_count_per_day_dist = dict(zip(count_per_day_df.index, count_per_day_df.values))
    count_per_hour_df = subset_df.groupby(pre + 'CURR_HOUR')['tweet_id'].count()
    msg_count_per_hour_dist = dict(zip(count_per_hour_df.index, count_per_hour_df.values))
    msg_per_day_entropy_add1 = entropy(count_per_day_df.values, tw, delta=1.0)
    msg_per_hour_entropy_add01 = entropy(count_per_hour_df.values, 24 * tw, delta=0.1)
    max_msgs_per_hour = max(count_per_hour_df.values)
    
    # engagement features
    pct_msgs_rt = subset_df[pre + 'IS_RT'].sum() / float(num_msgs)
    mean_rts_per_day = subset_df[pre + 'IS_RT'].sum() / float(tw)
    pct_msgs_replies = subset_df[pre + 'IS_REPLY'].sum() / float(num_msgs)
    mean_replies_per_day = subset_df[pre + 'IS_REPLY'].sum() / float(tw)
    mean_mentions_per_tweet = subset_df[pre + 'NUM_MENTIONS'].sum() / float(num_msgs)
    mean_msgs_with_mention = (subset_df[pre + 'NUM_MENTIONS'] > 0.).sum() / float(num_msgs)
    
    # content
    pct_msgs_with_url = (subset_df[pre + 'NUM_URLS'] > 0.).sum() / float(num_msgs)
    shared_url = 1. * (subset_df[pre + 'NUM_URLS'].sum() > 0.)
    pct_msgs_with_personal_url = (subset_df[pre + 'NUM_PERSONAL_URLS'] > 0.).sum() / float(num_msgs)
    shared_personal_url = 1. * (subset_df[pre + 'NUM_PERSONAL_URLS'].sum() > 0.)
    
    # sentiment
    pct_msgs_with_positive_sentiment = subset_df[pre + 'IS_POSITIVE_SENTIMENT'].sum() / float(num_msgs)
    median_sentiment = subset_df['tweet_sentiment_score'].median()
    mean_sentiment = subset_df['tweet_sentiment_score'].mean()
    std_sentiment = subset_df['tweet_sentiment_score'].std()
    
    # topic
    num_topics = 50  # 50-dimensional NNMF model
    topic_count_df = subset_df.groupby(pre + 'MAX_TOPIC')['tweet_id'].count()
    count_per_topic = dict(zip(topic_count_df.index, topic_count_df.values))
    topic_dist_entropy_add1 = entropy(topic_count_df.values, num_topics, delta=1.0)
    topic_dist_entropy_add01 = entropy(topic_count_df.values, num_topics, delta=0.1)
    plurality_topic = topic_count_df.idxmax()
    pct_msgs_with_plurality_topic = topic_count_df.max() / float(num_msgs)
    pct_msgs_with_plurality_topic_add1 = (topic_count_df.max()+1.) / (float(num_msgs) + 50.)

    agg_row = [num_msgs, has_tweet_last_friday, pct_msgs_on_friday, pct_fridays_with_tweet,
               pct_msgs_9to12_utc, pct_msgs_9to12_et, pct_msgs_9to12_local,
               pct_days_with_some_msg, mean_tweets_per_day, msg_count_per_day_dist,
               msg_count_per_hour_dist, msg_per_day_entropy_add1, msg_per_hour_entropy_add01,
               max_msgs_per_hour, pct_msgs_rt, mean_rts_per_day, pct_msgs_replies,
               mean_replies_per_day, mean_mentions_per_tweet, mean_msgs_with_mention,
               pct_msgs_with_url, shared_url, pct_msgs_with_personal_url, shared_personal_url,
               pct_msgs_with_positive_sentiment, median_sentiment, mean_sentiment,
               std_sentiment, num_topics, count_per_topic, topic_dist_entropy_add1,
               topic_dist_entropy_add01, plurality_topic, pct_msgs_with_plurality_topic,
               pct_msgs_with_plurality_topic_add1]
    
    return agg_row


def collect_features_from_user_timeline_table(timeline_path, tracked_uids, out_path,
                                              tws=[1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
                                              min_timestamp=datetime.datetime(2018, 10, 14, 12),
                                              max_timestamp=datetime.datetime(2019, 4, 5, 12)):
    one_day = datetime.timedelta(days=1)
    
    # read in timeline, add timestamp to each tweet
    timeline_df = pd.read_table(timeline_path)
    timeline_df['curr_datetime'] = timeline_df['created_at'].map(lambda x: datetime.datetime.fromtimestamp(x))
    timeline_df['curr_day'] = timeline_df['curr_datetime'].map(lambda x: (x.year, x.month, x.day))
    
    # restrict to users in study
    timeline_df = timeline_df[timeline_df['user_id'].isin(tracked_uids)]
    
    # restrict to tweets during study
    timeline_df = timeline_df.loc[timeline_df['curr_datetime'] >= (min_timestamp -
                                                                   datetime.timedelta(days=max(tws)))]
    timeline_df = timeline_df.loc[timeline_df['curr_datetime'] < max_timestamp]
    
    # list days to build samples for
    sampled_dts = []
    curr_dt = min_timestamp
    while curr_dt <= max_timestamp:
        sampled_dts.append(curr_dt)
        curr_dt += one_day
    
    # extract new columns
    DERIVED_COLUMNS = ['IS_FRIDAY', 'IS_9TO12_UTC', 'IS_9TO12_EST', 'IS_9TO12_LOCAL',
                       'IS_RT', 'IS_REPLY', 'NUM_MENTIONS', 'HAS_MENTION', 'NUM_URLS',
                       'NUM_PERSONAL_URLS', 'NORM_DAY', 'CURR_HOUR', 'MAX_TOPIC',
                       'IS_POSITIVE_SENTIMENT']
    DERIVED_COLUMNS = ['DerivedFeature-' + c for c in DERIVED_COLUMNS]
    
    df_map = {c: [] for c in DERIVED_COLUMNS}
    
    # mapping from aggregation key to row IDs.
    key_to_rowindexes = {}
    start = time.time()
    
    for ridx, (row_id, r) in enumerate(timeline_df.iterrows()):
        if not ((ridx+1) % 1000):
            print('({:d}s) Row {:.2f}M/{:.2f}M ; Extracting tweet features'.format(
                    int(time.time() - start), (ridx+1)/1000000., timeline_df.shape[0]/1000000.)
            )
        
        row = extract_tweet_level_features_from_row(r)
        for c, v in zip(DERIVED_COLUMNS, row):
            df_map[c].append(v)
        
        # group tweets together by <user, time window, date>
        
        uid = r['user_id'],
        cday = r['curr_day']
        
        norm_day = r['DerivedFeature-NORM_DAY']
        #norm_day = datetime.datetime(year=cday[0], month=cday[1], day=cday[2], hour=12)
        if r['curr_datetime'].hour < 12:
            norm_day -= one_day
        
        for tw in tws:
            for lag in range(1, tw+1):
                day_bin = norm_day + (lag * one_day)
                key = (uid, tw, day_bin)
                
                if key not in key_to_rowindexes:
                    key_to_rowindexes = []
                
                key_to_rowindexes[key].append(row_id)
    
    # join new columns with original dataframe
    new_col_df = pd.DataFrame(df_map, index=timeline_df.index)
    timeline_df = pd.merge(timeline_df, new_col_df)
    
    # aggregate hypotheses + control features over recent past

    AGG_COLUMNS = ['NUM_MSGS', 'HAS_TWEET_LAST_FRIDAY', 'PCT_MSGS_ON_FRIDAY', 'PCT_FRIDAYS_WITH_TWEET',
                   'PCT_MSGS_9TO12_UTC', 'PCT_MSGS_9TO12_ET', 'PCT_MSGS_9TO12_LOCAL',
                   'PCT_DAYS_WITH_SOME_MSG', 'MEAN_TWEETS_PER_DAY', 'MSG_COUNT_PER_DAY_DIST',
                   'MSG_COUNT_PER_HOUR_DIST', 'MSG_PER_DAY_ENTROPY_ADD1', 'MSG_PER_HOUR_ENTROPY_ADD01',
                   'MAX_MSGS_PER_HOUR', 'PCT_MSGS_RT', 'MEAN_RTS_PER_DAY', 'PCT_MSGS_REPLIES',
                   'MEAN_REPLIES_PER_DAY', 'MEAN_MENTIONS_PER_TWEET', 'MEAN_MSGS_WITH_MENTION',
                   'PCT_MSGS_WITH_URL', 'SHARED_URL', 'PCT_MSGS_WITH_PERSONAL_URL', 'SHARED_PERSONAL_URL',
                   'PCT_MSGS_WITH_POSITIVE_SENTIMENT', 'MEDIAN_SENTIMENT', 'MEAN_SENTIMENT',
                   'STD_SENTIMENT', 'NUM_TOPICS', 'COUNT_PER_TOPIC', 'TOPIC_DIST_ENTROPY_ADD1',
                   'TOPIC_DIST_ENTROPY_ADD01', 'PLURALITY_TOPIC', 'PCT_MSGS_WITH_PLURALITY_TOPIC',
                   'PCT_MSGS_WITH_PLURALITY_TOPIC_ADD1']
    agg_pre = 'past-'
    AGG_COLUMNS = [agg_pre + c for c in AGG_COLUMNS]
    AGG_COLUMNS = ['user_id', 'sampled_datetime', 'history_agg_window'] + AGG_COLUMNS
    
    agg_rows = []
    for key_index, (key, indexes) in enumerate(key_to_rowindexes.items()):
        if not ((key_index + 1) % 100):
            print('({:d}) {:.1f}K/{:.1f}K ; Aggregating'.format(int(time.time() - start),
                                                                (key_index+1) / 1000.,
                                                                len(key_to_rowindexes) / 1000.))
        
        agg_row = list(key) + aggregate_tweet_level_features(timeline_df.iloc[indexes], key[2], key[1])
        agg_rows.append(agg_row)
    
    agg_df = pd.DataFrame(agg_rows, columns=AGG_COLUMNS)
    
    agg_df.to_csv(out_path, compression='gzip', sep='\t', header=True, index=False)


def collect_dvs_from_user_info_table(dynamic_user_info_path, tracked_uids, out_path,
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
    COLUMNS += ['current-' + k for k in ['follower_count', 'log_follower_count',
                                         'friend_count', 'log_friend_count',
                                         'list_count', 'user_impact_score']]
    COLUMNS += ['future-horizon{}-'.format(tw) + k for tw in tws
                for k in ['follower_count', 'log_follower_count', 'pct_change_follower_count',
                          'user_impact_score']]
    COLUMNS += ['past-' + k for k in ['mean_friendsPerDay', 'mean_followersPerDay']]
    
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
    
    start = time.time()
    
    uid_uniq = df['user_id'].unique()
    for uid_idx, uid in enumerate(uid_uniq):
        feature_rows = []
        user_df = df[df['user_id'] == uid]  # restrict to a single user and extract samples for this one person
        
        for dt_idx, curr_dt in enumerate(sampled_dts):
            if not (dt_idx % 10):
                print('({}s) Starting user {}/{}, sample {}/{}'.format(int(time.time() - start),
                                                                       uid_idx+1,
                                                                       len(uid_uniq),
                                                                       dt_idx+1,
                                                                       len(sampled_dts)))
            
            # extract current day features
            curr_day_idx = str((curr_dt.year, curr_dt.month, curr_dt.day))
            
            try:
                curr_df = user_df.loc[curr_day_idx]
            except Exception as ex:
                # user has no user information sampled on this day, skip
                continue
            
            # pick value closest to 12pm
            if len(curr_df.shape) > 1:
                curr_df['distfrom12'] = (curr_df['curr_datetime'] -
                                         curr_dt).map(lambda x: abs(x.total_seconds()))
                min_row = curr_df.iloc[curr_df['distfrom12'].values.argmin()]
            else:
                min_row = curr_df
            
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

                fut_idx_dt = curr_dt + datetime.timedelta(days=horizon)
                fut_idx = str((fut_idx_dt.year, fut_idx_dt.month, fut_idx_dt.day))
                
                try:
                    fut_df = user_df.loc[fut_idx]
                except Exception as ex:
                    # we are missing samples for this day, insert null values
                    future_vals += [None, None, None, None]
                    continue
                
                # pick value closest to 12pm
                if len(fut_df.shape) > 1:
                    fut_df['distfrom12'] = (fut_df['curr_datetime'] -
                                            fut_idx_dt).map(lambda x: abs(x.total_seconds()) )
                    min_row = fut_df.iloc[fut_df['distfrom12'].values.argmin()]
                else:
                    min_row = fut_df
                
                follower_count = min_row['followers_count']
                friend_count = min_row['friends_count']
                user_impact = np.log( (1. + min_row['listed_count']) *
                                      (1. + follower_count)**2. /
                                       (1. + friend_count) )
                
                # add small value to avoid inf if user had zero followers previously
                pct_follower_change = 100. * ((follower_count + 0.01) / (curr_vals[0] + 0.01) - 1.)
                
                future_vals += [follower_count,
                                np.log(1. + follower_count),
                                pct_follower_change,
                                user_impact]
            
            # extract past features
            for agg_window in tws:
                tmp_row = [uid, curr_dt, agg_window] + curr_vals + future_vals
                
                past_idx_dt = curr_dt-datetime.timedelta(days=agg_window)
                past_idx = str((past_idx_dt.year, past_idx_dt.month, past_idx_dt.day))
                
                try:
                    past_df = user_df.loc[past_idx]
                except Exception as ex:
                    # we are missing samples for this day, insert null
                    feature_rows.append( tmp_row + [None, None] )
                    continue
                
                # pick value closest to 12pm
                if len(past_df.shape) > 1:
                    past_df['distfrom12'] = (past_df['curr_datetime'] - past_idx_dt).map(
                            lambda x: abs(x.total_seconds())
                    )
                    min_row = past_df.iloc[past_df['distfrom12'].values.argmin()]
                else:
                    min_row = past_df
                
                follower_count = min_row['followers_count']
                friend_count = min_row['friends_count']
                
                mean_followers_per_day = (curr_vals[0] - follower_count) / float(agg_window)
                mean_friends_per_day = (curr_vals[2] - friend_count) / float(agg_window)
                
                # add another sample to the table
                feature_rows.append( tmp_row + [mean_followers_per_day, mean_friends_per_day] )
        
        all_feature_rows += feature_rows
        
        print('({}s) Extracted total of {} samples for user {}/{}'.format(int(time.time() - start),
                                                                          len(feature_rows),
                                                                          uid_idx,
                                                                          len(uid_uniq)))
    
    extracted_features = pd.DataFrame(all_feature_rows, columns=COLUMNS)
    
    extracted_features.to_csv(out_path, compression='gzip', sep='\t', header=True, index=False)


def extract_net_features(promoting_users, promoting_user_subsets, out_dir):
    num_procs = len(promoting_user_subsets)
    net_out_paths = [os.path.join(out_dir, 'net_features.{}.tsv.gz'.format(i)) for i in range(num_procs)]
    
    dynamic_info_path = os.path.join(in_dir, 'info/user_info_dynamic.tsv.gz')
    
    # extract network features for each user and sample date
    if num_procs == 1:
        collect_dvs_from_user_info_table(dynamic_info_path,
                                         set(promoting_users),
                                         net_out_paths[0],
                                         tws=[1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
                                         min_timestamp=datetime.datetime(2018, 10, 14, 12),
                                         max_timestamp=datetime.datetime(2019, 4, 5, 12))
    else:
        procs = [mp.Process(target=collect_dvs_from_user_info_table,
                            args=(dynamic_info_path, set(p_users), op))
                 for p_users, op in zip(promoting_user_subsets, net_out_paths)]
        
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    
    # concatenate network features for all users
    joined_net_out_path = os.path.join(out_dir, 'net_features.joined.tsv.gz')
    joined_df = None
    for op in net_out_paths:
        if joined_df is None:
            joined_df = pd.read_table(op)
        else:
            joined_df = pd.concat([joined_df, pd.read_table(op)])
    
    joined_df.sort_values(by=['history_agg_window', 'user_id', 'sampled_datetime'], inplace=True)
    joined_df.to_csv(joined_net_out_path, compression='gzip', sep='\t', header=True, index=False)


def extract_text_features(promoting_users, promoting_user_subsets, out_dir):
    num_procs = len(promoting_user_subsets)
    text_out_paths = [os.path.join(out_dir, 'text_features.{}.tsv.gz'.format(i)) for i in range(num_procs)]
    
    #timeline_path = os.path.join(in_dir, 'timeline/user_tweets.noduplicates.tsv.gz')
    timeline_path = os.path.join(in_dir, 'timeline/promoting_user_timeline.noduplicates.withTopicAndSentiment.tsv.gz')
    
    # extract text features for each user and sample date
    if num_procs == 1:
        collect_features_from_user_timeline_table(timeline_path,
                                         set(promoting_users),
                                         text_out_paths[0],
                                         tws=[1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
                                         min_timestamp=datetime.datetime(2018, 10, 14, 12),
                                         max_timestamp=datetime.datetime(2019, 4, 5, 12))
    else:
        procs = [mp.Process(target=collect_features_from_user_timeline_table,
                            args=(timeline_path, set(p_users), op))
                 for p_users, op in zip(promoting_user_subsets, text_out_paths)]
        
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    
    # concatenate text features for all users
    joined_text_out_path = os.path.join(out_dir, 'text_features.joined.tsv.gz')
    joined_df = None
    for op in text_out_paths:
        if joined_df is None:
            joined_df = pd.read_table(op)
        else:
            joined_df = pd.concat([joined_df, pd.read_table(op)])
    
    joined_df.sort_values(by=['history_agg_window', 'user_id', 'sampled_datetime'], inplace=True)
    
    # compute interactivity feature
    
    agg_pre = 'past-'
    grped_by_tw = joined_df.groupby(['sampled_datetime', 'history_agg_window'], as_index=False)
    # calculate mean tweets/day
    mean_tpd = grped_by_tw[agg_pre + 'MEAN_TWEETS_PER_DAY'].mean()
    # """" % replies
    mean_pr = grped_by_tw[agg_pre + 'PCT_MSGS_REPLIES'].mean()
    # """" avg mentions/tweet
    mean_mpt = grped_by_tw[agg_pre + 'MEAN_MENTIONS_PER_TWEET'].mean()
    
    joined_df_twidx = joined_df.set_index(['sampled_datetime', 'history_agg_window'], drop=False)
    joined_df_twidx['avgAcrossUsers-MEAN_TWEETS_PER_DAY'] = mean_tpd
    joined_df_twidx['avgAcrossUsers-MEAN_PCT_REPLIES'] = mean_pr
    joined_df_twidx['avgAcrossUsers-MEAN_MENTIONS_PER_TWEET'] = mean_mpt
    
    joined_df_twidx.to_csv(joined_text_out_path, compression='gzip', sep='\t', header=True, index=False)


def main(in_dir, out_dir, num_procs, max_users):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    promoting_users = static_info.loc[
                        static_info['classify_account-mace_label'] == 'promoting'
                      ]['user_id'].dropna().unique().tolist()
    
    if max_users > 0:
        promoting_users = promoting_users[:max_users]
    
    promoting_user_subsets = [[p_user for i, p_user
                               in enumerate(promoting_users)
                               if ((i % num_procs) == j)] for j in range(num_procs)]
    
    ## adrian : already extracted these features to here
    ## /exp/abenton/twitter_brand/TEST_OUTPUT/net_features.joined.tsv.gz
    #extract_net_features(promoting_users, promoting_user_subsets, out_dir)

    extract_text_features(promoting_users, promoting_user_subsets, out_dir)


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
    parser.add_argument('--num_procs', required=True, type=int,
                        default=1, help='number of processes working in parallel')
    parser.add_argument('--max_users', required=False, type=int,
                        default=-1, help='maximum number of users to process')
    args = parser.parse_args()
    
    in_dir = args.input_dir
    out_dir = args.out_dir
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    main(in_dir, out_dir, args.num_procs, args.max_users)
