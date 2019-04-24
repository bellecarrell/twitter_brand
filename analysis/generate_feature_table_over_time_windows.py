import pandas as pd
import os
import argparse
import datetime


def main(in_dir, out_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    info = pd.read_table(os.path.join(in_dir, 'info/user_info_dynamic.tsv.gz'))
    print('read dynamic user info')
    timeline = pd.read_table(os.path.join(in_dir, 'timeline/user_tweets.noduplicates.tsv.gz'))
    print('read tweets')
    promoting_users = static_info.loc[
        static_info['classify_account-mace_label'] == 'promoting'
    ]['user_id'].dropna().unique().tolist()
    promoting_users = [promoting_users[0]]
    
    tws = [datetime.timedelta(days=d) for d in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    dv_types = [('delta', 'followers_count'), ('percent', 'followers_count')]
    iv_types = [('average', 'rt')]
    
    EST = datetime.timezone(datetime.timedelta(hours=-5))
    
    rows = []
    
    # AB: Refactored to write one row per <user_id, window_size_days, eval_date> triple.
    # Easier to fit models with this format.
    COLUMNS = ['user_id', 'window_size_days', 'window_start', 'window_stop', 'eval_date']
    COLUMNS += ['IV_{}-{}'.format(iv_agg_name, iv_name) for iv_agg_name, iv_name in iv_types]
    COLUMNS += ['DV_horizon{}_{}-{}'.format(horizon_width.days, dv_agg_name, dv_name)
                for horizon_width in tws
                for dv_agg_name, dv_name in dv_types]
    
    for user in promoting_users:
        u_infos = info.loc[info['user_id'] == user]
        u_infos['datetime'] = u_infos['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x, tz=EST))
        u_infos['date'] = u_infos['datetime'].map(lambda x: datetime.date(x.year, x.month, x.day))
        
        # AB: need to iterate over all dates that the user was active, not just the days on which they tweeted
        tweet_dates = timeline.loc[timeline['user_id']==user]['created_at'].unique().tolist()
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
            print('Start gen features for user {} day {}/{}'.format(user, date_idx, len(info_dates)))
            for i, window in enumerate(tws):
                end = date + window
                if end <= max(tweet_dates):
                    row = [user, window.days, date, end, end]
                    
                    # compute independent var values
                    for compute, column in iv_types:
                        iv_vals = timeline.loc[(timeline['user_id'] == user) &
                                               (date <= timeline['created_at'] <= end)][column].tolist()
                        if compute == 'average':
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

    ft = pd.DataFrame(rows, columns=['user_id', 'window_size_days', 'window_start',
                                     'window_stop', 'eval_date',
                                     'iv_type', 'iv_value', 'dv_type', 'dv_value'])
    
    ft.to_csv(os.path.join(out_dir, 'feature_table.csv.gz'), compression='gzip')


if __name__ == '__main__':
    """
    build a vocabulary from corpus and save to file.
    documents per user that are promoting in the static_info file.
    """

    parser = argparse.ArgumentParser(
        description='build vocabulary and extracted features for each tweet'
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