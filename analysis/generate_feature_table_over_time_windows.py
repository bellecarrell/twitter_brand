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
    promoting_users = promoting_users[0]

    tws = [datetime.timedelta(days=1), datetime.timedelta(days=2), datetime.timedelta(days=3), datetime.timedelta(days=4), datetime.timedelta(days=5), datetime.timedelta(days=6),
           datetime.timedelta(days=7), datetime.timedelta(weeks=2), datetime.timedelta(weeks=3), datetime.timedelta(weeks=4)]
    dv_types = [('delta', 'followers_count'), ('percent','followers_count')]
    iv_types = [('average', 'rt')]

    rows = []
    for user in promoting_users:
        tweet_dates = timeline.loc[timeline['user_id']==user]['created_at'].unique().tolist()
        info_dates = info.loc[info['user_id']==user]['timestamp'].unique().tolist()

        for date in tweet_dates:
            for i, window in enumerate(tws):
                end = date + window
                if end <= max(tweet_dates):
                    # compute independent var values
                    for compute, column in iv_types:
                        iv_vals = timeline.loc[(timeline['user_id'] == user) & (date <= timeline['created_at'] <= end)][column].tolist()
                        if compute == 'average':
                            iv_val = sum(iv_vals)/len(iv_vals)
                        iv_type = compute + '_' + column


                        for j, horizon in enumerate(tws):
                            h_start = end
                            h_end = h_start + horizon
                            if datetime.datetime.timestamp(h_end) <= max(info_dates):

                                #compute dependent var values
                                datetime.datetime.timestamp(h_start)
                                u_infos = info.loc[info['user_id']==user]
                                u_infos['date'] = datetime.date
                                #todo: remove once date is added to user_infos table
                                for index, row in u_infos.itterrows():
                                    row['date'] = datetime.datetime.date(row['timestamp'])

                                for compute, column in dv_types:
                                    c_start = u_infos.loc[u_infos['date']==h_start][column]
                                    c_end = u_infos.loc[u_infos['date'] == h_end][column]
                                    if compute == 'delta':
                                        dv_val = c_end - c_start
                                    if compute == 'percent':
                                        dv_val = (c_end - c_start)/c_start
                                    dv_type = compute + '_' + column

                                    row = [user, i, j, date, end, h_start, h_end, iv_type, iv_val, dv_type, dv_val]
                                    rows.append(row)

    ft = pd.DataFrame(rows, columns=['user_id','window_size','horizon_size', 'window_start',
                               'window_stop', 'horizon_start','horizon_stop','iv_type','iv_value', 'dv_type','dv_value'])

    ft.to_csv(os.path.join(out_dir,'feature_table.csv.gz'),compression='gzip')


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