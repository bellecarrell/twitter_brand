'''
Split by user set and time range into train and test folds.
'''

import pandas as pd
import random


# TODO: update with actual paths and columns
UID_COL = 'user_id'
DATE_COL = 'eval_date'

IN_PATH_FMT        = '/.../dynamic_feature_table_{}-history.csv.gz'
TRAIN_SET_PATH_FMT = '/.../dynamic_feature_table_{}-history.train.csv.gz'
DEV_SET_PATH_FMT   = '/.../dynamic_feature_table_{}-history.dev.csv.gz'
TEST_SET_PATH_FMT  = '/.../dynamic_feature_table_{}-history.test.csv.gz'

PROP_USERS_IN_TRAIN = 0.8  # proportion of users selected for the training set
PROP_TIME_RANGE_IN_TRAIN = 0.6  # proportion of days within full time range for
PROP_TIME_RANGE_IN_DEV = 0.2  # proportion of days to allot for a dev set (tune on the same users in train)


def main(inp, trp, devp, tstp, prop_users, prop_tr_time, prop_dev_time):
    df = pd.read_table(inp, sep=',')
    
    sorted_df = df.sort_values(DATE_COL)
    
    uids = list(sorted_df[UID_COL].unique())
    random.shuffle(uids)
    
    tr_uids = set(uids[:int(prop_users*len(uids))])
    tst_uids = set(uids[int(prop_users*len(uids)):])
    
    dates = sorted(list(sorted_df[DATE_COL].unique()))
    tr_end_date = dates[int(prop_tr_time*len(dates))]
    dev_end_date = dates[int((prop_tr_time+prop_dev_time)*len(dates))]

    tr_df  = sorted_df[(sorted_df[UID_COL].isin(tr_uids)) &
                       (sorted_df[DATE_COL] < tr_end_date)]
    dev_df = sorted_df[(sorted_df[UID_COL].isin(tr_uids)) &
                       (sorted_df[DATE_COL] >= tr_end_date) &
                       (sorted_df[DATE_COL] < dev_end_date)]
    tst_df = sorted_df[(sorted_df[UID_COL].isin(tst_uids)) &
                       (sorted_df[DATE_COL] >= dev_end_date)]
    
    tr_df.to_csv(trp,   compression='gzip')
    dev_df.to_csv(devp, compression='gzip')
    tst_df.to_csv(tstp, compression='gzip')
    
    print('Size: Original {}, Train {}, Dev {}, Test {}'.format(df.shape[0],
                                                                tr_df.shape[0],
                                                                dev_df.shape[0],
                                                                tst_df.shape[0]))


if __name__ == '__main__':
    for history in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        in_path    = IN_PATH_FMT.format(history)
        train_path = TRAIN_SET_PATH_FMT.format(history)
        dev_path   = DEV_SET_PATH_FMT.format(history)
        test_path  = TEST_SET_PATH_FMT.format(history)
        main(in_path, train_path, dev_path, test_path,
             PROP_USERS_IN_TRAIN, PROP_TIME_RANGE_IN_TRAIN, PROP_TIME_RANGE_IN_DEV)
        print('Finished splitting {} to {} and {}'.format(in_path, train_path, test_path))
