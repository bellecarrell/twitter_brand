'''
Split by user set and time range into train and test folds.
'''

import datetime
import os
import pandas as pd
import random

UID_COL = 'user_id'
DATE_COL = 'sampled_datetime'

IN_DIR = '/exp/abenton/twitter_brand_workspace_20190417/extracted_features_20190508/'

IN_PATH = IN_DIR + 'joined_features.with_domain.tsv.gz'
TRAIN_PATH = IN_DIR + 'joined_features.with_domain.train.tsv.gz'
DEV_PATH = IN_DIR + 'joined_features.with_domain.dev.tsv.gz'
TEST_PATH = IN_DIR + 'joined_features.with_domain.test.tsv.gz'

PROP_USERS_IN_TRAIN = 0.6  # proportion of users selected for the training set

TEST_THRESH_DATE = datetime.datetime(year=2019, month=3, day=1)


def main(inp, trp, devp, tstp, prop_users, test_thresh_date):
    df = pd.read_table(inp, parse_dates=[DATE_COL])
    sorted_df = df.sort_values(DATE_COL)
    
    uids = list(sorted_df[UID_COL].unique())
    random.shuffle(uids)
    
    tr_uids = set(uids[:int(prop_users*len(uids))])
    tst_uids = set(uids[int(prop_users*len(uids)):])
    
    tr_df  = sorted_df[(sorted_df[UID_COL].isin(tr_uids)) &
                       (sorted_df[DATE_COL] < test_thresh_date)]
    dev_df = sorted_df[(sorted_df[UID_COL].isin(tr_uids)) &
                       (sorted_df[DATE_COL] >= test_thresh_date)]
    tst_df = sorted_df[(sorted_df[UID_COL].isin(tst_uids)) &
                       (sorted_df[DATE_COL] >= test_thresh_date)]
    
    tr_df.to_csv(trp, sep='\t', header=True, index=False, compression='gzip')
    dev_df.to_csv(devp, sep='\t', header=True, index=False, compression='gzip')
    tst_df.to_csv(tstp, sep='\t', header=True, index=False, compression='gzip')
    
    print('Size: Original {}, Train {}, Dev {}, Test {}'.format(df.shape[0],
                                                                tr_df.shape[0],
                                                                dev_df.shape[0],
                                                                tst_df.shape[0]))


if __name__ == '__main__':
    main(IN_PATH, TRAIN_PATH, DEV_PATH, TEST_PATH,
         PROP_USERS_IN_TRAIN, TEST_THRESH_DATE)
    print('Finished splitting {} to {}, {}, {}'.format(os.path.basename(IN_PATH),
                                                       os.path.basename(TRAIN_PATH),
                                                       os.path.basename(DEV_PATH),
                                                       os.path.basename(TEST_PATH)))
