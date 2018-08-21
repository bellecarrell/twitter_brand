#!/usr/bin/env python

import datetime
import gzip
import logging
import os
import pickle
import sys

from pandas import DataFrame as df

logging.basicConfig(level=logging.INFO)
epoch = datetime.datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    role = out_dir.split('/')[-2]

    created_ats = []
    statuses_counts = []

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('pickle.gz'):
                f_name = os.path.join(root, file)
                f = gzip.open(f_name, 'rb')
                uinfos = pickle.load(f)

                if len(uinfos) > 0:
                    for uinfo in uinfos:
                        time_since_epoch = unix_time_millis(uinfo.created_at)
                        created_ats.append(time_since_epoch)
                        statuses_counts.append(uinfo.statuses_count)

    data = df(data={'created_at': created_ats, 'statuses_count': statuses_counts})

    summ = data.describe()

    f_name = out_dir + "summ_stats_statuses.csv"

    summ.to_csv(f_name)

    summ = data['created_at'].describe()

    f_name = out_dir + "summ_stats_created.csv"

    summ.to_csv(f_name)
