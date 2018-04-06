#!/usr/bin/env python

import logging
import sys
import os
from collections import Counter, defaultdict
import pickle
import gzip
from dateutil import parser
import pandas as pd
from pandas import DataFrame as df
import numpy as np
logging.basicConfig(level=logging.INFO)

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
                        created_at_pd = pd.Timestamp(uinfo.created_at)
                        created_ats.append(created_at_pd)
                        statuses_counts.append(uinfo.statuses_count)

    data = df(data={'created_at': created_ats, 'statuses_count': statuses_counts})

    summ = data.describe()

    f_name = out_dir + "summ_stats_statuses.csv"

    summ.to_csv(f_name)
