#!/usr/bin/env python

from __future__ import division
import logging
import sys
import os
from collections import Counter, defaultdict
import pickle
import gzip

import numpy as np
import matplotlib.mlab as mlab
import matplotlib

matplotlib.use('Agg')

"""
Script for plotting log success (from Cohn User Impact paper) in box
& whisker plot for each role/#followers bin
"""

import datetime
import matplotlib.pyplot as plt
import random
from math import log

epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    role = out_dir.split('/')[-2]

    bins = ['100-200', '1000-1500', '5000-6000', '10000-11000']
    uinfos_by_bin = [[] for i in range(len(bins))]

    jan_1_time_since_epoch = unix_time_millis(datetime.datetime(2017, 1, 1, 0, 0, 0))

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('pickle.gz'):
                f_name = os.path.join(root, file)
                f = gzip.open(f_name, 'rb')
                try:
                    uinfos = pickle.load(f)
                except:
                    pass

                if len(uinfos) > 0:
                    for uinfo in uinfos:

                        statuses_count = uinfo.statuses_count
                        time_since_epoch = unix_time_millis(uinfo.created_at)

                        if statuses_count < 4450 and statuses_count > 90:

                            followers_count = int(uinfo.followers_count)
                            following = int(uinfo.friends_count)
                            listed_count = int(uinfo.listed_count)
                            theta = 1

                            success_num = (listed_count + theta) * (followers_count + theta) ** 2
                            success_denom = following + theta

                            success = log(success_num / success_denom)

                            if 100 < followers_count and followers_count < 200:
                                uinfos_by_bin[0].append(success)
                            if 1000 < followers_count and followers_count < 1500:
                                uinfos_by_bin[1].append(success)
                            if 5000 < followers_count and followers_count < 6000:
                                uinfos_by_bin[2].append(success)
                            if 10000 < followers_count and followers_count < 11000:
                                uinfos_by_bin[3].append(success)

    # make and save plot for each
    for i, bin in enumerate(bins):
        if len(uinfos_by_bin[i]) > 0:

            plt.boxplot(uinfos_by_bin[i])

            f_name = out_dir + "plot_for_" + str(bin) + ".png"

            plt.savefig(f_name)
            plt.close()
