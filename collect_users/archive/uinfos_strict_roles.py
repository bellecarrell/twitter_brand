#!/usr/bin/env python

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

import datetime
from uinfo_summ_stats import unix_time_millis
import matplotlib.pyplot as plt
import random
import csv

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    role = out_dir.split('/')[-2]

    artist_roles = ['designer', 'poet', 'rapper', 'writer', 'singer', 'musician', 'actor',
                    'artist', 'dancer', 'freelance', 'photographer', 'journalist', 'reporter']

    if role in artist_roles:
        artist_roles.remove(role)

    bins = ['100-200', '1000-1500', '5000-6000', '10000-11000']
    uinfos_by_bin = [[] for i in range(len(bins))]

    jan_1_time_since_epoch = unix_time_millis(datetime.datetime(2017, 1, 1, 0, 0, 0))

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('pickle.gz'):
                f_name = os.path.join(root, file)
                f = gzip.open(f_name, 'rb')
                uinfos = pickle.load(f)

                if len(uinfos) > 0:
                    for uinfo in uinfos:
                        statuses_count = uinfo.statuses_count
                        time_since_epoch = unix_time_millis(uinfo.created_at)
                        desc = uinfo.description.lower()

                        desc_contains_other_roles = False

                        uinfo_str = [uinfo.description.encode("utf-8"), uinfo.id_str.encode("utf-8"), str(
                            uinfo.statuses_count), str(uinfo.created_at)]

                        uinfos_by_bin[0].append(uinfo_str)

                        # filter out descriptions that contain other roles
                        # for not_role in artist_roles:
                        #    if not_role in desc:
                        #        desc_contains_other_roles = True

                        if time_since_epoch >= jan_1_time_since_epoch and statuses_count < 4450 and statuses_count > 90 and desc_contains_other_roles is False:

                            followers_count = int(uinfo.followers_count)

                            uinfo_str = uinfo.description.encode("utf-8") + " " + uinfo.id_str.encode(
                                "utf-8") + " " + str(uinfo.statuses_count) + " " + str(uinfo.created_at)

                            if 100 < followers_count and followers_count < 200:
                                uinfos_by_bin[0].append(uinfo_str)
                            if 1000 < followers_count and followers_count < 1500:
                                uinfos_by_bin[1].append(uinfo_str)
                            if 5000 < followers_count and followers_count < 6000:
                                uinfos_by_bin[2].append(uinfo_str)
                            if 10000 < followers_count and followers_count < 11000:
                                uinfos_by_bin[3].append(uinfo_str)

    for i, bin in enumerate(bins):
        if len(uinfos_by_bin[i]) > 0:
            print(uinfos_by_bin[i])
            header = ['desc', 'id_str', 'statuses_count', 'created_at']
            f_name = out_dir + "uinfos_for_" + str(bin) + ".csv"
            with open(f_name, "wb") as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerows(uinfos_by_bin[i])

