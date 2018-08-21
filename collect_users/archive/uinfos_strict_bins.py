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

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    role = out_dir.split('/')[-2]

    bins = ['100-200', '1000-1500', '5000-6000', '10000-11000']
    uinfos_by_bin = [[] for i in range(len(bins))]

    artist_roles = ['designer', 'poet', 'rapper', 'writer', 'singer', 'musician', 'actor',
                    'artist', 'dancer', 'freelance', 'photographer', 'journalist', 'reporter']

    if role in artist_roles:
        artist_roles.remove(role)

    jan_1_time_since_epoch = unix_time_millis(datetime.datetime(2017, 1, 1, 0, 0, 0))

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('pickle.gz'):
                f_name = os.path.join(root, file)
                f = gzip.open(f_name,'rb')
                uinfos = pickle.load(f)

                if len(uinfos) > 0:
                    for uinfo in uinfos:

                        statuses_count = uinfo.statuses_count
                        time_since_epoch = unix_time_millis(uinfo.created_at)

                        desc = uinfo.description
                        desc_contains_other_roles = False

                        # filter out descriptions that contain other roles
                        for not_role in artist_roles:
                            if not_role in desc:
                                desc_contains_other_roles = True

                        if time_since_epoch >= jan_1_time_since_epoch and statuses_count < 4450 and statuses_count > 90 and desc_contains_other_roles is False:

                            followers_count = int(uinfo.followers_count)

                            uinfo_str = uinfo.description.encode("utf-8") + "\t" + uinfo.id_str.encode("utf-8") + "\t" + str(uinfo.statuses_count) + "\t" + str(uinfo.created_at) + "\n"

                            if 10000 < followers_count and followers_count < 11000:
                                uinfos_by_bin[3].append(uinfo_str)


    for i, bin in enumerate(bins):
        if len(uinfos_by_bin[i]) > 0:
            f_name = out_dir + "uinfos_for_" + str(bin) + ".tsv"
            with open(f_name, 'w+') as f:
                random.shuffle(uinfos_by_bin[i])
                f.write("description" + "\t" + "id" + "\t" + "tweets" + "\t" + "created_at" + "\n")
                for uinfo in uinfos_by_bin[i]:
                    f.write(uinfo)