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



import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    role = out_dir.split('/')[-2]

    followers = []

    bins = [10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 1000001]
    ids_by_bin = [[] for i in range(len(bins))]

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('pickle.gz'):
                f_name = os.path.join(root, file)
                f = gzip.open(f_name,'rb')
                uinfos = pickle.load(f)

                if len(uinfos) > 0:
                    for uinfo in uinfos:

                        followers_count = uinfo.followers_count

                        followers.append(followers_count)

                        bin_idx = None
                        u_id = int(uinfo.id_str)

                        for bin in bins:
                            if followers_count < bin:
                                bin_idx = bins.index(bin)-1
                                break
                        if followers_count > 1000000:
                            bin_idx = 10

                        ids_by_bin[bin_idx].append(u_id)

    logging.info('followers {}'.format(followers))
    logging.info('ids_by_bin {}'.format([(bins[i], ids_by_bin[i]) for i in range(len(bins))]))

    x_pos = [str(i) for i, _ in enumerate(bins)]
    bins_t = [str(bin) for bin in bins]
    print(bins_t)

    y = [len(bin) for bin in ids_by_bin]

    f_name = out_dir + "followers"

    plt.bar(x_pos, y)
    plt.title(out_dir.split('/')[-2])
    plt.xticks(x_pos, bins_t)
    plt.savefig(f_name)
    plt.close()

    f_name = out_dir + "followers.tsv"
    with open(f_name, 'w+')

    for i, bin in enumerate(bins):
        f_name = out_dir + str(bin) + ".tsv"
        with open(f_name, 'w+') as f:
            for id in ids_by_bin[i]:
                f.write(str(id) + "\n")