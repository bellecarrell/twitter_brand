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

def write_ids_from_tsv(tsv_filename, out):
    ids = []
    with open(tsv_filename, 'r') as f:
        first_line = f.readline()
        for line in f:
            try:
                id = line.split('\t')[1]
                if id is not 'id':
                    ids.append(id)
            except IndexError:
                pass

    with open(out, 'w+') as f:
        for id in ids[:4000]:
            f.write(str(id)+'\n')

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    role = out_dir.split('/')[-2]

    bins = ['100-200', '1000-1500', '5000-6000', '10000-11000']

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            for bin in bins:
                f_name = "uinfos_for_" + str(bin) + ".tsv"
                if file == f_name:
                    f_name = os.path.join(root, file)
                    print(f_name)
                    out = out_dir + bin + ".txt"
                    write_ids_from_tsv(f_name, out)