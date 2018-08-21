#!/usr/bin/env python

import logging
import sys
import os
from collections import Counter, defaultdict
import pickle

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out = sys.argv[2]

    total_unigrams = Counter()
    total_bigrams = Counter()
    total_trigrams = Counter()

    artist_roles = ['designer', 'poet', 'rapper', 'writer', 'singer', 'musician', 'actor',
                    'artist', 'dancer', 'freelance', 'photographer', 'journalist', 'reporter']

    role_counts = Counter()

    role_files = {}

    for role in artist_roles:
        fname = out + role + '.txt'
        role_files[role] = open(fname, 'w+')

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.p'):
                counts = pickle.load(open(os.path.join(root, file), "rb"))
                role_counts = role_counts + counts

            if file.endswith('.txt'):
                for role in artist_roles:
                    if role in file:
                        with open(os.path.join(root, file), 'r+') as f:
                            for line in f:
                                role_files[role].write(line)

    for role in artist_roles:
        role_files[role].close()

    fname2 = out + '_agg_count.tsv'
    with open(fname2, 'w+') as f:
        for role in artist_roles:
            f.write(role + '\t' + str(role_counts[role]) + '\n')