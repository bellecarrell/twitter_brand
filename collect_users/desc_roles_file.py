#!/usr/bin/env python

import sys
from collections import Counter
import os
import gzip
import re
import logging
import pickle

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    """
    Reads in Twitter stream from specified dir and counts freqs of all "I am a/I'm a" matches
    unigrams, bigrams, and trigrams

    Args:
    twitter_dir: Directory containing items 
    user_dir: Directory to put user data file in
    """
    #in_dir = sys.argv[1]
    twitter_dir = sys.argv[1]
    out_dir = sys.argv[2]
    #
    # total_unigrams = Counter()
    # total_bigrams = Counter()
    # total_trigrams = Counter()
    #
    # unigram = 'unigram'
    # bigram = 'bigram'
    # trigram = 'trigram'
    #
    # for dirpath, _, filenames in os.walk(in_dir):
    #     for filename in filenames:
    #         if filename.endswith(".p"):
    #             counts = pickle.load(open(os.path.join(dirpath, filename), "rb"))
    #             if unigram in filename:
    #                 total_unigrams = counts
    #             if bigram in filename:
    #                 total_bigrams = counts
    #             if trigram in filename:
    #                 total_trigrams = counts

    logging.info('-'*20)
    logging.info("sending data to: " + out_dir)

    # unis = total_unigrams.keys()
    # bis = total_bigrams.keys()
    # tris = total_trigrams.keys()

    artist_roles = ['designer', 'poet', 'rapper', 'writer', 'singer', 'musician', 'actor',
                    'artist', 'dancer', 'freelance', 'photographer', 'journalist', 'reporter']

    user_p = re.compile(r'"user":.*?\{(.+?)\}',re.S)
    desc_p = re.compile(r'"description":.*?"(.+?)"', re.S)
    id_p = re.compile(r'"id_str":.*?"(.+?)"', re.S)

    role_counts = Counter()

    for dirpath, _, filenames in os.walk(twitter_dir):
        for filename in filenames:
            with gzip.open(os.path.join(dirpath, filename), 'rt') as f:
                try:
                    for line in f:
                    #line = f.read()
                    #if line:
                        users = user_p.findall(line)

                        for user in users:
                            logging.info("user data: " + user)

                            desc = re.findall(desc_p, user)

                            user_id = re.findall(id_p, user)

                            if len(desc) > 0 and len(user_id) > 1:
                                desc = desc[0]
                                user_id = user_id[0]
                                logging.info("desc " + desc)
                                logging.info("user_id " + user_id)

                                for role in artist_roles:
                                    if role in desc:

                                        role_counts[role] += 1

                                        role_file = out_dir + role + '.txt'

                                        with open(role_file, 'a+') as f:
                                            f.write(user_id + "\n")

                except IOError as e:
                    logging.info(e)

    artist_out = open(out_dir + "artist_roles.p", 'wb+')


    pickle.dump(role_counts, artist_out)

    artist_out.close()
    roles_sorted = sorted(role_counts, key=role_counts.get, reverse=True)

    logging.info("roles: {}".format(roles_sorted))

    artist_kv_out = out_dir + "artist_roles.tsv"

    with open(artist_kv_out, 'w+') as f:
        for role in roles_sorted:
            f.write(role + "\t" + str(role_counts[role]) + "\n")