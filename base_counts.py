#!/usr/bin/env python

import sys
import gzip
import os

from collections import Counter
import os
import gzip
import re
import logging

logging.basicConfig(level=logging.INFO)

def grab_user_data(twitter_dir, user_filename):

    with gzip.open(twitter_dir, 'rt') as f:
    #with open(twitter_dir, 'r') as f:
        data = f.read()

        logging.info('reading data')

        return re.findall(r'"description":.*?"(.+?)"', data)


if __name__ == '__main__':
    """
    Reads in Twitter stream from specified dir and puts ids into social role bins

    Args:
    twitter_dir: Directory containing items 
    user_dir: Directory to put user data file in
    """
    twitter_dir = sys.argv[1]
    user_dir = sys.argv[2]
    user_filename = user_dir + "_user.txt"

    print("in file")
    users_by_role = Counter()
    users_by_role_l = {}

    ids = []

    chunks = grab_user_data(twitter_dir, user_filename)

    logging.info("number of non-empty descriptions: {}".format(len(chunks)))

    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            logging.info("description {} being processed".format(i))

        for word in re.findall(r'\b[a-z]{3,15}\b', chunk):
            users_by_role[word.lower()] += 1

    print(users_by_role.most_common(200))

    with open(user_filename, 'w+') as f:
         for k, v in users_by_role.most_common(200):
             f.write(k + '\t' + str(v) + '\n')



