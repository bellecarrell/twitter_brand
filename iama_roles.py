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

        return re.findall(r'"text":.*?"(.+?)"', data)


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
    with gzip.open(twitter_dir, 'rt') as f:
    #with open(twitter_dir, 'r') as f:
        for line in f:

            logging.info('reading data')

            chunk = re.findall(r'"text":.*?"(.+?)"', data)

            logging.info(chunk)

            if i % 100 == 0:
                logging.info("description {} being processed".format(i))

            for role1, role2 in re.findall(r'I am a ([a-zA-Z]{3,15})|I\'m a ([a-zA-Z]{3-15})', chunk):
                logging.info(role1)
                users_by_role[role1.lower()] += 1
                logging.info(role1)
                users_by_role[role1.lower()] += 1

        print(users_by_role.most_common(200))

        with open(user_filename, 'w+') as f:
             for k, v in users_by_role.most_common(200):
                 f.write(k + '\t' + str(v) + '\n')



