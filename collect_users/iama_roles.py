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

if __name__ == '__main__':
    """
    Reads in Twitter stream from specified dir and counts freqs of all "I am a/I'm a" matches

    Args:
    twitter_dir: Directory containing items 
    user_dir: Directory to put user data file in
    """
    twitter_dir = sys.argv[1]
    out_dir = sys.argv[2]
    out = out_dir + "iama_out.txt"

    print("in file")
    users_by_role = Counter()
    users_by_role_l = {}

    ids = []

    for dirpath, _, filenames in os.walk(twitter_dir):
        for filename in filenames:
            with gzip.open(os.path.join(dirpath, filename), 'rt') as f:
                for line in f:
                    chunks = re.findall(r'"text":.*?"(.+?)"', line)
                    print("in line processing")

                    for chunk in chunks:

                        for role1, role2 in re.findall(r"I am a ([a-zA-Z]{3,15})|I'm a ([a-zA-Z]{3,15})", chunk):
                            if role1:
                                print("i_ama match")
                                logging.info(role1)
                                users_by_role[role1.lower()] += 1
                            if role2:
                                print("i'm a match")
                                logging.info(role2)
                                users_by_role[role2.lower()] += 1

    with open(out, 'w+') as f:
        print("writing to file")
        for k, v in users_by_role.items():
            f.write(k + '\t' + str(v) + '\n')



