#!/usr/bin/env python

import logging
import sys
import os
from collections import Counter

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out = sys.argv[2]

    iama_roles = Counter()

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                for line in f:
                    role, count = line.split('\t')
                    iama_roles[role] += int(count)

    iama_roles_sorted = sorted(iama_roles, key=iama_roles.get, reverse=True)

    with open(out, 'w+') as f:
        for role in iama_roles_sorted:
            f.write(role + '\t' + str(iama_roles[role]) + '\n')

