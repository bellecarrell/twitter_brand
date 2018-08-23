#!/usr/bin/env python

import argparse
import gzip
import io
import logging
import os
from collections import defaultdict
import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
import ijson

from configs.config import *

logging.basicConfig(level=logging.INFO)


def all_occupations(occupations_by_sector):
    all_occupations = []
    for sector in occupations_by_sector.keys():
        all_occupations = all_occupations + occupations_by_sector[sector]
    return all_occupations


def get_user_ids_from_input(in_dir, occupations, args):
    occupations_ids = defaultdict(set)
    occupations_counts = defaultdict(int)
    occupations_desc = defaultdict(set)

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            with gzip.open(os.path.join(dirpath, filename), 'rt') as f:
                try:
                    for line in f:
                        line_as_file = io.StringIO(line)
                        # Use a new parser for each line
                        if line_as_file:
                            try:
                                users = ijson.items(line_as_file, 'user')
                                for user in users:
                                    if user['description']:
                                        user_description_lower = user['description'].lower()
                                        for occupation in occupations:
                                            if occupation in user_description_lower:
                                                user_id = str(user['id'])
                                                occupations_ids[occupation].add(user_id)
                                                if args.stats:
                                                    occupations_counts[occupation] = len(occupations_ids[occupation])
                                                    if occupations_counts[occupation] % 5 == 0:
                                                        occupations_desc[occupation].add(user_description_lower)
                            except ijson.common.IncompleteJSONError as e:
                                pass
                except (IOError, EOFError) as e:
                    logging.info(e)
    return occupations_ids, occupations_counts, occupations_desc


def write_user_ids_to_out(out_dir, occupations_ids, occupations_counts, occupations_desc, args):
    counts_fn = out_dir + 'counts.txt'
    counts_file = open(counts_fn, 'a+')

    for occupation in occupations_ids.keys():

        ids = occupations_ids[occupation]

        id_file = out_dir + occupation + '_ids.txt'
        with open(id_file, 'a+') as f:
            for user_id in ids:
                f.write(user_id + "\n")

        if args.stats:
            desc = occupations_desc[occupation]

            desc_file = out_dir + occupation + '_descriptions.txt'
            with open(desc_file, 'a+') as f:
                for description in desc:
                    f.write(description + "\n")

            counts_file.write(occupation + '\t' + str(occupations_counts[occupation]) + '\n')

    counts_file.close()


if __name__ == '__main__':
    """
    Gets IDs of users who self-describe as any of
    the occupations listed in the config file.
    """

    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    parser = add_descriptive_stats_flag(parser)

    args = parser.parse_args()
    # todo: generalize by adding mode arg
    occupation_section = 'SEED_OCCUPATIONS'

    config = configparser_for_file('collect_users.ini')

    in_dir = args.input
    out_dir = args.output

    logging.info('-' * 20)
    logging.info("sending data to: " + out_dir)

    occupations_by_sector = occupations_by_sector(config, occupation_section)
    occupations = all_occupations(occupations_by_sector)

    occupations_ids, occupations_counts, occupations_desc = get_user_ids_from_input(in_dir, occupations, args)

    write_user_ids_to_out(out_dir, occupations_ids, occupations_counts, occupations_desc, args)
