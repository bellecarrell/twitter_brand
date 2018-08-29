#!/usr/bin/env python

import gzip
import logging
import sys

sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
import ijson
import re
from configs.config import *
import time

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
    occupations_json = defaultdict(set)

    user_p = re.compile(r'"user":.*?(\{.+?\})', re.S)
    desc_p = re.compile(r'"description":.*?"(.+?)"', re.S)
    id_p = re.compile(r'"id_str":.*?"(.+?)"', re.S)
    createdat_p = re.compile(r'"created_at":.*?"(.+?)"', re.S)

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            with gzip.open(os.path.join(dirpath, filename), 'rt') as f:
                try:
                    count = 0
                    for line in f:
                        # Use a new parser for each line
                        if line:
                            try:
                                users = user_p.findall(line)
                                for user in users:

                                    desc = re.findall(desc_p, user)
                                    user_id = re.findall(id_p, user)
                                    created_at = re.findall(createdat_p, user)
                                    created_at = created_at[0]
                                    date2017 = time.strptime('2017-01-01', '%Y-%m-%d') < time.strptime(created_at,
                                                                                                       '%a %b %d %H:%M:%S +0000 %Y')

                                    if len(desc) > 0 and len(
                                            user_id) > 0:
                                        user_description_lower = desc[0].lower()
                                        user_id = user_id[0]

                                        for occupation in occupations:
                                            if occupation in user_description_lower and date2017:
                                                count += 1
                                                occupations_ids[occupation].add(user_id)
                                                if args.stats:
                                                    occupations_desc[occupation].add(user_description_lower)
                                                    occupations_json[occupation].add(user)
                            except ijson.common.IncompleteJSONError as e:
                                pass
                except (IOError, EOFError, Exception) as e:
                    logging.info(e)
    for occupation in occupations:
        occupations_counts[occupation] = len(occupations_ids[occupation])
    return occupations_ids, occupations_counts, occupations_desc, occupations_json


def write_user_ids_to_out(out_dir, occupations_ids, occupations_counts, occupations_desc, occupations_json, args):
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

            jsons = occupations_json[occupation]

            json_file = out_dir + occupation + '_json.txt'
            with open(json_file, 'a+') as f:
                for json in jsons:
                    f.write(json + "\n")

            counts_file.write(occupation + '\t' + str(occupations_counts[occupation]) + '\n')

    counts_file.close()


if __name__ == '__main__':
    """
    Gets IDs, descriptions, and JSON of users who self-describe as any of
    the occupations listed in the config file.
    """

    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    parser = add_descriptive_stats_flag(parser)
    parser = add_occupation_mode(parser)

    args = parser.parse_args()

    config = configparser_for_file('collect_users.ini')

    in_dir = args.input
    out_dir = args.output
    occupation_section = args.om

    logging.info('-' * 20)
    logging.info("sending data to: " + out_dir)

    occupations_by_sector = occupations_by_sector(config, occupation_section)
    occupations = all_occupations(occupations_by_sector)

    occupations_ids, occupations_counts, occupations_desc, occupations_json = get_user_ids_from_input(in_dir,
                                                                                                      occupations, args)

    logging.info('-' * 20)
    logging.info("writing " + str(len(occupations_ids)) + " ids")

    write_user_ids_to_out(out_dir, occupations_ids, occupations_counts, occupations_desc, occupations_json, args)
