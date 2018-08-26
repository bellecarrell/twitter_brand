#!/usr/bin/env python

import logging
from collections import Counter
import argparse

logging.basicConfig(level=logging.INFO)

def create_aggregate_output_files(out_dir, occupations):
    aggregate_files = {}
    for occupation in occupations:
        fname = out_dir + occupation + '.txt'
        aggregate_files[occupation] = open(fname, 'w+')
    return aggregate_files

def write_to_aggregate_output_files(in_dir, aggregate_files, occupations):
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.txt'):
                for occupation in occupations:
                    if occupation in file:
                        with open(os.path.join(root, file), 'r+') as f:
                            for line in f:
                                aggregate_files[occupation].write(line)

if __name__ == '__main__':
    """
    Aggregates the unfiltered user IDs collected
    via get_user_ids for the occupations in the config file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--location", choices=['LOCAL', 'GRID'],
                        default='LOCAL', help="location of script execution")
    parser.add_argument("-i", "--input", help="input filepath. only needed if not in config file")
    parser.add_argument("-o", "--output", help="output filepath. only needed if not in config file")

    args = parser.parse_args()
    location = args.location

    config = configparser_for_file('aggregate_user_ids.ini')

    in_dir = config_section_map(config, location)['in']
    out_dir = config_section_map(config, location)['out']

    if args.input:
        in_dir = args.input
    if args.output:
        out_dir = args.output

    logging.info('-' * 20)
    logging.info("sending data to: " + out_dir)

    occupations = config_section_map(config, location)['occupations'].split(',')

    role_counts = Counter()

    aggregate_files = create_aggregate_output_files(out_dir, occupations)

    write_to_aggregate_output_files(in_dir, aggregate_files, occupations)

    for occupation in occupations:
        aggregate_files[occupation].close()
