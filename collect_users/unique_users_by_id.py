#!/usr/bin/env python
import sys

sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
from configs.config import *
import re
from utils.file_io import *

def unique_ids_desciptions_and_json(in_dir):
    user_ids = set()
    user_descriptions = []
    user_json = set()
    user_id_p = re.compile(r'"id":.*?(\d+)', re.S)
    desc_p = re.compile(r'"description":.*?"(.+?)"', re.S)

    duplicate_users = 0

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if filename.endswith("_json.txt"):
                with open(os.path.join(dirpath, filename), 'rt') as f:
                    for line in f:
                        user_id = user_id_p.findall(line)
                        user_id = user_id[0]
                        if user_id not in user_ids:
                            user_ids.add(user_id)

                            user_description = desc_p.findall(line)
                            user_description = user_description[0]
                            user_descriptions.append(user_description)

                            user_json.add(line)

    return user_ids, user_descriptions, user_json, duplicate_users

def write_unique_users_to_file(out_dir, user_ids, user_descriptions, user_json):
    id_fname = out_dir + "blogger_ids.txt"
    desc_fname = out_dir + "blogger_descriptions.txt"
    json_fname = out_dir + "blogger_json.txt"

    write_list_to_file(id_fname, user_ids, mode='w+')
    write_list_to_file(desc_fname, user_descriptions, mode='w+')
    write_list_to_file(json_fname, user_json, mode='w+')

def write_count_to_file(out_dir, count):
    fname = out_dir + "count.txt"
    with open(fname, 'w+') as f:
        f.write("count\t" + str(count))

if __name__ == '__main__':
    """
    From input directory containing multiple files of type {role}_description, {role}_json,
    aggregates and outputs {role}_description and {role}_json for all unique users based on user id.
    """
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output

    user_ids, user_descriptions, user_json, duplicate_users = unique_ids_desciptions_and_json(in_dir)

    write_unique_users_to_file(out_dir, user_ids, user_descriptions, user_json)
    write_count_to_file(out_dir, len(user_ids))

