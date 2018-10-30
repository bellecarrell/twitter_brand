
import os
import numpy.random
import requests
from utils.twitter_user_util import *
import requests

def users_from_sample(in_dir, sample_size):
    # hardcoded. todo: count files ending in users.txt?
    users_per_interval = sample_size / 10
    sample_subset = []

    used_fname = in_dir + 'used_ids.txt'
    used_ids = []
    with open(used_fname, 'r+') as f:
        used_ids = [int(line) for line in f.readlines()]

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            curr_f_users = 0
            with open(os.path.join(dirpath, filename)) as f:
                for line in f:
                    if line != '\n':
                        id = field_from_json('id_str', line)
                        if id:
                            if curr_f_users < users_per_interval:
                                try:
                                    if int(id) not in used_ids and is_active_id(str(id)) and has_linked_page(line):
                                        sample_subset.append(int(id))
                                        curr_f_users += 1
                                        add_to_used(in_dir, id)
                                except requests.exceptions.ConnectionError:
                                    pass

    all_ids = list(set(used_ids)) + sample_subset
    all_ids_set = set(all_ids)

    # ensuring sample has no overlap with used ids
    assert(len(all_ids) == len(all_ids_set))

    numpy.random.shuffle(sample_subset)
    return sample_subset


def add_to_used(in_dir, id):
    used_fname = in_dir + 'used_ids.txt'
    with open(used_fname, 'a+') as f:
        f.write(str(id) + "\n")
