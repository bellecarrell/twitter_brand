
import os
import numpy.random
import requests
from utils.twitter_user_util import *
import requests

def users_from_sample(in_dir, sample_size):
    # all files under this directory that do not start with "used_ids" are assumed to
    # have user profiles in json format
    json_uid_paths = [os.path.join(in_dir, p) for p in os.listdir(in_dir)
                      if not p.startswith('used_ids')]
    num_intervals = len(json_uid_paths)
    
    # hardcoded. todo: count files ending in users.txt?
    users_per_interval = sample_size / num_intervals
    sample_subset = []
    
    used_fname = os.path.join(in_dir, 'used_ids.txt')
    used_ids = []
    with open(used_fname, 'r+') as f:
        used_ids = [int(line.strip()) for line in f]
    
    for pidx, p in enumerate(json_uid_paths):
        curr_f_users = 0
        with open(p, 'rt') as f:
            print('Checking out "{}" {}/{}, finished {}/{}'.format(
              p, pidx, len(json_uid_paths), len(sample_subset), sample_size)
            )
            for line in f:
                if not line.strip():
                    continue
                
                id = field_from_json('id_str', line)
                if id:
                    if curr_f_users < users_per_interval:
                        try:
                            if int(id) not in used_ids and is_active_id(str(id)) and has_linked_page(line):
                                sample_subset.append(int(id))
                                curr_f_users += 1
                                add_to_used(in_dir, id)
                                print('sampled {}'.format(id))
                        except requests.exceptions.ConnectionError:
                            print('ConnectionError on {}'.format(id))
    
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
