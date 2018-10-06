from collect_users.sample_users_by_percentile_range import is_active_id
import os
import numpy.random

def users_from_sample(in_dir, sample_size):
    #hardcoded. todo: count files ending in users.txt?
    users_per_interval = sample_size/10
    sample_subset = []

    used_fname = in_dir + 'used_ids.txt'
    used_ids = []
    with open(used_fname, 'a+') as f:
        used_ids = f.readlines()

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if filename.endswith("users.txt"):
                curr_f_users = 0
                with open(os.path.join(dirpath, filename)) as f:
                    for id in f:
                        id = int(id)
                        if curr_f_users < users_per_interval:
                            if str(id) not in used_ids and is_active_id(str(id)):
                                sample_subset.append(id)
                                curr_f_users += 1
                                add_to_used(in_dir, id)

    numpy.random.shuffle(sample_subset)
    return sample_subset

def add_to_used(in_dir, id):
    used_fname = in_dir + 'used_ids.txt'
    with open(used_fname, 'a+') as f:
        f.write(str(id) + "\n")

