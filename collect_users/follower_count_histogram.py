#!/usr/bin/env python
import sys

import numpy as np

sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
from configs.config import *
import re
import math


def collect_follower_counts(in_dir):
    follower_counts = []
    follower_count_p = re.compile(r'"followers_count":.*?(\d+)', re.S)
    users = set()

    #todo: set based on id
    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if filename.endswith("_json.txt"):
                    with open(os.path.join(dirpath, filename), 'rt') as f:
                        for line in f:
                            users.add(line)

    for user in users:
        follower_count = follower_count_p.findall(user)
        if follower_count:
            follower_count = int(follower_count[0])
            follower_counts.append(follower_count)

    return follower_counts

def log_followers_and_percentiles(out_dir, follower_counts):
    log_followers = [math.log(count, 10) if count > 0 else 0 for count in follower_counts]
    percentile = [np.percentile(log_followers, per) for per in list(range(0, 110, 10))]

    fname = out_dir + "percentiles.txt"

    with open(fname, 'a+') as f:
        for val in percentile:
            f.write("log_follower: {} follower: {}\n".format(val, 10**val))

def plot_and_save(out_dir, follower_counts):
    log_followers = [math.log(count) if count > 0 else 0 for count in follower_counts]
    bins = [np.percentile(log_followers, per) for per in list(range(0,110,10))]
    hist, bin_edges = np.histogram(follower_counts, bins) # make the histogram
    #
    # cdf = np.cumsum(hist)
    #
    # ax = plt.plot(bin_edges[1:], cdf)

    # fig, ax = plt.subplots()
    #
    # # https://stackoverflow.com/questions/33497559/display-a-histogram-with-very-non-uniform-bin-widths
    #
    # # Plot the histogram heights against integers on the x axis
    # ax.bar(range(len(hist)), hist, width=1)
    #
    # # Set the ticks to the middle of the bars
    # ax.set_xticks([0.5 + i for i, j in enumerate(hist)])
    #
    # # Set the xticklabels to a string that tells us what the bin edges were
    # ax.set_xticklabels(['{}'.format(bins[i + 1]) for i, j in enumerate(hist)])
    #
    # ax.set_xlabel('Follower Count')
    # ax.set_ylabel('Number of Followers')
    # ax.set_title('Number of Followers Per Interval')
    #
    # fname = out_dir + "follower_hist.png"
    # plt.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output

    follower_counts = collect_follower_counts(in_dir)
    log_followers_and_percentiles(out_dir,follower_counts)

