#!/usr/bin/env python
import sys

import numpy as np

sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
from configs.config import *
import re
import math
import matplotlib as plt

def collect_follower_counts(in_dir):
    user_ids = set()
    follower_counts = []
    follower_count_p = re.compile(r'"followers_count":.*?(\d+)', re.S)
    user_id_p = re.compile(r'"id":.*?(\d+)', re.S)

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if filename.endswith("_json.txt"):
                    with open(os.path.join(dirpath, filename), 'rt') as f:
                        for line in f:
                            user_id = user_id_p.findall(line)
                            user_id = user_id[0]
                            if user_id not in user_ids:
                                user_ids.add(user_id)
                                follower_count = follower_count_p.findall(line)
                                if follower_count:
                                    follower_count = int(follower_count[0])
                                    follower_counts.append(follower_count)

    return follower_counts

def log_followers_and_percentiles(follower_counts):
    log_follower_counts = [math.log(count, 10) if count > 0 else 0 for count in follower_counts]
    percentiles = [np.percentile(log_follower_counts, per) for per in list(range(0, 110, 10))]

    return log_follower_counts, percentiles

def save_percentiles(out_dir, percentiles):
    fname = out_dir + "percentiles.txt"

    with open(fname, 'a+') as f:
        for val in percentiles:
            f.write("log_follower: {} follower: {}\n".format(val, 10**val))

def cum_density_plot(out_dir, log_follower_counts):
    hist, bin_edges = np.histogram(log_follower_counts) # make the histogram

    cdf = np.cumsum(hist)

    plt.plot(bin_edges[1:], cdf / cdf[-1])

    fname = out_dir + "cum_density.png"
    plt.savefig(fname)

    plt.close()

def histogram_non_uniform_bin_widths(out_dir, log_follower_counts):
    #todo: ask Adrian what bins would be good to see in hist
    hist, bin_edges = np.histogram(log_follower_counts)

    fig, ax = plt.subplots()

    # https://stackoverflow.com/questions/33497559/display-a-histogram-with-very-non-uniform-bin-widths

    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)), hist, width=1)

    # Set the ticks to the middle of the bars
    ax.set_xticks([0.5 + i for i, j in enumerate(hist)])

    # Set the xticklabels to a string that tells us what the bin edges were
    ax.set_xticklabels(['{}'.format(bins[i + 1]) for i, j in enumerate(hist)])

    ax.set_xlabel('Follower Count')
    ax.set_ylabel('Number of Followers')
    ax.set_title('Number of Followers Per Interval')

    fname = out_dir + "follower_hist.png"
    plt.savefig(fname)


def plot_and_save(out_dir, follower_counts):
    log_follower_counts, percentiles = log_followers_and_percentiles(follower_counts)

    save_percentiles(out_dir, percentiles)

    cum_density_plot(out_dir, log_follower_counts)

    histogram_non_uniform_bin_widths()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output

    follower_counts = collect_follower_counts(in_dir)

    log_followers_and_percentiles(out_dir, follower_counts)

