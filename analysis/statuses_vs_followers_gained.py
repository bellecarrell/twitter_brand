import argparse
import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def main(in_dir, out_dir):
    static_info = pd.read_csv(os.path.join(in_dir,'static_info/static_user_info.csv'))
    users = static_info.loc[static_info['classify_account-mace_label']=='promoting']['user_id']

    x = []
    y = []
    for u in users:
        sc = static_info.loc[static_info['user_id']==u]['statuses_tw'].values[0]
        fc = static_info.loc[static_info['user_id'] == u]['followers_count'].values[0]
        pc = static_info.loc[static_info['user_id']==u]['percent_change'].values[0]

        if not math.isinf(pc) and not math.isnan(pc):
            followers_gained = fc*pc
            x.append(sc)
            y.append(followers_gained)

    plt.scatter(x,y)
    plt.xlabel('Tweets In Time Window')
    plt.ylabel('Followers Gained')
    #plt.show()
    fname = os.path.join(out_dir,'statuses_vs_followers_gained.png')
    plt.savefig(fname)

if __name__ == '__main__':
    """
    Create plot of statuses vs followers gained.
    """

    parser = argparse.ArgumentParser(
        description='calculate initial statistics for "shotgun" analysis of user data.'
    )
    parser.add_argument('--input_dir', required=True,
                        dest='input_dir', metavar='INPUT_DIR',
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs')
    parser.add_argument('--output_prefix', required=True,
                        dest='output_prefix', metavar='OUTPUT_PREFIX',
                        help='prefix to write out final labels, ' +
                             'descriptive statistics, and plots')
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_prefix

    main(in_dir, out_dir)