import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
import logging
from analysis.rlr import *
logging.basicConfig(level=logging.INFO)
import nltk
import os

from analysis.batching import *

SEED = 12345
VERBOSE = True
import numpy as np

from analysis.file_data_util import *
import argparse
name = 'fc_by_sp_joint'
n_batches = 100

def filter_by_stratification(X,us, static_info,n_features,s_to_y_val):
    print('Filtering users for batch')
    filt_us = []
    for i, u in enumerate(us):
        y = _generate_single_y_joint(u, static_info,s_to_y_val)
        if y:
            filt_us.append((i,u,y))

    print('{} users that have high or low follower counts in batch'.format(len(filt_us)))
    filt_X = np.ndarray(shape=(len(filt_us),n_features))
    ys = np.ndarray(shape=(len(filt_us),))
    for i, (old_i,u,y) in enumerate(filt_us):
        filt_X[i] = X[old_i]
        ys[i] = y

    return filt_X, ys

def _generate_single_y_joint(u, static_info,s_to_y_val):
    percentile = int(static_info.loc[static_info['user_id'] == u]['percentile'].values[0])
    s = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]

    if percentile >= 90:
        return s_to_y_val[(s,1)]
    if percentile < 70:
        return s_to_y_val[(s,0)]
    else:
       return None


def fit_batches(in_dir, static_info, model_dir, s_to_y_val, batch='month', l1_range=[0.0, 1.0]):
    np.random.seed(SEED)

    vocab = from_gz(os.path.join(in_dir,'features'),'vocab')

    rr = RandomizedRegression(is_continuous=False, model_dir=model_dir, log_l1_range=True)


    if rr.log_l1_range:
        l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
    else:
        l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

    start = time.time()

    # fit each batch
    for dirpath, _, filenames in os.walk(os.path.join(in_dir,'batches','{}_batches'.format(batch))):
        filenames = [f for f in filenames if f.startswith('batch')]
        for b, (filename, l1) in enumerate(zip(filenames,l1s)):
            data = np.load(os.path.join(dirpath,filename))
            X, us, tw = batch_from_compressed(data, len(vocab))
            X, y= filter_by_stratification(X,us,static_info,len(vocab),s_to_y_val)

            print('{} size of y'.format(y.shape))
            rr.fit_batch(X, y, l1, b)
            if VERBOSE:
                print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))

    return rr


def main(in_dir,out_dir):
    out_dir = os.path.join(out_dir, name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    static_info = pd.read_csv(os.path.join(in_dir,'static_info/static_user_info.csv'))
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    s_l_h = [(s,v) for s in specializations for v in [0,1]]
    y_to_s_val = dict((i, s) for i, s in enumerate(s_l_h))
    s_to_y_val = dict((v,k) for k, v in y_to_s_val.items())
    vocab = from_gz(os.path.join(in_dir, 'features'), 'vocab')

    model_dir = './log_reg_models_{}_'.format(name)

    rr = fit_batches(in_dir, static_info, model_dir, s_to_y_val, l1_range=[0.0, 10.0])
    sf = rr.get_salient_features(dict((v,k) for k,v in vocab.items()), y_to_s_val,n=1300)

    with open(os.path.join(out_dir,'rlr_selected_features_{}.txt'.format(name)),'w+') as f:
        for s, feat in sf.items():
            f.write('Target: {} Salient features: {}\n\n'.format(s, feat))
    
if __name__ == '__main__':
    """
    Run randomized logistic regression to see features selected for different specializations.
    May add different class of Ys in the future for user impact (so, change in follower count or
    a related metric)
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