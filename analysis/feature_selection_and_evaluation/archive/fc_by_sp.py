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
name = 'fc_by_sp'
n_batches = 100

def filter_by_stratification(X,us, static_info,n_features,s):
    filt_us = []
    for i, u in enumerate(us):
        y = _generate_single_y_low_high(u,static_info,s)
        if y:
            filt_us.append((i,u,y))

    filt_X = np.ndarray(shape=(len(filt_us),n_features))
    y_l = np.ndarray(shape=(len(filt_us), 1))
    y_h = np.ndarray(shape=(len(filt_us), 1))
    for i, (old_i,u,y) in enumerate(filt_us):
        filt_X[i] = X[old_i]
        l, h = y
        y_l[i] = l
        y_h[i] = h

    return filt_X, y_l, y_h

def _generate_single_y_low_high(u, static_info,s):
    percentile = int(static_info.loc[static_info['user_id'] == u]['percentile'].values[0])
    specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]

    if specialization == s:
        if percentile >= 90:
            return (0,1)
        if percentile < 70:
            return (1,0)
    else:
       return None


def fit_batches(in_dir, static_info, s, model_dir, batch='month', l1_range=[0.0, 1.0]):
    np.random.seed(SEED)

    vocab = from_gz(os.path.join(in_dir,'features'),'vocab')
    low_md = model_dir + '_low'
    low = RandomizedRegression(is_continuous=False, model_dir=low_md, log_l1_range=True)
    high_md = model_dir + '_high'
    high = RandomizedRegression(is_continuous=False, model_dir=high_md, log_l1_range=True)


    if low.log_l1_range:
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
            X, y_l, y_h = filter_by_stratification(X,us,static_info,len(vocab),s)

            print('{} size of y'.format(y_l.shape))
            low.fit_batch(X, y_l, l1, b)
            high.fit_batch(X, y_h, l1, b)
            if VERBOSE:
                print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))

    return low, high


def main(in_dir,out_dir):
    out_dir = os.path.join(out_dir, name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    static_info = pd.read_csv(os.path.join(in_dir,'static_info/static_user_info.csv'))
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    y_to_s_val = dict((i, s) for i, s in enumerate(specializations))
    vocab = from_gz(os.path.join(in_dir, 'features'), 'vocab')

    for s in specializations:
        model_dir = './log_reg_models_{}_{}'.format(name,s)

        low, high = fit_batches(in_dir, static_info,s, model_dir, l1_range=[0.0, 10.0])
        low_sf = low.get_salient_features(dict((v,k) for k,v in vocab.items()), {0:'low'},n=100)
        high_sf = high.get_salient_features(dict((v, k) for k, v in vocab.items()), {0: 'high'}, n=100)
        salient_features = dict(low_sf, **high_sf)

        with open(os.path.join(out_dir,'rlr_selected_features_{}_{}.txt'.format(name,s)),'w+') as f:
            for s, feat in salient_features.items():
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