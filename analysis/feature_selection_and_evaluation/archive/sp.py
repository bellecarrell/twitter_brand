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
name = 'sp'
n_batches = 100

def _generate_y_multi(us, static_info):
    y = np.array([0 for u in us])
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    s_to_y_val = dict((s, i) for i, s in enumerate(specializations))
    for i, u in enumerate(us):
        specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]
        y[i] = s_to_y_val[specialization]

    return y


def fit_batches(rr, in_dir, static_info, batch_tw='month', l1_range=[0.0, 1.0]):
    np.random.seed(SEED)

    vocab = from_gz(os.path.join(in_dir,'features'),'vocab')

    if rr.log_l1_range:
        l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
    else:
        l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

    start = time.time()

    # fit each batch
    for dirpath, _, filenames in os.walk(os.path.join(in_dir,'batches','{}_batches'.format(batch_tw))):
        filenames = [f for f in filenames if f.startswith('batch')]
        for b, (filename, l1) in enumerate(zip(filenames,l1s)):
            data = np.load(os.path.join(dirpath,filename))
            X, us, tw = load_train(data, len(vocab),static_info)
            y = _generate_y_multi(us, static_info)

            print('{} size of y'.format(y.shape))
            rr.fit_batch(X, y, l1, b)
            if VERBOSE:
                print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))


def main(in_dir,out_dir):
    static_info = pd.read_csv(os.path.join(in_dir,'static_info/static_user_info.csv'))
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    y_to_s_val = dict((i, s) for i, s in enumerate(specializations))
    vocab = from_gz(os.path.join(in_dir, 'features'), 'vocab')


    batch_tws = ['all_data', 'month', 'one_week', 'two_week']

    for tw in batch_tws:
        model_dir = './log_reg_models_{}_{}'.format(name,tw)

        log_reg = RandomizedRegression(is_continuous=False, model_dir=model_dir, log_l1_range=True)
        fit_batches(log_reg, in_dir, static_info, batch_tw=tw, l1_range=[0.0, 10.0])
        salient_features = log_reg.get_salient_features(dict((v,k) for k,v in vocab.items()), y_to_s_val,n=1300)

        with open(os.path.join(out_dir,'rlr_selected_features_{}_{}.txt'.format(name,tw)),'w+') as f:
            for s, feat in salient_features.items():
                f.write('Specialization: {} Salient features: {}\n\n'.format(s, feat))

        ensemble = Ensemble(model_dir)
        X, us = load_test(in_dir, vocab, static_info)
        y = _generate_y_multi(X, static_info)
        ensemble.eval_to_file(X,y, os.path.join(out_dir,'eval_{}_batch_{}'.format(name,tw)))

    
if __name__ == '__main__':
    """
    Run randomized logistic regression to see features selected for different specializations.

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