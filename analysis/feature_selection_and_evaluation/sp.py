import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
import logging
from analysis.rlr import *
logging.basicConfig(level=logging.INFO)
import nltk
import os

from analysis.batching import *
from analysis.feature_selection_and_evaluation.exp_util import *
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




def main(in_dir,out_dir):
    out_dir = os.path.join(out_dir, name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    static_info = pd.read_csv(os.path.join(in_dir,'static_info/static_user_info.csv'))
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    y_to_s_val = dict((i, s) for i, s in enumerate(specializations))
    vocab = from_gz(os.path.join(in_dir, 'features'), 'vocab')


    #batch_tws = ['all_data', 'month', 'one_week', 'two_week']
    batch_tws = ['month']

    for tw in batch_tws:


        model_dir = '/home/hltcoe/acarrell/PycharmProjects/twitter_brand/analysis/models/svm_log_reg_models_{}_{}'.format(name,tw)

        ensemble = Ensemble(model_dir)
        X, us = load_test(in_dir, vocab, static_info)
        # #X, us = load_fold(in_dir, vocab, static_info,'dev')
        y = _generate_y_multi(us, static_info)
        ensemble.eval_to_file(X,y, 1, os.path.join(out_dir,'svm_eval_{}_batch_{}'.format(name,tw)))

    
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