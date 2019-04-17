'''
Fit topic model to promoting user tweets and extract topic distribution per tweet
(compare NMF and LDA qualitatively, tuning hyperparameters for heldout perplexity).
'''

import os
import multiprocessing as mp
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy.sparse

PROP_TRAIN = 0.9

WORKSPACE_DIR = '/exp/abenton/twitter_brand_workspace_20190417/'
TWEET_PATH = os.path.join(WORKSPACE_DIR, 'promoting_user_tweets.merged_with_user_info.noduplicates.tsv.gz')
IDF_FEATURE_PATH = os.path.join(WORKSPACE_DIR, 'topic_modeling_per_tweet.unigram_idf.npz')
IDF_FEATURE_IDX_PATH = os.path.join(WORKSPACE_DIR, 'topic_modeling_per_tweet.index.tsv')

TOPIC_DIR = os.path.join(WORKSPACE_DIR, 'topic_modeling')


def fit_nmf(train_max, heldout_max=None, k=10, alpha_regularization=0.0):
    pass


def fit_lda(train_max, heldout_max=None, k=10, alpha=1.0, beta=10**-3):
    pass


def fit_model(args):
    if args[0] == 'nmf':
        fit_nmf()
        pass
    elif args[0] == 'lda':
        pass
    else:
        raise Exception('Do not recognize model "{}"'.format(args[0]))
    
    pass


def main(num_procs=4):
    all_max = scipy.sparse.load_npz(IDF_FEATURE_PATH)
    
    if PROP_TRAIN < 1.0:
        split_idx = int(PROP_TRAIN*all_max.shape[0])
        train_max = all_max[:split_idx, :]
        heldout_max = all_max[split_idx:, :]
    else:
        train_max = all_max
        heldout_max = None
    
    args = []
    
    for k in [10, 20, 50, 100]:
        # NMF runs
        for alpha_reg in [0.0, 1.0, 10.0]:
            args.append(('nmf', k, alpha_reg))
        
        # LDA runs
        args.append(('lda', k, 0.5, 10**-3))


if __name__ == '__main__':
    if not os.path.exists(TOPIC_DIR):
        os.mkdir(TOPIC_DIR)
    
    main()
