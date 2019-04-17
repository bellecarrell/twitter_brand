'''
Fit topic model to promoting user tweets and extract topic distribution per tweet
(compare NMF and LDA qualitatively, tuning hyperparameters for heldout perplexity).
'''

import os
import multiprocessing as mp
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation, NMF

import scipy.sparse

PROP_TRAIN = 0.9

WORKSPACE_DIR = '/exp/abenton/twitter_brand_workspace_20190417/'
TWEET_PATH = os.path.join(WORKSPACE_DIR, 'promoting_user_tweets.merged_with_user_info.noduplicates.tsv.gz')
IDF_FEATURE_PATH = os.path.join(WORKSPACE_DIR, 'topic_modeling_per_tweet.unigram_idf.npz')
IDF_FEATURE_IDX_PATH = os.path.join(WORKSPACE_DIR, 'topic_modeling_per_tweet.index.tsv')
VOCAB_PATH = os.path.join(WORKSPACE_DIR, 'vocab.json')

TOPIC_DIR = os.path.join(WORKSPACE_DIR, 'topic_modeling')


def fit_nmf(train_max, heldout_max=None, vocab=None, k=10, alpha_regularization=0.0):
    nmf = NMF(k, alpha=alpha_regularization, verbose=True)
    
    train_nmf = nmf.fit_transform(train_max)
    
    if heldout_max is not None:
        heldout_max
    
    import pdb; pdb.set_trace()


def fit_lda(train_max, heldout_max=None, vocab=None, k=10, alpha=1.0, beta=10**-3):
    pass


def fit_model(args):
    if args[0] == 'nmf':
        fit_nmf(args[1], args[2], args[3], args[4])
    elif args[0] == 'lda':
        fit_lda(args[1], args[2], args[3], args[4], args[5])
    else:
        raise Exception('Do not recognize model "{}"'.format(args[0]))
    
    pass


def main():
    all_max = scipy.sparse.load_npz(IDF_FEATURE_PATH)
    
    if PROP_TRAIN < 1.0:
        split_idx = int(PROP_TRAIN*all_max.shape[0])
        train_max = all_max[:split_idx, :]
        heldout_max = all_max[split_idx:, :]
    else:
        train_max = all_max
        heldout_max = None
    
    arg_lst = []
    
    for k in [10, 20, 50, 100]:
        # NMF runs
        for alpha_reg in [0.0, 1.0, 10.0]:
            arg_lst.append(('nmf', train_max, heldout_max, k, alpha_reg))
        
        # LDA runs
        #arg_lst.append(('lda', train_max, heldout_max, k, 0.5, 10**-3))
    
    for i, args in enumerate(arg_lst):
        fit_model(args)
        
        print('Finished {}/{}'.format(i+1, len(arg_lst)))


if __name__ == '__main__':
    if not os.path.exists(TOPIC_DIR):
        os.mkdir(TOPIC_DIR)
    
    main()
