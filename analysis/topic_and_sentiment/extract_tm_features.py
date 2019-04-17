'''
Fit topic model to promoting user tweets and extract topic distribution per tweet
(compare NMF and LDA qualitatively, tuning hyperparameters for heldout
perplexity/reconstruction error).
'''

import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import time

from sklearn.decomposition import LatentDirichletAllocation, NMF

import scipy.sparse

PROP_TRAIN = 0.9

WORKSPACE_DIR = '/exp/abenton/twitter_brand_workspace_20190417/'
TWEET_PATH = os.path.join(WORKSPACE_DIR, 'promoting_user_tweets.merged_with_user_info.noduplicates.tsv.gz')
IDF_FEATURE_PATH = os.path.join(WORKSPACE_DIR, 'topic_modeling_per_tweet.unigram_idf.npz')
IDF_FEATURE_IDX_PATH = os.path.join(WORKSPACE_DIR, 'topic_modeling_per_tweet.index.tsv')
VOCAB_PATH = os.path.join(WORKSPACE_DIR, 'vocab.json')

TOPIC_DIR = os.path.join(WORKSPACE_DIR, 'topic_modeling')


def get_top_words(model, vocab, n=10, verbose=False):
    words_per_topic = []
    
    for topic_idx, topic in enumerate(model.components_):
        words = [vocab[i] for i in topic.argsort()[:-n - 1:-1]]
        words_per_topic.append(words)
        
        if verbose:
            message  = 'Topic #{}: '.format(topic_idx)
            message += ' '.join(words)
            print(message)
    
    return words_per_topic


def fit_nmf(train_max, heldout_max=None, vocab=None, k=10, alpha_regularization=0.0):
    nmf = NMF(k, alpha=alpha_regularization, verbose=False)
    
    train_nmf = nmf.fit_transform(train_max)
    
    if heldout_max is not None:
        heldout_nmf = nmf.transform(heldout_max)
    else:
        heldout_nmf = None
    
    batch_size = 100
    
    prop_train_reconst_errs = []
    prop_heldout_reconst_errs = []
    
    for iteration in range(20):
        train_idxes = np.random.choice(train_nmf.shape[0], batch_size, replace=False)
        
        reconst_train_max = nmf.inverse_transform(train_nmf[train_idxes])
        
        prop_train_reconst_err = np.linalg.norm(reconst_train_max - train_max[train_idxes]) / \
                                 scipy.sparse.linalg.norm(train_max[train_idxes])
        prop_train_reconst_errs.append(prop_train_reconst_err)
        
        if heldout_nmf is not None:
            heldout_idxes = np.random.choice(heldout_nmf.shape[0], batch_size, replace=False)
            
            reconst_heldout_max = nmf.inverse_transform(heldout_nmf[heldout_idxes])
            prop_heldout_reconst_err = np.linalg.norm(reconst_heldout_max - heldout_max[heldout_idxes]) / \
                                       scipy.sparse.linalg.norm(heldout_max[heldout_idxes])
            prop_heldout_reconst_errs.append(prop_heldout_reconst_err)
        else:
            prop_heldout_reconst_errs.append(-1.0)
    
    print('Train reconstruction error: {}'.format(np.mean(prop_train_reconst_errs)))
    print('Heldout reconstruction error: {}'.format(np.mean(prop_heldout_reconst_errs)))
    
    top_words_per_topic = get_top_words(nmf, vocab, n=20, verbose=False)
    
    topic_path = os.path.join(TOPIC_DIR, 'nmf-k{}-alpha{}.topics.txt'.format(k, alpha_regularization))
    topic_dist_path = os.path.join(TOPIC_DIR, 'nmf-k{}-alpha{}.topic_distribution_per_tweet.txt'.format(k, alpha_regularization))
    model_path = os.path.join(TOPIC_DIR, 'nmf-k{}-alpha{}.model.pickle'.format(k, alpha_regularization))
    
    # save top words per topic
    with open(topic_path, 'wt', encoding='utf8') as topic_file:
        for topic_idx, words in enumerate(top_words_per_topic):
            topic_file.write('Topic #{}:'.format(topic_idx))
            for w in words:
                topic_file.write(' ')
                topic_file.write(w)
            topic_file.write('\n')
    
    # save NMF model
    with open(model_path, 'wb') as model_file:
        pickle.dump(nmf, model_file)
    
    # save topic activation for each tweet, compressed numpy format
    if heldout_nmf is not None:
        all_nmf = np.concatenate((train_nmf, heldout_nmf), axis=0)
    else:
        all_nmf = train_nmf
    
    np.savez_compressed(topic_dist_path, topics_per_tweet=all_nmf)


def fit_lda(train_max, heldout_max=None, vocab=None, k=10, alpha=1.0, beta=10**-3):
    lda = LatentDirichletAllocation(k, alpha, beta)
    
    train_topic_probs = lda.fit_transform(train_max)

    if heldout_max is not None:
        heldout_topic_probs = lda.transform(heldout_max)
    else:
        heldout_topic_probs = None
    
    train_ppl_varlowerbound   = lda.perplexity(train_max)
    heldout_ppl_varlowerbound = lda.perplexity(heldout_max)
    
    top_words_per_topic = get_top_words(lda, vocab, n=20, verbose=False)
    
    topic_path = os.path.join(TOPIC_DIR, 'lda-k{}-alpha{}-beta{}.topics.txt'.format(k, alpha, beta))
    topic_dist_path = os.path.join(TOPIC_DIR,
                                   'lda-k{}-alpha{}-beta{}.topic_distribution_per_tweet.txt'.format(k,
                                                                                                    alpha,
                                                                                                    beta))
    model_path = os.path.join(TOPIC_DIR, 'nmf-k{}-alpha{}-beta{}.model.pickle'.format(k, alpha, beta))

    # save top words per topic
    with open(topic_path, 'wt', encoding='utf8') as topic_file:
        for topic_idx, words in enumerate(top_words_per_topic):
            topic_file.write('Topic #{}:'.format(topic_idx))
            for w in words:
                topic_file.write(' ')
                topic_file.write(w)
            topic_file.write('\n')

    # save model
    with open(model_path, 'wb') as model_file:
        pickle.dump(lda, model_file)

    # save topic activation for each tweet, compressed numpy format
    if heldout_topic_probs is not None:
        topic_probs = np.concatenate((train_topic_probs,
                                      heldout_topic_probs),
                                     axis=0)
    else:
        topic_probs = train_topic_probs

    np.savez_compressed(topic_dist_path, topics_per_tweet=topic_probs)

    pass


def main():
    # read in text matrix
    all_max = scipy.sparse.load_npz(IDF_FEATURE_PATH)
    onehot_max = all_max.copy()
    onehot_max[onehot_max.nonzero()] = 1
    
    if PROP_TRAIN < 1.0:
        split_idx = int(PROP_TRAIN*all_max.shape[0])
        train_max = all_max[:split_idx, :]
        heldout_max = all_max[split_idx:, :]
        
        onehot_train_max   = onehot_max[:split_idx, :]
        onehot_heldout_max = onehot_max[split_idx:, :]
    else:
        train_max = all_max
        heldout_max = None
        
        onehot_train_max   = onehot_max
        onehot_heldout_max = None
    
    # read in vocabulary
    with open(VOCAB_PATH, 'rt') as vocab_file:
        vocab = json.load(vocab_file)
        rev_vocab = {int(v): k for k, v in vocab.items()}
    
    arg_lst = []
    
    for k in [10, 20, 50, 100]:
        # NMF runs
        for alpha_reg in [0.0, 1.0, 10.0]:
            arg_lst.append(('nmf', k, alpha_reg))
        
        # LDA runs
        arg_lst.append(('lda', k, 0.5, 10.**-3))
    
    for i, args in enumerate(arg_lst):
        start = time.time()
        
        if args[0] == 'nmf':
            fit_nmf(train_max, heldout_max, vocab, args[1], args[2])
        elif args[0] == 'lda':
            fit_lda(onehot_train_max, onehot_heldout_max, rev_vocab, args[1], args[2], args[3])
        else:
            raise Exception('Do not recognize model "{}"'.format(args[0]))
        
        end = time.time()
        print('({}s) Finished {}/{} ({})'.format(int(end - start), i+1, len(arg_lst), args))


if __name__ == '__main__':
    if not os.path.exists(TOPIC_DIR):
        os.mkdir(TOPIC_DIR)
    
    main()
