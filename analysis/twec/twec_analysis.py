'''
Fit "compass" word2vec embeddings on all tweets, and then on different types of users.
'''

import argparse
import os
import pandas as pd
import re
import time

from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec

from twokenizer import tokenizeRawTweetText as tokenize

HAS_ALPHA = re.compile('.*[a-z].*')
SEED_WORDS = ['success', 'failure', 'content', 'travel']


def write_folds(full_df: pd.DataFrame, column: str, out_dir: str):
    start = time.time()
    full_df = full_df.sample(frac=1.)
    full_df['tokenized_tweet'] = full_df['text'].map(
        lambda text: [t for t in tokenize(text.lower())
                      if (HAS_ALPHA.match(t) and
                          not t.startswith('@') and
                          not t.startswith('http') and
                          not t.startswith('www') and
                          not t in {'rt'}) ]
    )
    print('({}s) Tokenized all tweets'.format(int(time.time() - start)))
    
    full_df = full_df[~pd.isna(full_df[column])]
    
    sliced_dfs = {k: df for k, df in full_df.groupby(column)}
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(os.path.join(out_dir, 'COMPASS.txt'), 'wt') as out_file:
        for tokens in full_df['tokenized_tweet']:
            out_file.write(' '.join(tokens) + ' ')
        
        print('({}s) Saved full data frame'.format(int(time.time() - start)))
    
    for key, df in sliced_dfs.items():
        df = df.sample(frac=1.)
        with open(os.path.join(out_dir, '{}.txt'.format(key)), 'wt') as out_file:
            for tokens in df['tokenized_tweet']:
                out_file.write(' '.join(tokens) + ' ')
        
        print('({}s) Saved category "{}"'.format(int(time.time() - start), key))

    return list(sliced_dfs.keys())


def main(input_path: str, category_to_split: str, slice_dir: str, k: int,
         training_objective: str, dynamic_iterations: int, static_iterations: int,
         negative_samples: int, window: int, alpha: float, workers: int, min_token_count: int,
         model_dir: str):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    start = time.time()
    
    full_df = pd.read_table(input_path, sep='\t')
    print('({}s) read full data frame'.format(int(time.time() - start)))
    
    category_keys = write_folds(full_df, category_to_split, slice_dir)
    
    aligner = TWEC(size=k, sg=0 if training_objective == 'cbow' else 1, siter=static_iterations,
                   diter=dynamic_iterations, ns=negative_samples, window=window,
                   alpha=alpha, min_count=min_token_count, workers=workers, opath=model_dir)
    aligner.train_compass(os.path.join(slice_dir, 'COMPASS.txt'), overwrite=False)
    print('({}s) finished training compass'.format(int(time.time() - start)))
    
    sliced_models = {}
    
    for category in category_keys:
        sliced_models[category] = aligner.train_slice(
            os.path.join(slice_dir, '{}.txt'.format(category)), save=True
        )
        print('({}s) finished training slice "{}"'.format(int(time.time() - start), category))
    
    for word in SEED_WORDS:
        top_word_df = pd.DataFrame({k: [w for w, score in m.similar_by_word(word, 20)]
                                    for k, m in sliced_models.items()})
        top_word_df.to_csv('similarTo_{}_k={}.tsv'.format(word, k),
                           header=True, index=False, sep='\t')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_path', required=True,
                      default='/Users/abenton10/additional_projects/twitter_brand/twitter_brand_workspace_20190417/promoting_user_tweets.merged_with_user_info.noduplicates.tsv.gz',
                      help='tweet table with user metadata, tab-separated')
  parser.add_argument('--slice_dir', required=True, default='./tweetText_by_mainCategory/',
                      help='where to save different slices of the data for adaptation')
  parser.add_argument('--model_dir', required=True, default='./twec_models/',
                      help='where to save the sliced models')
  parser.add_argument('--category_to_split', required=True, default='category_most_index-mace_label',
                      help='column in dataframe to split users by')
  parser.add_argument('-k', default=50, type=int, help='dimensionality of embeddings')
  parser.add_argument('--training_objective', default='cbow', choices=['cbow', 'skipgram'],
                      help='objective')
  parser.add_argument('--dynamic_iterations', default=5, type=int,
                      help='number of epochs to train on data subsets')
  parser.add_argument('--static_iterations', default=5, type=int,
                      help='number of epochs to train on the full dataset')
  parser.add_argument('--negative_samples', default=10, type=int, help='number of negative samples to take for each word')
  parser.add_argument('--window', default=5, type=int, help='number of words to predict left and right')
  parser.add_argument('--alpha', default=0.025, type=float, help='initial learning rate')
  parser.add_argument('--workers', default=2, type=int, help='number of worker threads')
  parser.add_argument('--min_token_count', default=3, type=int, help='minimum token count in corpus')
  args = parser.parse_args()
  
  main(args.input_path, args.category_to_split, args.slice_dir, args.k, args.training_objective,
       args.dynamic_iterations, args.static_iterations, args.negative_samples, args.window,
       args.alpha, args.workers, args.min_token_count, args.model_dir)
