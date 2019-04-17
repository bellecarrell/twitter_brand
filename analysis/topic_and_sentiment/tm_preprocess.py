'''
Extract normalized unigrams for topic modeling.
'''

from nltk.corpus import stopwords
import json
import os
import pandas as pd
import re
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import time
from twokenize.twokenize import tokenizeRawTweetText as tokenize

stop = set(stopwords.words('english')) | set(string.punctuation) | {'rt'}


def all_tweets_by_user(users, dates_tweets):
    return (' '.join([dt[1] for dt in dates_tweets[u]]) for u in users)


def stringify_dict(d):
    return dict((str(k), str(v)) for k, v, in d.items())


def norm_token(t):
    # t = t.lower() ## AB: assume sentence is already lower-cased
    
    if t.startswith('@'):
        return '<USER>'
    elif t.startswith('http'):
        return '<URL>'
    elif t in stop:
        return None
    elif re.search('[a-z]', t) is None:
        return None
    
    t_repl_digits = re.sub('\d', '0', t)  # normalize digits
    
    return t_repl_digits


def featurize_tweet(t):
    toks = tokenize(t.lower())
    norm_toks = [norm_token(t) for t in toks]
    norm_toks = [t for t in norm_toks if t]
    
    feats = list(set(norm_toks))
    
    return feats


def extract_vocab_and_features(tweets_merged_with_uinfo_path, out_dir, max_rows=None, min_df=10, max_df=0.5):
    timeline_df = pd.read_table(tweets_merged_with_uinfo_path, sep=',')
    only_promoting_df = timeline_df
    
    if max_rows is not None:
        only_promoting_df = only_promoting_df.head(max_rows)
    
    n = only_promoting_df.shape[0]
    
    start = time.time()
    unigrams_per_tweet = []
    
    for tidx, tweet in enumerate(only_promoting_df.text):
        features = featurize_tweet(tweet)
        unigrams_per_tweet.append(features)
        
        if not (tidx % 10000):
            print('Featurized tweet {:.2f}M/{:.2f}M ({}s)'.format(tidx / 10 ** 6, n / 10 ** 6,
                                                                  int(time.time() - start)))
    
    vectorizer = TfidfVectorizer(stop_words=stop, ngram_range=(1, 1),
                                 min_df=min_df, max_df=max_df)
    feature_matrix = vectorizer.fit_transform(unigrams_per_tweet)
    
    vocab = vectorizer.vocabulary_
    vocab = stringify_dict(vocab)
    
    json.dump(vocab, os.path.join(out_dir, 'vocab.txt'))
    print('Restricted to vocabulary size of {}'.format(len(vocab)))
    
    scipy.sparse.save_npz(os.path.join(out_dir, 'topic_modeling_per_tweet.unigram_idf.npz'), feature_matrix)
    print('Saved unigram IDF feature matrix for topic modeling')
    
    only_promoting_df[['tweet_id', 'user_id']].to_csv(os.path.join(out_dir,
                                                                   'topic_modeling_per_tweet.index.tsv'),
                                                      sep='\t', header=True, index=False)
    print('Saved feature matrix index <tweet_id, user_id>')
    
    # filter vocabulary
    filtered_unigrams_per_tweet = [[vocab[f] for f in features if f in vocab]
                                   for features in unigrams_per_tweet]
    only_promoting_df['text_filtered_unigrams'] = filtered_unigrams_per_tweet
    print('Add filtered unigrams to table')
    
    return only_promoting_df


def main(tweet_merged_with_uinfo_path, out_dir, min_df=10, max_df=0.5):
    promoting_df_with_features = extract_vocab_and_features(tweet_merged_with_uinfo_path,
                                                            out_dir,
                                                            min_df=min_df,
                                                            max_df=max_df)
    
    promoting_df_with_features.to_csv(
            os.path.join(out_dir,
                         'promoting_user_tweets.with_extracted_tweet_features.noduplicates.tsv.gz'),
            compression='gzip',
            sep='\t',
            header=True,
            index=False)
    print('Wrote out table including unigram features')


if __name__ == '__main__':
    WORKSPACE_DIR = '/exp/abenton/twitter_brand_workspace/20190417/'
    main(os.path.join(WORKSPACE_DIR, 'promoting_user_tweets.merged_with_user_info.noduplicates.tsv.gz'),
         WORKSPACE_DIR, min_df=10, max_df=0.5)
