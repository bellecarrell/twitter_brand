import argparse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
import time
import csv
import gzip
import os
from collections import defaultdict
import datetime
import pandas as pd
import json
from twokenize.twokenize import tokenizeRawTweetText as tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import re

stemmer = PorterStemmer()
stop = set(stopwords.words('english')) | set(string.punctuation) | ['rt']

def posted_recently(collection_date,user_date):
    return datetime.datetime.fromtimestamp(collection_date) - datetime.timedelta(days=60) <= datetime.datetime.fromtimestamp(user_date)


# def filter_inactive_users(users, dates_tweets,most_recent):
#     print('Filtering users with {} most recent'.format(most_recent))
#     recent_users_removed = 0
#     all_users = copy.deepcopy(list(dates_tweets.keys()))
#     for user in all_users:
#         u_date = dates[user]
#         if not posted_recently(most_recent, u_date):
#             del dates_tweets[user]
#             del dates[user]
#             recent_users_removed += 1
#     print('{} users removed'.format(recent_users_removed))


def dated_tweets_by_user(f, users):
    """
    Get tweets from the timeline compressed file and place in dict sorted by user.
    :param in_dir: top-level directory containing promoting user dfs
    :return: dict of tweets by user ID
    """
    reader = csv.reader(f)
    dates_tweets = defaultdict(list)
    dates = defaultdict(int)
    most_recent = 0
    count = 0

    print('Reading rows from timeline file to filter users')
    for row in reader:
        if count == 0:
            count += 1
            continue
        tweet_id, created_at, text, user_id = row
        created_at, user_id = int(created_at), int(user_id)
        if user_id in users:
            dates_tweets[user_id].append((created_at, text))
            if created_at > dates[user_id]:
                dates[user_id] = created_at
            if created_at > most_recent:
                most_recent = created_at
        count += 1
        if count % 100000 == 0:
            print('Read {} rows'.format(count))

    return dates_tweets


def all_tweets_by_user(users,dates_tweets):
    return (' '.join([dt[1] for dt in dates_tweets[u]]) for u in users)


def longest_doc(corpus):
    max = 0
    for doc in corpus:
        toks = word_tokenize(doc)
        if len(toks) > max:
            max = len(toks)
    return max


def stringify_dict(d):
    return dict((str(k),str(v)) for k, v, in d.items())


def to_json_file(data, out_dir, fname):

    json_str = json.dumps(data) + "\n"  # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')  # 3. bytes (i.e. UTF-8)

    with gzip.GzipFile(os.path.join(out_dir,'{}.json.gz'.format(fname)), 'w') as fout:  # 4. gzip
        fout.write(json_bytes)


def norm_token(t):
    # t = t.lower() ## AB: I assume sentence is already tokenized
    
    if t.startswith('@'):
        return '<USER>'
    elif t.startswith('http'):
        return '<URL>'
    
    if t in stop:
        return None
    
    t_repl_digits = re.sub('\d', '0', t)  # normalize digits
    t_stemmed = stemmer.stem(t_repl_digits)  # stem words
    
    return t_stemmed


def featurize_tweet(t):
    toks = tokenize(t.lower())
    norm_toks = [norm_token(t) for t in toks]
    norm_toks = [t for t in norm_toks if t]
    
    feats = {pair for pair in zip(norm_toks, norm_toks[1:])}
    feats |= {(t,) for t in norm_toks}
    
    return feats


def extract_vocab_and_features(promoting_users, timeline_path, min_df=50, max_df=0.8):
    timeline_df = pd.read_table(timeline_path, sep=',')
    only_promoting_df = timeline_df[timeline_df['user_id'].isin(promoting_users)]
    n = only_promoting_df.shape[0]
    
    full_vocab = {}
    ngramset_per_tweet = []
    
    start = time.time()
    
    for tidx, tweet in enumerate(only_promoting_df.text):
        features = featurize_tweet(tweet)
        ngramset_per_tweet.append(features)
        
        for f in features:
            if f not in full_vocab:
                full_vocab[f] = 0
            full_vocab[f] += 1
        
        if not (tidx % 10000):
            print('Featurized tweet {:.2f}M/{:.2f}M ({}s)'.format(tidx/10**6, n/10**6, int(time.time() - start)))
    
    # filter vocabulary
    filtered_vocab = {f for f, c in full_vocab.items() if (c >= min_df) and ((c / n) <= max_df)}
    vocab_key = dict(enumerate(sorted(list(filtered_vocab))))
    rev_vocab_key  = {v: k for k, v in vocab_key.items()}
    
    print('Filtered vocabulary from {} to {} types'.format(len(full_vocab), len(filtered_vocab)))
    
    filtered_ngramset_per_tweet = [{rev_vocab_key[f] for f in features if f in filtered_vocab}
                                   for features in ngramset_per_tweet]
    only_promoting_df['extracted_features'] = filtered_ngramset_per_tweet
    
    return rev_vocab_key, only_promoting_df


def featurize_by_vectorizer():
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    tweet_f = gzip.open(os.path.join(in_dir, 'timeline/tweet_texts.csv.gz'), 'rt')
    timeline = gzip.open(os.path.join(in_dir, 'timeline/user_tweets.noduplicates.tsv.gz'), 'rt')
    users = static_info.loc[static_info['classify_account-mace_label'] == 'promoting'][
            'user_id'].dropna().unique().tolist()
    dates_tweets = dated_tweets_by_user(timeline, users)
    corpus = all_tweets_by_user(users, dates_tweets)
    
    # todo: remove usernames, normalize numbers, and lemmatize
    sw = set(stopwords.words('english'))
    sw.union({c for c in list(string.punctuation) if c is not "#" and c is not "@"})
    
    vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=sw, ngram_range=(1, 2),
                                 min_df=50, max_df=0.8)
    vectorizer.fit(corpus)
    
    vocab = vectorizer.vocabulary_
    vocab = stringify_dict(vocab)
    
    to_json_file(vocab, os.path.join(out_dir), 'vocab')
    to_json_file(dates_tweets, os.path.join(out_dir), 'dates_tweets')


def main(in_dir, out_dir, min_df=50, max_df=0.8):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    timeline_path = os.path.join(in_dir, 'timeline/user_tweets.noduplicates.tsv.gz')
    promoting_users = static_info.loc[
        static_info['classify_account-mace_label'] == 'promoting'
    ]['user_id'].dropna().unique().tolist()
    
    print('Featurizing tweets for {}/{} self-promoting users'.format(len(promoting_users), static_info.shape[0]))
    rev_vocab_key, promoting_df = extract_vocab_and_features(promoting_users,
                                                             timeline_path,
                                                             min_df=min_df,
                                                             max_df=max_df)
    
    import pdb; pdb.set_trace()
    
    to_json_file({str(k): v for k, v in rev_vocab_key.items()}, out_dir, 'vocab')
    promoting_df[['tweet_id',
                  'created_at',
                  'user_id',
                  'extracted_features']].to_csv(
            os.path.join(out_dir, 'user_features_per_tweet.noduplicates.tsv.gz'), compression='gzip', sep='\t',
            header=True,
            index=False)


if __name__ == '__main__':
    """
    build a vocabulary from corpus and save to file.
    documents per user that are promoting in the static_info file.
    """
    
    parser = argparse.ArgumentParser(
        description='build vocabulary and extracted features for each tweet'
    )
    parser.add_argument('--input_dir', required=True,
                        dest='input_dir', metavar='INPUT_DIR',
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs. should have vocab')
    parser.add_argument('--out_dir', required=True,
                        dest='out_dir', metavar='OUTPUT_PREFIX',
                        help='output directory')
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.out_dir

    main(in_dir, out_dir)