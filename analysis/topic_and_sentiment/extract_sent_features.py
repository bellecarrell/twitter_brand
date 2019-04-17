'''
Use sentiment lexicon to attach sentiment score to each tweet.

Use the following sentiment lexicon:

Sentiment140 Lexicon
@InProceedings{MohammadKZ2013,
  author    = {Mohammad, Saif and Kiritchenko, Svetlana and Zhu, Xiaodan},
  title     = {NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets},
  booktitle = {Proceedings of the seventh international workshop on Semantic Evaluation Exercises (SemEval-2013)},
  month     = {June},
  year      = {2013},
  address   = {Atlanta, Georgia, USA}
}

Procedure used to tag for tweet sentiment is the same as that used in section 6.2.1 of:

Kiritchenko, S., Zhu, X., Mohammad, S. (2014). Sentiment Analysis of Short Informal Texts.  Journal of Artificial Intelligence Research, 50:723-762, 2014

We use the NRC Emoticon Affirmative and Negated Context Lexicons and take the sum of unigram sentiment scores as
the tweet sentiment.  Features occurring in a negated context (span from a negation word in a closed class
to punctuation or the end of the utterance) are treated separately from an affirmative context (everything else).
'''

from nltk.corpus import stopwords
import os
import pandas as pd
import re

from twokenize.twokenize import tokenizeRawTweetText as tokenize

WORKSPACE_DIR = '/exp/abenton/twitter_brand_workspace_20190417/'
TWEET_PATH = os.path.join(WORKSPACE_DIR, 'promoting_user_tweets.merged_with_user_info.noduplicates.tsv.gz')

# Under: /exp/abenton/twitter_brand/twitter_brand/analysis/topic_and_sentiment/
LEXICON_PATH = 'resources/Emoticon-AFFLEX-NEGLEX-unigrams.txt.gz'

SENT_DIR = os.path.join(WORKSPACE_DIR, 'sentiment')

NEGS = {"ain't", "aint", "can't", "cant", "couldn't", "didn't", "doesn't",
        "don't", "dont", "hasn't", "haven't", "never", "no", "not",
        "nothing", "won't"}
PUNCTS = {',', '.', ':', ';', '!', '?'}


def norm_token(t):
    if t.startswith('@'):
        return '<USER>'
    elif t.startswith('http'):
        return '<URL>'
    elif re.search('[a-z]', t) is None:
        return None
    
    t_repl_digits = re.sub('\d', '0', t)  # normalize digits
    
    return t_repl_digits

    
def normalize_tweet(tweet):
    '''
    Break tweet into normalized tokens and append _NEG or _NEGFIRST tags
    to tokens that appear in negated context since we are using a
    contextual lexicon.  Returns list of normalized tokens.
    '''
    
    tokens = [norm_token(t) for t in tokenize(tweet.lower())]
    return [t for t in tokens if t]


class UnsupervisedSentimentScorer:
    def __init__(self, contextual_lexicon_path=LEXICON_PATH, negs=NEGS, puncts=PUNCTS):
        self.contextual_lexicon_path = contextual_lexicon_path
        self.negs = negs
        self.puncts = puncts
        self.sentiment_lexicon = None
        
        self.load_unigram_lexicon()
    
    def load_unigram_lexicon(self):
        df = pd.read_table(self.contextual_lexicon_path, header=None, names=['feature',
                                                                             'score',
                                                                             'num_pos',
                                                                             'num_neg'])
        
        self.sentiment_lexicon = {r['feature']: r['score'] for ridx, r in df.iterrows()}
    
    def featurize(self, tweet):
        ''' Extract features belonging to our lexicon. '''
        
        extracted_features = []
        
        normed_tokens = normalize_tweet(tweet)
        
        is_neg_span = False
        prev_is_neg = False
        for t in normed_tokens:
            # use feature suffix to encode whether we are in a negation span
            if prev_is_neg:
                suffix = '_NEGFIRST'
            elif is_neg_span:
                suffix = '_NEG'
            else:
                suffix = ''
            
            feat = t + suffix
            if feat in self.sentiment_lexicon:
                extracted_features.append(feat)
            
            prev_is_neg = False
            if t in self.negs:
                prev_is_neg = True
                is_neg_span = True
            
            if t in self.puncts:
                is_neg_span = False
        
        return extracted_features
    
    def score(self, tweet):
        lex_features = self.featurize(tweet)
        return sum([self.sentiment_lexicon[f] for f in lex_features])
    
    def classify(self, tweet):
        tweet_score = self.score(tweet)
        
        if tweet_score > 0.0:
            return 1
        elif tweet_score < 0.0:
            return -1
        else:
            return 0


def main():
    sentiment_scorer = UnsupervisedSentimentScorer(LEXICON_PATH, NEGS, PUNCTS)
    
    df = pd.read_table(TWEET_PATH)
    
    df['tweet_sentiment_score'] = df['text'].map(lambda tweet: sentiment_scorer.score(tweet))
    df['tweet_sentiment_class'] = df['text'].map(lambda tweet: sentiment_scorer.classify(tweet))

    df.to_csv(
            os.path.join(SENT_DIR,
                         'promoting_user_tweets.with_lexiconbased_sentiment.noduplicates.tsv.gz'),
            compression='gzip',
            sep='\t',
            header=True,
            index=False)


if __name__ == '__main__':
    if not os.path.exists(SENT_DIR):
        os.mkdir(SENT_DIR)
    
    main()
