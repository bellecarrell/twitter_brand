#!/usr/bin/env python

import sys
from collections import Counter
import os
import gzip
import re
import logging
import pickle

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    """
    Reads in Twitter stream from specified dir and counts freqs of all "I am a/I'm a" matches
    unigrams, bigrams, and trigrams

    Args:
    twitter_dir: Directory containing items 
    user_dir: Directory to put user data file in
    """
    twitter_dir = sys.argv[1]
    out_dir = sys.argv[2]

    print("in file")

    total_unigrams = Counter()
    total_bigrams = Counter()
    total_trigrams = Counter()

    user_p = re.compile(r'"user":.*?{(.+?)}')
    desc_p = re.compile(r'"description":.*?"(.+?)"')
    iama_p = re.compile(r"I am a (\w+\W\w+\W\w+)|I'm a (\w+\W\w+\W\w+)")

    for dirpath, _, filenames in os.walk(twitter_dir):
        for filename in filenames:
            with gzip.open(os.path.join(dirpath, filename), 'rt') as f:
                try:
                    for line in f:
                        chunks = text_p.findall(line)

                        for chunk in chunks:

                            for no_comma, comma in iama_p.findall(chunk):
                                if no_comma:
                                    token = re.split("\W", no_comma)
                                    token = [tok.lower() for tok in token]

                                    unigrams = Counter(token)
                                    bigrams = Counter([' '.join(token[:2]), ' '.join(token[-2:])])
                                    trigrams = Counter([' '.join(token)])

                                    total_unigrams = total_unigrams + unigrams
                                    total_bigrams = total_bigrams + bigrams
                                    total_trigrams = total_trigrams + trigrams
                                if comma:
                                    token = re.split("\W", comma)
                                    token = [tok.lower() for tok in token]

                                    unigrams = Counter(token)
                                    bigrams = Counter([' '.join(token[:2]), ' '.join(token[-2:])])
                                    trigrams = Counter([' '.join(token)])

                                    total_unigrams = total_unigrams + unigrams
                                    total_bigrams = total_bigrams + bigrams
                                    total_trigrams = total_trigrams + trigrams

                except IOError as e:
                    logging.info(e)

    unigram_out = open(out_dir + "iama_out_unigram.p", 'wb+')
    bigram_out = open(out_dir + "iama_out_bigram.p", 'wb+')
    trigram_out = open(out_dir + "iama_out_trigram.p", 'wb+')

    pickle.dump(total_unigrams, unigram_out)
    pickle.dump(total_bigrams, bigram_out)
    pickle.dump(total_trigrams, trigram_out)

    unigram_out.close()
    bigram_out.close()
    trigram_out.close()



