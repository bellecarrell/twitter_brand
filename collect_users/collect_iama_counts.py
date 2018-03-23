#!/usr/bin/env python

import logging
import sys
import os
from collections import Counter
import pickle

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out = sys.argv[2]

    total_unigrams = Counter()
    total_bigrams = Counter()
    total_trigrams = Counter()

    unigram = 'unigram'
    bigram = 'bigram'
    trigram = 'trigram'

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.p'):
                counts = pickle.load(open(os.path.join(root, file), "rb"))
                if unigram in file:
                    total_unigrams = total_unigrams + counts
                if bigram in file:
                    total_bigrams = total_bigrams + counts
                if trigram in file:
                    total_trigrams = total_trigrams + counts

    unigrams_sorted = sorted(total_unigrams, key=total_unigrams.get, reverse=True)
    bigrams_sorted = sorted(total_bigrams, key=total_bigrams.get, reverse=True)
    trigrams_sorted = sorted(total_trigrams, key=total_trigrams.get, reverse=True)

    unigram_out = open(out + "iama_out_unigram.p", 'wb+')
    bigram_out = open(out + "iama_out_bigram.p", 'wb+')
    trigram_out = open(out + "iama_out_trigram.p", 'wb+')

    pickle.dump(total_unigrams, unigram_out)
    pickle.dump(total_bigrams, bigram_out)
    pickle.dump(total_trigrams, trigram_out)

    print(unigrams_sorted)
    print(bigrams_sorted)
    print(trigrams_sorted)

    uni_kv_out = out + "uni_kv.tsv"
    bi_kv_out = out + "bi_kv.tsv"
    tri_kv_out = out + "tri_kv.tsv"

    with open(uni_kv_out, 'w+') as f:
        for ng in unigrams_sorted:
            f.write(ng + "\t" + str(total_unigrams[ng]) + "\n")

    with open(bi_kv_out, 'w+') as f:
        for ng in bigrams_sorted:
            f.write(ng + "\t" + str(total_bigrams[ng]) + "\n")

    with open(tri_kv_out, 'w+') as f:
        for ng in trigrams_sorted:
            f.write(ng + "\t" + str(total_trigrams[ng]) + "\n")
