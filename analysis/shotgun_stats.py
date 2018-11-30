import argparse
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from collections import Counter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy
import math

#https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def tf_idf_specialization_percentile(static_info, timeline_dir, out_dir):
    """
    Stats 1: TF-IDF per specialization
    Treat each specialization as a document (all tweets made by users with that specialization).
Calculate TF-IDF over token types. Report highest-ranked types for each specialization. Do
similarly but splitting between high and low follower count users
    """

    grouping_cols = ['percentile', 'category_most_index-mace_label']

    for gc in grouping_cols:
        docs_dir = os.path.join(out_dir,'docs')
        groups = static_info[gc].dropna().unique().tolist()

        for g in groups:
            with open(os.path.join(docs_dir,'{}_{}.txt'.format(gc,g)),'w+') as g_document:
                g_users = static_info.loc[static_info[gc] == g]['user_id'].values

                for u in g_users:
                    u_timeline_fname = os.path.join(timeline_dir, '{}.csv'.format(u))

                    if os.path.isfile(u_timeline_fname):
                        with open(u_timeline_fname, 'r') as f:
                            u_timeline_df = pd.read_csv(f)
                            u_tweets = u_timeline_df['text'].values.tolist()
                            g_document.write(' '.join(u_tweets))

        g_corpus = [os.path.join(docs_dir,f) for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f)) and gc in f]
        my_stop_words = text.ENGLISH_STOP_WORDS.union(['and','or','not','of'])
        vectorizer = TfidfVectorizer(input='filename',stop_words=text.ENGLISH_STOP_WORDS,min_df=0.3)
        X = vectorizer.fit_transform(g_corpus)
        #vec = vectorizer.named_steps['vec']
        features = vectorizer.get_feature_names()

        for i, g in enumerate(groups):
            top_tfidf = top_feats_in_doc(X, features, i)
            top_tfidf.to_csv(os.path.join(out_dir,'{}_{}_top_tfidf.csv'.format(gc,g)))

def follower_count_bins_by_specialization(static_info, output_dir):
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()

    for s in specializations:
        percentile_counts = Counter(static_info.loc[static_info['category_most_index-mace_label'] == s]['percentile'])
        percentiles = [int(p) for p in percentile_counts.keys()]
        counts = list(percentile_counts.values())
        percentiles, counts = zip(*sorted(zip(percentiles, counts)))
        y_pos = np.arange(len(percentiles))

        plt.bar(y_pos, counts, align='center', alpha=0.5)
        plt.xticks(y_pos, percentiles)
        plt.ylabel('Number of users')
        plt.title('Number of users by follower count percentile for {} specialization'.format(s))

        s_fname = os.path.join(output_dir,'{}_follower_count_bins.png'.format(s))
        plt.savefig(s_fname)
        plt.close()


def correlation_matrix(static_info, output_dir):
    specializations = [col for col in static_info.columns.values if col.startswith('category_all') and col.endswith('mace_label')]
    m = len(specializations)

    correlation_mat = pd.DataFrame(index=specializations,columns=specializations)

    for i, _ in enumerate(specializations):
        for j, _ in enumerate(specializations):
            s_i = []
            for v in static_info[specializations[i]].values:
                if type(v) == float:
                    s_i.append(2)
                elif v == 'y':
                    s_i.append(1)
                elif v == 'n':
                    s_i.append(0)
            s_j = []
            for v in static_info[specializations[j]].values:
                #hacky check for nan
                if type(v) == float:
                    s_j.append(2)
                elif v == 'y':
                    s_j.append(1)
                elif v == 'n':
                    s_j.append(0)

            #print('{} -- coefficient'.format(scipy.stats.pearsonr(s_i, s_j)))
            #print('{} -- matrix val'.format(correlation_mat.iloc[i,j]))
            correlation_mat.iloc[i,j] = scipy.stats.pearsonr(s_i, s_j)[0]

    correlation_mat.to_csv(os.path.join(output_dir,'correlation_matrix.csv'))

def percent_increase(initial, final):
    return (final-initial)/initial

def increase_follower_count(static_info, dynamic_info_dir, output_dir):
    grouping_cols = ['percentile', 'category_most_index-mace_label']

    for gc in grouping_cols:
        groups = static_info[gc].dropna().unique().tolist()
        avg_per_increases = []

        for g in groups:
            g_per_increases = []
            g_users = static_info.loc[static_info[gc] == g]['user_id'].values

            for u in g_users:
                u_initial = static_info[static_info['user_id'] == u]
                if os.path.isfile(os.path.join(dynamic_info_dir,'{}.csv'.format(u))):
                    u_dynamic = pd.read_csv(os.path.join(dynamic_info_dir,'{}.csv'.format(u)))

                    follow_start = u_initial['followers_count'].values[0]
                    timestamps = u_dynamic['date'].values
                    latest_date = max(timestamps)
                    follow_stop = u_dynamic[u_dynamic.loc['date'] == latest_date]['followers_count']
                    g_per_increases.append(percent_increase(follow_start,follow_stop))

            avg_per_increases.append(np.mean(g_per_increases))

        pd.DataFrame({'group': groups, 'average_increase':avg_per_increases}).to_csv(os.path.join(output_dir,'{}_percent_increase.csv'.format(gc)))


def main(in_dir, output_dir):
    static_info = pd.read_csv(os.path.join(in_dir,'static_info/static_user_info.csv'))
    dynamic_info_dir = os.path.join(in_dir,'info')
    timeline_dir = os.path.join(in_dir, 'timeline')

    tf_idf_specialization_percentile(static_info, timeline_dir, output_dir)
    follower_count_bins_by_specialization(static_info, output_dir)
    correlation_matrix(static_info, output_dir)
    increase_follower_count(static_info, dynamic_info_dir, output_dir)


if __name__ == '__main__':
    """
    Initial statistics for "shotgun" analysis of user data.
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