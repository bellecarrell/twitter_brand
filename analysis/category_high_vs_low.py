import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
from analysis.batching import *
from analysis.file_data_util import *

logging.basicConfig(level=logging.INFO)
import time
import datetime
import numpy as np
import json
SEED = 12345
VERBOSE = True

def _X_us_for_spec(static_info, s, X, us):
    us = list(us)
    filt_us = []
    for u in us:
        specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]
        if s == specialization:
            filt_us.append(u)

    filt_X = np.zeros(shape=(len(filt_us),20000))
    for i, u in enumerate(filt_us):
        orig_i = us.index(u)
        filt_X[i] = X[orig_i]

    return X, us

def _generate_y(us, static_info, cutoff=70):
    y = np.array([0 for u in us])
    for i, u in enumerate(us):
        percentile = int(static_info.loc[static_info['user_id'] == u]['percentile'].values[0])
        if percentile >= cutoff:
            y[i] = 1
    return y

def fit_batches_all_spec(Xs_users, n_batches, static_info, cutoff=70, l1_range=[0.0, 1.0]):
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    np.random.seed(SEED)

    rr = RR(is_continuous=False, model_dir='./log_reg_models', log_l1_range=True)

    if rr.log_l1_range:
        l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
    else:
        l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

    start = time.time()

    rrs_for_spec = [(s, RR(is_continuous=False, model_dir='./log_reg_models', log_l1_range=True)) for s in
                    specializations]

    # fit each batch
    for b, (X_us, l1) in enumerate(zip(Xs_users, l1s)):
        X, us = X_us

        for s_rrs in rrs_for_spec:
            s, rr = s_rrs
            X, us = _X_us_for_spec(static_info,s,X,us)
            y = _generate_y(us, static_info)
            rr.fit_batch(X, y, l1, b)
        if VERBOSE:
            print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))

    return rrs_for_spec

def main(in_dir, out_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))

    #todo: switch back to vocab once
    vocab = from_gz(in_dir,'vocab')
    n_batches = 100
    dates_tweets = from_gz(os.path.join(in_dir,'json'),'dates_tweets')
    dates_tweets = clean_dates_tweets(dates_tweets)

    sw = set(stopwords.words('english'))
    sw.union({c for c in list(string.punctuation) if c is not "#" and c is not "@"})


    vectorizer = CountVectorizer(vocabulary=vocab,stop_words=sw)

    Xs_users = generate_batches(static_info, dates_tweets, vectorizer,n_batches=n_batches)


    with open(os.path.join(out_dir, 'rlr_selected_features_low_high.txt'), 'w+') as f:
        spec_fitted = fit_batches_all_spec(Xs_users, n_batches, static_info, l1_range=[0.0, 10.0])

        for s in spec_fitted:
            name, rr = s
            salient_features = rr.get_salient_features(dict((v, k) for k, v in vocab.items()), {0: 'low',1:'high'})
            f.write('Specialization: {} Range: {} Salient features: {}\n'.format(s, 'high', salient_features))
            f.write('\n')

if __name__ == '__main__':
    """
    Run randomized logistic regression to see features selected for high and low follower count, stratified by category.
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