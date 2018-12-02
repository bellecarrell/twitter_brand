import argparse
from analysis.file_data_util import *
from analysis.batching import *
from utils.file_io import *
import numpy as np
import gzip


def main(in_dir, out_dir, n_batches=100):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    
    rev_vocab_key = from_gz(os.path.join(in_dir, 'features'), 'vocab')
    feature_df = pd.read_table(os.path.join(in_dir,
                                            'features',
                                            'user_features_per_tweet.noduplicates.tsv.gz'))
    
    batches_dir = os.path.join(out_dir, 'batches')
    if not os.path.isdir(batches_dir):
        os.mkdir(batches_dir)
    
    for tw_name, wsize, tr in [#('one_week', 7, (BATCH_START_WINDOW, BATCH_END_WINDOW)),
                               #('two_week', 14, (BATCH_START_WINDOW, BATCH_END_WINDOW)),
                               #('month', 30, (BATCH_START_WINDOW, BATCH_END_WINDOW)),
                               ('all_data', (BATCH_END_WINDOW - BATCH_START_WINDOW).days,
                                  (BATCH_START_WINDOW, BATCH_END_WINDOW)),
                               ('one_week_future', 7, (BATCH_END_WINDOW, BATCH_END_FOLLOWER_TS)),
                               ('two_week_future', 14, (BATCH_END_WINDOW, BATCH_END_FOLLOWER_TS)),
                               ('month_future', 30, (BATCH_END_WINDOW, BATCH_END_FOLLOWER_TS)),
                               ('all_data_future', (BATCH_END_FOLLOWER_TS - BATCH_END_WINDOW).days,
                                  (BATCH_END_WINDOW, BATCH_END_FOLLOWER_TS)),
                               ]:
        batch_generator = generate_batches_precomputed_features(static_info,
                                                                feature_df,
                                                                rev_vocab_key,
                                                                n_batches=n_batches,
                                                                window_size=wsize,
                                                                ret_tw=True,
                                                                full_time_range=tr)
        
        subdir = os.path.join(batches_dir, '{}_batches'.format(tw_name))
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
        
        for b, (X, user_ids, tw) in enumerate(batch_generator):
            rs, cs = X.nonzero()
            np.savez_compressed(os.path.join(subdir, 'batch_{}.npz'.format(b)),
                                **{'row': rs,
                                   'col': cs,
                                   'value': np.array(X[(rs, cs)])[0],
                                   'user_id': user_ids,
                                   'time_window': tw})
            
            print('Wrote batch {} for {}'.format(b, tw_name))


def main_unprocessed(in_dir, out_dir):
    static_info = pd.read_csv(os.path.join(in_dir, 'static_info/static_user_info.csv'))
    #todo: switch back to current vocab once fixed
    vocab = from_gz(in_dir,'vocab')
    n_batches = 100
    dates_tweets = load_dates_tweets(os.path.join(in_dir,'json'),'dates_tweets')
    sw = set(stopwords.words('english'))
    sw.union({c for c in list(string.punctuation) if c is not "#" and c is not "@"})
    vectorizer = CountVectorizer(vocabulary=vocab,stop_words=sw)

    batches_by_tw = [('one_week', generate_batches(static_info, dates_tweets, vectorizer,n_batches=n_batches,window_size=7,ret_tw=True)),
    (('two weeks'), generate_batches(static_info, dates_tweets, vectorizer,n_batches=n_batches,window_size=14,ret_tw=True)),
    (('month'), generate_batches(static_info, dates_tweets, vectorizer, n_batches=n_batches, window_size=30,ret_tw=True))]

    batches_dir = os.path.join(out_dir,'batches')
    if not os.path.isdir(batches_dir):
        os.mkdir(batches_dir)

    for tw_name, Xs_us_tw in batches_by_tw:
        subdir = os.path.join(batches_dir,'{}_batches'.format(tw_name))
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
        for b, (X, us, tw) in enumerate(Xs_us_tw):
            tw = [tw]
            X_f = gzip.GzipFile(os.path.join(subdir, 'batch_{}_X.np.gz'.format(b)),"w")
            us_f = os.path.join(subdir, 'batch_{}_us.gz'.format(b))
            tw_f = os.path.join(subdir, 'batch_{}_tw.gz'.format(b))
            np.save(file=X_f,arr=X)
            write_list_to_file(us_f,us,mode='w+')
            write_list_to_file(tw_f, tw, mode='w+')


if __name__ == '__main__':
    """
    """

    parser = argparse.ArgumentParser(
        description='write batches to file'
    )
    parser.add_argument('--input_dir', required=True,
                        dest='input_dir', metavar='INPUT_DIR',
                        help='directory with user information, should have info/, static_info/, and timeline/ subdirs')
    parser.add_argument('--out_dir', required=True,
                        dest='out_dir', metavar='OUTPUT_DIR',
                        help='output directory')
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.out_dir

    main(in_dir, out_dir)
