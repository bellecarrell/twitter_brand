import sys
sys.path.append('/home/hltcoe/acarrell/PycharmProjects/twitter_brand/')
import logging

logging.basicConfig(level=logging.INFO)
import nltk

nltk.download('stopwords')
from analysis.batching import *

BATCH_START_WINDOW = datetime.datetime(2018,4,1)
BATCH_END_WINDOW = datetime.datetime(2018,7,31)
SEED = 12345
VERBOSE = True

from analysis.file_data_util import *

def dated_tweets_by_user(f, users):
    """
    Get tweets from the timeline compressed file and place in dict sorted by user.
    Filtered to include only tweets within window
    :param in_dir: top-level directory containing promoting user dfs
    :return: dict of tweets by user ID
    """
    reader = csv.reader(f)
    tweets = defaultdict(list)
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
            tweets[user_id].append((created_at, text))
            if created_at > dates[user_id]:
                dates[user_id] = created_at
            if created_at > most_recent:
                most_recent = created_at
        count += 1
        if count % 100000 == 0:
            print('Read {} rows'.format(count))

    print('Filtering users with {} most recent'.format(most_recent))
    recent_users_removed = 0
    all_users = copy.deepcopy(list(tweets.keys()))
    for user in all_users:
        u_date = dates[user]
        if not posted_recently(most_recent, u_date):
            del tweets[user]
            del dates[user]
            recent_users_removed += 1
    print('{} users removed'.format(recent_users_removed))

    return tweets.keys(), tweets

def filter_by_tw_and_specialization(static_info, dates_tweets, tw):

    start, stop = tw
    tweets = defaultdict(list)
    for user in dates_tweets.keys():
        specialization = static_info.loc[static_info['user_id'] == user]['category_most_index-mace_label'].values[0]
        for created_at, text in dates_tweets[user]:
            if start <= datetime.datetime.fromtimestamp(created_at) <= stop:
                if len(text) > 0 and type(specialization) is not float:
                    tweets[user].append(text)
    return tweets.keys(), tweets

def _generate_y_binary(us,s, static_info):
    y = np.array([0 for u in us])
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    s_to_y_val = dict((s, i) for i, s in enumerate(specializations))
    s_i = s_to_y_val[s]
    for i, u in enumerate(us):
        specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]
        u_i = s_to_y_val[specialization]
        if u_i == s_i:
            y[i] = 1

    return y

def _generate_y_multi(us, static_info):
    y = np.array([0 for u in us])
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    s_to_y_val = dict((s, i) for i, s in enumerate(specializations))
    for i, u in enumerate(us):
        specialization = static_info.loc[static_info['user_id'] == u]['category_most_index-mace_label'].values[0]
        y[i] = s_to_y_val[specialization]

    return y


def fit_batches(rr, Xs_users, n_batches, static_info, l1_range=[0.0, 1.0]):
    np.random.seed(SEED)

    if rr.log_l1_range:
        l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
    else:
        l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

    start = time.time()

    # fit each batch
    for b, (X_us, l1) in enumerate(zip(Xs_users, l1s)):
        X, us = X_us
        y = _generate_y_multi(us, static_info)

        print('{} size of y'.format(y.shape))
        rr.fit_batch(X, y, l1, b)
        if VERBOSE:
            print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))


def main(in_dir,out_dir):
    static_info = pd.read_csv(os.path.join(in_dir,'static_info/static_user_info.csv'))
    specializations = static_info['category_most_index-mace_label'].dropna().unique().tolist()
    y_to_s_val = dict((i, s) for i, s in enumerate(specializations))
    users = static_info.loc[static_info['classify_account-mace_label'] == 'promoting'][
        'user_id'].dropna().unique().tolist()


    #todo: switch back to vocab once
    vocab = from_gz(in_dir,'vocab')
    n_batches = 100
    dates_tweets = from_gz(os.path.join(in_dir,'json'),'dates_tweets')
    dates_tweets = clean_dates_tweets(dates_tweets)

    sw = set(stopwords.words('english'))
    sw.union({c for c in list(string.punctuation) if c is not "#" and c is not "@"})


    vectorizer = CountVectorizer(vocabulary=vocab,stop_words=sw)

    Xs_users = generate_batches(static_info, dates_tweets, vectorizer,n_batches=n_batches)


    log_reg = RR(is_continuous=False, model_dir='./log_reg_models', log_l1_range=True)
    fit_batches(log_reg, Xs_users, n_batches, static_info, l1_range=[0.0, 10.0])
    salient_features = log_reg.get_salient_features(dict((v,k) for k,v in vocab.items()), y_to_s_val,n=1300)

    with open(os.path.join(out_dir,'rlr_selected_features_multi.txt'),'w+') as f:
        for s, feat in salient_features.items():
            f.write('Specialization: {} Salient features: {}\n\n'.format(s, feat))
    
if __name__ == '__main__':
    """
    Run randomized logistic regression to see features selected for different specializations.
    May add different class of Ys in the future for user impact (so, change in follower count or
    a related metric)
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