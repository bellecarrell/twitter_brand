'''
Fit models with all controls + strategies.  Estimate test error after regularizing.
'''

import argparse
import hashlib
import numpy as np
import os
import pandas as pd
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error, r2_score
import time

import statsmodels.api as sm

HISTORY_WINDOWS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
HORIZON_WINDOWS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

IN_DIR = '/exp/abenton/twitter_brand_workspace_20190417/extracted_features_20190508/'

# what to control for
PRIMARY_DOMAINS = ['arts', 'travel', 'other', 'health', 'business', 'politics',
                   'style', 'beauty', 'books', 'gastronomy', 'sports',
                   'science and technology', 'family', 'games']

# hypotheses to test
CONTROLS_TO_EVAL = ['current-log_follower_count',
                    'current-user_impact_score',
                    'geo_enabled'] + ['primary_domain-mace_label[{}]'.format(d) for d in PRIMARY_DOMAINS]

ALL_CONTROLS = CONTROLS_TO_EVAL

ALL_STRATEGIES = ['past-PCT_FRIDAYS_WITH_TWEET', 'past-PCT_MSGS_9TO12_ET',
                  'past-PCT_DAYS_WITH_SOME_MSG', 'past-MEAN_TWEETS_PER_DAY',
                  'past-MAX_MSGS_PER_HOUR', 'past-PCT_MSGS_RT',
                  'past-PCT_MSGS_REPLIES', 'past-MEAN_MENTIONS_PER_TWEET',
                  'past-IS_INTERACTIVE', 'past-PCT_MSGS_WITH_URL',
                  'past-MEDIAN_SENTIMENT', 'past-TOPIC_DIST_ENTROPY_ADD01',
                  'past-PCT_MSGS_WITH_PLURALITY_TOPIC']

ALL_TOPICS = ['past-PROP_TOPIC_ADD01_{}'.format(i) for i in range(50)]

ALL_CONTROLS_AND_STRATEGIES = ALL_CONTROLS + ALL_STRATEGIES

ALL_CONTROLS_AND_TOPICS = ALL_CONTROLS + ALL_TOPICS

ALL_CONTROLS_STRATEGIES_AND_TOPICS = ALL_CONTROLS + ALL_STRATEGIES + ALL_TOPICS

ALL_CONTROLS_STRATEGIES_AND_DT_INTERACTIONS = ALL_CONTROLS + ALL_STRATEGIES + \
                                              ['past-INTERACTION_PROP_TOPIC_ADD01_{}_DOMAIN_{}'.format(i, d)
                                               for i in range(50) for d in PRIMARY_DOMAINS]

FEATURES_TO_EVAL = [('controls', ALL_CONTROLS),
                    ('strategies', ALL_STRATEGIES),
                    ('topics', ALL_TOPICS),
                    ('controls+strategies', ALL_CONTROLS_AND_STRATEGIES),
                    ('controls+topics', ALL_CONTROLS_AND_TOPICS),
                    ('controls+strategies+topics', ALL_CONTROLS_STRATEGIES_AND_TOPICS),
                    ('controls+strategies+topicxdomain', ALL_CONTROLS_STRATEGIES_AND_DT_INTERACTIONS)]

# outcomes to predict
REGRESSION_DEP_VARS_FMT = ['future-horizon{}-pct_change_follower_count']


def fit_regularized_model(tr_df, features, dv, alpha=0.0):
    X = tr_df[['const'] + list(features)]
    y = tr_df[dv]
    
    try:
        model = sm.OLS(y, X, missing='drop', hasconst=True)
        res = model.fit_regularized(alpha=alpha, L1_wt=0.0)
    except Exception as ex:
        print(ex)
        return None, None
    
    return res, model


def prep_dfs(tr_df, dev_df, tst_df):
    ''' Generate missing columns, remove rows with extreme follower count change. '''
    
    all_dfs = [tr_df, dev_df, tst_df]
    
    horizon_to_threshs = {}
    
    # compute thresholds for each horizon
    for h in HORIZON_WINDOWS:
        pct_change_col = 'future-horizon{}-pct_change_follower_count'.format(h)
        vals = [v for v in tr_df[pct_change_col] if not pd.isna(v)]
        
        threshs = np.nanquantile(vals, q=[0.0005, 0.33, 0.66, 0.9995])
        horizon_to_threshs[h] = threshs
        
        print('Thresholds for horizon {}: {}'.format(h, list(threshs)))
    
    # extract topic proportions
    count_key = 'past-COUNT_PER_TOPIC'
    topic_dist_key_fmt = 'past-PROP_TOPIC_ADD01_{}'
    
    delta = 0.1
    for df in all_dfs:
        topic_key_to_props = {topic_dist_key_fmt.format(t): [] for t in range(50)}
        
        for _, r in df.iterrows():
            counts = eval(r[count_key])
            denom = 50*delta + sum(counts.values())
            dist = [counts[float(t)]/denom if float(t) in counts else delta/denom for t in range(50)]
            
            for t, d in enumerate(dist):
                topic_key_to_props[topic_dist_key_fmt.format(t)].append(d)
        
        for key in topic_key_to_props:
            df[key] = topic_key_to_props[key]
    
    interaction_key_fmt = 'past-INTERACTION_PROP_TOPIC_ADD01_{}_DOMAIN_{}'
    for df in all_dfs:
        # map columns from bool to float
        df['geo_enabled'] = df['geo_enabled'].map(lambda x: 1. if x==True else (0. if x==False else None))
        df['past-IS_INTERACTIVE'] = df['past-IS_INTERACTIVE'].map(
                lambda x: 1. if x==True else (0. if x==False else None)
        )
        
        for d in PRIMARY_DOMAINS:
            df['primary_domain-mace_label[{}]'.format(d)] = df['primary_domain-mace_label'].map(
                    lambda x: 1. if x == d else (None if pd.isna(x) else 0.)
            )
            
            # extract interaction terms between domain and topic
            for t in range(50):
                df[interaction_key_fmt.format(t, d)] = df[topic_dist_key_fmt.format(t)] *\
                                                       df['primary_domain-mace_label[{}]'.format(d)]
        
        # remove extreme changes in follower count, replacing with null
        for h in HORIZON_WINDOWS:
            pct_change_col = 'future-horizon{}-pct_change_follower_count'.format(h)
    
            neg_thresh = horizon_to_threshs[h][0]  # 0.05% threshold
            pos_thresh = horizon_to_threshs[h][3]  # 99.95% threshold
    
            df[pct_change_col] = df[pct_change_col].map(lambda x: None if (pd.isna(x) or
                                                                           x < neg_thresh or
                                                                           x > pos_thresh) else x)
        
        # mean-normalize dependent variable in each fold
        for h in HORIZON_WINDOWS:
            df['future-horizon{}-pct_change_follower_count'.format(h)] = \
                df['future-horizon{}-pct_change_follower_count'.format(h)] -\
                np.nanmean(df['future-horizon{}-pct_change_follower_count'.format(h)])
    
    return tr_df, dev_df, tst_df


def main(train_path, dev_path, test_path, horizon, out_dir, args_obj):
    args_hash = hashlib.md5(str(args_obj).encode('utf8')).hexdigest()
    curr_time = time.time()
    
    if horizon not in HORIZON_WINDOWS:
        raise Exception('Did not compute follower count change for horizon of {} days!'.format(horizon))
    
    dep_vars = [dvf.format(horizon) for dvf in REGRESSION_DEP_VARS_FMT]
    
    tr_df = pd.read_table(train_path)
    tr_df['const'] = 1.
    dev_df = pd.read_table(dev_path)
    dev_df['const'] = 1.
    tst_df = pd.read_table(test_path)
    tst_df['const'] = 1.
    
    tr_df, dev_df, tst_df = prep_dfs(tr_df, dev_df, tst_df)
    
    start = time.time()
    for history in HISTORY_WINDOWS:
        for feature_set_key, feature_set in FEATURES_TO_EVAL:
            for alpha in [0.0, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.0, 10.0]:
                for dv in dep_vars:
                    run_key = '({})_({})_({})_({})'.format(history, dv, feature_set_key, alpha)
                    
                    res, model = fit_regularized_model(tr_df[tr_df['history_agg_window']==history],
                                                       feature_set, dv, alpha=alpha)
                    if res is None or model is None:
                        print('Problem fitting model for: {}'.format(run_key))
                        continue
                    
                    payload = {'args': args_obj}
                    
                    payload['params'] = res.params
                    
                    payload['history'] = history
                    payload['features'] = feature_set
                    payload['dv'] = dv
                    payload['model_class'] = 'ridge'
                    payload['alpha'] = alpha
                    payload['name_to_param'] = dict(zip(['const'] + list(feature_set), res.params))
                    
                    for fold, df in [('train', tr_df), ('dev', dev_df), ('test', tst_df)]:
                        true = df.loc[df['history_agg_window']==history, dv]
                        preds = res.predict(df.loc[df['history_agg_window']==history,
                                                   ['const'] + list(feature_set)])
                        
                        # drop nans
                        if (np.isnan(true).sum() + np.isnan(preds).sum()) > 0:
                            orig_len = true.shape[0]
                            zipped = list(zip(true, preds))
                            
                            true  = [t for t, p in zipped if not (np.isnan(t) or np.isnan(p))]
                            preds = [p for t, p in zipped if not (np.isnan(t) or np.isnan(p))]
                            
                            print('Restricted fold "{}" examples from {} -> {}'.format(fold,
                                                                                       orig_len,
                                                                                       len(true)))
                        
                        payload['{}_nobs'.format(fold)] = len(true)
                        
                        for eval_key, eval_fn in zip(['mse', 'mae', 'r2'],
                                                     [mean_squared_error, mean_absolute_error, r2_score]):
                            
                            if len(true) > 1:
                                payload[fold + '_' + eval_key] = eval_fn(true, preds)
                            else:
                                payload[fold + '_' + eval_key] = None
                    out_path = os.path.join(out_dir,
                                            'dynamic_model_full.{}.{}.{}.npz'.format(run_key,
                                                                                     int(curr_time),
                                                                                     args_hash))
                    np.savez_compressed(out_path, **payload)
                
                print('({}s) Finished training for history {}: {} ~ {} + {}'.format(int(time.time() - start),
                                                                                    history, dv,
                                                                                    feature_set_key, alpha))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='test which hypotheses lead to network growth'
    )
    parser.add_argument('--train_path', required=True,
                        default=IN_DIR + 'joined_features.with_domain.train.tsv.gz',
                        help='path to the training set')
    parser.add_argument('--dev_path', required=True,
                        default=IN_DIR + 'joined_features.with_domain.dev.dev.gz',
                        help='path to the dev set')
    parser.add_argument('--test_path', required=True,
                        default=IN_DIR + 'joined_features.with_domain.test.test.gz',
                        help='path to the test set')
    parser.add_argument('--horizon_window', required=True, type=int, default=1,
                        help='number of days in the future to predict follower count change')
    parser.add_argument('--out_dir', required=True,
                        default='/exp/abenton/twitter_brand_workspace_20190417/dynamic_full_models/',
                        help='where to save models and model performance')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    main(args.train_path, args.dev_path, args.test_path,
         args.horizon_window, args.out_dir, args.__dict__)
