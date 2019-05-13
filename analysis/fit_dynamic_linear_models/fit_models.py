'''
Fit models to evaluate if hypotheses can predict change in follower count,
after controlling for various confounds.
'''

import argparse
import hashlib
import numpy as np
import os
import pandas as pd
from sklearn.metrics.classification import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error, r2_score
import time

import statsmodels.api as sm
import statsmodels.formula.api as smf

HISTORY_WINDOWS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
HORIZON_WINDOWS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

IN_DIR = '/exp/abenton/twitter_brand_workspace_20190417/extracted_features_20190508/'

#### TODO: need to update variable columns once table is generated ####

# what to control for
CONTROLS = [('none', ()),
            ('logf', ('current-log_follower_count',)),
            ('logf+imp', ('current-log_follower_count',
                          'current-user_impact_score')),
            ('logf+imp+geo', ('current-log_follower_count',
                              'current-user_impact_score',
                              'geo_enabled')),
            ('logf+imp+geo+dom', ('current-log_follower_count',
                                  'current-user_impact_score',
                                  'geo_enabled',
                                  'primary_domain-mace_label'))]

# hypotheses to test
CONTROLS_TO_EVAL = ['current-log_follower_count',
                    'current-user_impact_score',
                    'geo_enabled']

TIMING_VARS = ['past-PCT_MSGS_ON_FRIDAY',
               'past-PCT_FRIDAYS_WITH_TWEET',
               'past-HAS_TWEET_LAST_FRIDAY',
               'past-PCT_MSGS_9TO12_UTC',
               'past-PCT_MSGS_9TO12_ET']
ENGAGEMENT_VARS = ['past-PCT_DAYS_WITH_SOME_MSG',
                   'past-MEAN_TWEETS_PER_DAY',
                   'past-MSG_PER_DAY_ENTROPY_ADD1',
                   'past-MSG_PER_HOUR_ENTROPY_ADD01',
                   'past-MAX_MSGS_PER_HOUR',
                   'past-PCT_MSGS_RT',
                   'past-MEAN_RTS_PER_DAY',
                   'past-PCT_MSGS_REPLIES',
                   'past-MEAN_REPLIES_PER_DAY',
                   'past-MEAN_MENTIONS_PER_TWEET',
                   'past-MEAN_MSGS_WITH_MENTION',
                   'past-IS_INTERACTIVE']
CONTENT_VARS = ['past-PCT_MSGS_WITH_URL',
                'past-SHARED_URL',
                'past-PCT_MSGS_WITH_POSITIVE_SENTIMENT',
                'past-MEDIAN_SENTIMENT',
                'past-MEAN_SENTIMENT',
                'past-STD_SENTIMENT',
                'past-TOPIC_DIST_ENTROPY_ADD1',
                'past-TOPIC_DIST_ENTROPY_ADD01',
                'past-PCT_MSGS_WITH_PLURALITY_TOPIC',
                'past-PCT_MSGS_WITH_PLURALITY_TOPIC_ADD1']

FEATS_TO_SQRT = ['past-TOPIC_DIST_ENTROPY_ADD01',
                 'past-TOPIC_DIST_ENTROPY_ADD1',
                 'past-PCT_MSGS_WITH_PLURALITY_TOPIC_ADD1']
FEATS_TO_LOG  = ['past-PCT_MSGS_WITH_PLURALITY_TOPIC_ADD1']
SQRT_VARS = ['sqrt[{}]'.format(f) for f in FEATS_TO_SQRT]
LOG_VARS = ['log1p[{}]'.format(f) for f in FEATS_TO_LOG]

INDEXED_HYPOTHESES = [(v, (v,)) for v in CONTROLS_TO_EVAL +
                                         TIMING_VARS +
                                         ENGAGEMENT_VARS +
                                         CONTENT_VARS +
                                         SQRT_VARS +
                                         LOG_VARS]
INDEXED_HYPOTHESES += SQRT_VARS
INDEXED_HYPOTHESES += LOG_VARS

ALL_VAR = [('all', tuple(TIMING_VARS +
                         ENGAGEMENT_VARS +
                         CONTENT_VARS +
                         SQRT_VARS +
                         LOG_VARS))]

# all hypotheses
IND_VARS = INDEXED_HYPOTHESES + ALL_VAR

# outcomes to predict
REGRESSION_DEP_VARS_FMT = ['future-horizon{}-pct_change_follower_count']
CLASSIFICATION_DEP_VARS_FMT = ['future-horizon{}-direction_follower_count_change']


def fit_model(tr_df, controls, ivs, dv, model_class):
    import pdb; pdb.set_trace()
    
    formula = '{} ~ {}'.format(dv, ' + '.join(['const'] + controls + ivs))
    
    if model_class == 'logreg':
        model = smf.logit(formula, tr_df)
        res = model.fit()
    elif model_class == 'ols':
        model = smf.ols(formula, tr_df)
        res = model.fit()
    elif model_class == 'qr':
        model = smf.quantreg(formula, tr_df)
        res = model.fit(q=0.5)
    else:
        raise Exception('Do not recognize model "{}"'.format(model_class))
    
    return res, model


def prep_dfs(tr_df, dev_df, tst_df):
    ''' Generate missing columns, remove rows with extreme follower count change. '''
    
    all_dfs = [tr_df, dev_df, tst_df]
    
    horizon_to_threshs = {}

    # compute thresholds for each horizon
    for h in HORIZON_WINDOWS:
        pct_change_col = 'future-horizon{}-pct_change_follower_count'.format(h)
        vals = tr_df[pct_change_col]
        
        threshs = np.quantile(vals, q=[0.0005, 0.33, 0.66, 0.9995])
        horizon_to_threshs[h] = threshs
        
        print('Thresholds for horizon {}: {}'.format(h, list(threshs)))

    def to_classification_task(v, neg_thresh, pos_thresh):
        if v < neg_thresh:
            return 0
        elif v > pos_thresh:
            return 1
        else:
            return None
    
    for df in all_dfs:
        # add sqrt/log-transformed features
        for f in FEATS_TO_SQRT:
            df['sqrt[{}]'.format(f)] = df[f].map(np.sqrt)
        
        for f in FEATS_TO_LOG:
            df['log1p[{}]'.format(f)] = df[f].map(lambda x: np.log(1. + x))
        
        # remove extreme changes in follower count, replacing with null
        for h in HORIZON_WINDOWS:
            pct_change_col = 'future-horizon{}-pct_change_follower_count'.format(h)
    
            neg_thresh = horizon_to_threshs[0]  # 0.05% threshold
            pos_thresh = horizon_to_threshs[3]  # 99.95% threshold
    
            df[pct_change_col] = df[pct_change_col].map(lambda x: None if (x < neg_thresh or
                                                                           x > pos_thresh) else x)
        
        # add column for binary classification, discriminate between top and bottom third of examples
        for h in HORIZON_WINDOWS:
            neg_thresh = horizon_to_threshs[1]  # 33% threshold
            pos_thresh = horizon_to_threshs[2]  # 66% threshold
            
            df['future-horizon{}-direction_follower_count_change'.format(h)] = \
                df['future-horizon{}-pct_change_follower_count'.format(h)].map(
                        lambda x: to_classification_task(x, neg_thresh, pos_thresh)
                )
    
    return tr_df, dev_df, tst_df


def main(train_path, dev_path, test_path, horizon, model_class, out_dir, args_obj):
    args_hash = hashlib.md5(str(args_obj).encode('utf8')).hexdigest()
    curr_time = time.time()
    
    if horizon not in HORIZON_WINDOWS:
        raise Exception('Did not compute follower count change for horizon of {} days!'.format(horizon))
    
    if model_class in ['logreg']:
        dep_vars = [dvf.format(horizon) for dvf in CLASSIFICATION_DEP_VARS_FMT]
    else:
        dep_vars = [dvf.format(horizon) for dvf in REGRESSION_DEP_VARS_FMT]
    
    tr_df = pd.read_table(train_path, sep='\t')
    tr_df = sm.tools.tools.add_constant(tr_df)
    dev_df = pd.read_table(dev_path, sep='\t')
    dev_df = sm.tools.tools.add_constant(dev_df)
    tst_df = pd.read_table(test_path, sep='\t')
    tst_df = sm.tools.tools.add_constant(tst_df)
    
    tr_df, dev_df, tst_df = prep_dfs(tr_df, dev_df, tst_df)
    
    # TODO: need to step through this by hand to figure out how to estimate error on heldout set
    start = time.time()
    for ctrl_set_key, ctrls in CONTROLS:
        for hyp_key, ivs in IND_VARS:
            for dv in dep_vars:
                
                import pdb; pdb.set_trace()
                res, model = fit_model(tr_df, ctrls, ivs, dv, model_class)
                run_key = '{}_{}_{}'.format(dv, ctrl_set_key, hyp_key)
                
                payload = {'model': model, 'res': res, 'args': args_obj}
                
                for fold, df in [('dev', dev_df), ('test', tst_df)]:
                    true = df[dv]
                    preds = res.predict(df)
                    if model_class in ['logreg']:
                        for eval_key, eval_fn in zip(['accuracy', 'f1', 'precision', 'recall'],
                                                     [accuracy_score, f1_score, precision_score, recall_score]):
                            payload[fold + '_' + eval_key] = eval_fn(true, preds)
                    else:
                        for eval_key, eval_fn in zip(['mse', 'mae', 'r2'],
                                                     [mean_squared_error, mean_absolute_error, r2_score]):
                            payload[fold + '_' + eval_key] = eval_fn(true, preds)

                out_path = os.path.join(out_dir,
                                        'dynamic_model.{}.{}.{}.npz'.format(run_key,
                                                                            curr_time,
                                                                            args_hash))
                np.savez_compressed(out_path, **payload)
                
                print('({}s) Finished training: {} ~ {} + {}'.format(int(time.time() - start),
                                                                     dv, ctrl_set_key, hyp_key))


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
    parser.add_argument('--history_agg_window', required=True, type=int, default=1,
                        help='number of days in the past to aggregate features by')
    parser.add_argument('--horizon_window', required=True, type=int, default=1,
                        help='number of days in the future to predict follower count change')
    parser.add_argument('--model_class', required=True,
                        choices=['ols', 'qr', 'logreg'],
                        help='class of linear models to fit: ordinary least squares, ' +
                             'median regression (minimize l1 loss), or logistic regression ' +
                             'predicting whether followers will be gained or lost')
    parser.add_argument('--out_dir', required=True,
                        default='/exp/abenton/twitter_brand_workspace_20190417/dynamic_models_baseline_strategies/',
                        help='where to save models and model performance')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    main(args.train_path, args.test_path, args.horizon_window, args.model_class, args.out_dir, args.__dict__)
