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

#### TODO: need to update variable columns once table is generated ####

# what to control for
CONTROLS = [('none', ()),
            ('logf', ('curr_log_follower_count',)),
            ('logf+imp', ('curr_log_follower_count',
                          'curr_user_impact_score')),
            ('logf+imp+geo', ('curr_log_follower_count',
                              'curr_user_impact_score',
                              'curr_user_geo_enabled')),
            ('logf+imp+geo+dom', ('curr_log_follower_count',
                                  'curr_user_impact_score',
                                  'curr_user_geo_enabled',
                                  'user_domain'))]

# hypotheses to test
CONTROLS_TO_EVAL = ['curr_log_follower_count',
                    'curr_user_impact_score',
                    'curr_user_geo_enabled']
TIMING_VARS = ['pct_on_Fridays',
               'pct_covered_Fridays',
               'binary_covered_last_Friday',
               'pct_9to12',
               'binary_hastweet_9to12']
ENGAGEMENT_VARS = ['mean_tweets_per_day',
                   'entropy_tweets_daily',
                   'entropy_tweets_hourly',
                   'pct_retweets',
                   'pct_replies',
                   'pct_mentions',
                   'binary_interactivity']
CONTENT_VARS = ['pct_url',
                'binary_posted_url',
                'pct_personal_url',
                'binary_posted_personal_url',
                'pct_positive_sentiment',
                'mean_sentiment',
                'entropy_topics',
                'pct_plurality_topic']
INDEXED_HYPOTHESES = [(v, (v,)) for v in CONTROLS_TO_EVAL +
                                         TIMING_VARS +
                                         ENGAGEMENT_VARS +
                                         CONTENT_VARS]
ALL_VAR = [('all', tuple(TIMING_VARS + ENGAGEMENT_VARS + CONTENT_VARS))]

# all hypotheses
IND_VARS = INDEXED_HYPOTHESES + ALL_VAR

# outcomes to predict
REGRESSION_DEP_VARS_FMT = ['raw_follower_change_{}-horizon',
                           'signed_log_follower_change_{}-horizon',
                           'pct_follower_change_{}-horizon']
CLASSIFICATION_DEP_VARS_FMT = ['direction_follower_change_{}-horizon']


def fit_model(tr_df, controls, ivs, dv, model_class):
    formula = '{} ~ {}'.format(dv, ' + '.join(['const'] + controls + ivs))
    
    if model_class == 'logreg':
        model = smf.logit(formula, tr_df)
        res = model.fit()
    elif model_class == 'ols':
        model = smf.quantreg(formula, tr_df)
        res = model.fit(q=0.5)
    elif model_class == 'qr':
        model = smf.quantreg(formula, tr_df)
        res = model.fit()
    else:
        raise Exception('Do not recognize model "{}"'.format(model_class))
    
    return res, model


def main(train_path, dev_path, test_path, horizon, model_class, out_dir, args_obj):
    args_hash = hashlib.md5(str(args_obj).encode('utf8')).hexdigest()
    curr_time = time.time()
    
    if horizon not in HORIZON_WINDOWS:
        raise Exception('Did not compute follower count change for horizon of {} days!'.format(horizon))
    
    if model_class in ['logreg']:
        dep_vars = [dvf.format(horizon) for dvf in CLASSIFICATION_DEP_VARS_FMT]
    else:
        dep_vars = [dvf.format(horizon) for dvf in REGRESSION_DEP_VARS_FMT]
    
    tr_df = pd.read_table(train_path, sep=',')
    tr_df = sm.tools.tools.add_constant(tr_df)
    dev_df = pd.read_table(dev_path, sep=',')
    dev_df = sm.tools.tools.add_constant(dev_df)
    tst_df = pd.read_table(test_path, sep=',')
    tst_df = sm.tools.tools.add_constant(tst_df)
    
    # TODO: need to step through this by hand to figure out how to estimate error on heldout set
    start = time.time()
    for ctrl_set_key, ctrls in CONTROLS:
        for hyp_key, ivs in IND_VARS:
            for dv in dep_vars:
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
                        default='/.../dynamic_feature_table_{}-history.train.csv.gz',
                        help='path to the training set')
    parser.add_argument('--dev_path', required=True,
                        default='/.../dynamic_feature_table_{}-history.train.csv.gz',
                        help='path to the training set')
    parser.add_argument('--test_path', required=True,
                        default='/.../dynamic_feature_table_{}-history.test.csv.gz',
                        help='path to the test set')
    parser.add_argument('--horizon_window', required=True, type=int,
                        help='number of days in the future to predict follower count change')
    parser.add_argument('--model_class', required=True,
                        choices=['ols', 'qr', 'logreg'],
                        help='class of linear models to fit: ordinary least squares, ' +
                             'median regression (minimize l1 loss), or logistic regression ' +
                             'predicting whether followers will be gained or lost')
    parser.add_argument('--out_dir', required=True,
                        default='/exp/abenton/twitter_brand_workspace04252019/dynamic_models_baseline_strategies/',
                        help='where to save models and model performance')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    main(args.train_path, args.test_path, args.horizon_window, args.model_class, args.out_dir, args.__dict__)
