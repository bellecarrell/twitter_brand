'''
Sanity checks to make sure features are generated properly.
'''

import pandas as pd
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

IN_PATH = '/exp/abenton/twitter_brand_workspace_20190417/extracted_features_20190508/joined_features.with_domain.tsv.gz'
DESC_STAT_PATH = '/exp/abenton/twitter_brand_workspace_20190417/extracted_features_20190508/feature_stats.tsv'
DIST_PATH_FMT = '/exp/abenton/twitter_brand_workspace_20190417/extracted_features_20190508/tw-{}_dist.pdf'

stat_features = ['past-NUM_MSGS', 'past-HAS_TWEET_LAST_FRIDAY', 'past-PCT_MSGS_ON_FRIDAY',
                 'past-PCT_FRIDAYS_WITH_TWEET', 'past-PCT_MSGS_9TO12_UTC',
                 'past-PCT_MSGS_9TO12_ET', 'past-PCT_MSGS_9TO12_LOCAL',
                 'past-PCT_DAYS_WITH_SOME_MSG', 'past-MEAN_TWEETS_PER_DAY',
                 'past-MSG_PER_DAY_ENTROPY_ADD1', 'past-MSG_PER_HOUR_ENTROPY_ADD01',
                 'past-MAX_MSGS_PER_HOUR', 'past-PCT_MSGS_RT', 'past-MEAN_RTS_PER_DAY',
                 'past-PCT_MSGS_REPLIES', 'past-MEAN_REPLIES_PER_DAY',
                 'past-MEAN_MENTIONS_PER_TWEET', 'past-MEAN_MSGS_WITH_MENTION',
                 'past-PCT_MSGS_WITH_URL', 'past-SHARED_URL',
                 'past-PCT_MSGS_WITH_PERSONAL_URL', 'past-SHARED_PERSONAL_URL',
                 'past-PCT_MSGS_WITH_POSITIVE_SENTIMENT', 'past-MEDIAN_SENTIMENT',
                 'past-MEAN_SENTIMENT', 'past-STD_SENTIMENT', 'past-NUM_TOPICS',
                 'past-TOPIC_DIST_ENTROPY_ADD1',
                 'past-TOPIC_DIST_ENTROPY_ADD01',
                 'past-PCT_MSGS_WITH_PLURALITY_TOPIC',
                 'past-PCT_MSGS_WITH_PLURALITY_TOPIC_ADD1',
                 'past-IS_INTERACTIVE',
                 'current-follower_count', 'current-log_follower_count',
                 'current-friend_count', 'current-log_friend_count',
                 'current-list_count', 'current-user_impact_score',
                 'future-horizon1-follower_count', 'future-horizon1-log_follower_count',
                 'future-horizon1-pct_change_follower_count',
                 'future-horizon1-user_impact_score', 'future-horizon2-follower_count',
                 'future-horizon2-log_follower_count',
                 'future-horizon2-pct_change_follower_count',
                 'future-horizon2-user_impact_score', 'future-horizon3-follower_count',
                 'future-horizon3-log_follower_count',
                 'future-horizon3-pct_change_follower_count',
                 'future-horizon3-user_impact_score', 'future-horizon4-follower_count',
                 'future-horizon4-log_follower_count',
                 'future-horizon4-pct_change_follower_count',
                 'future-horizon4-user_impact_score', 'future-horizon5-follower_count',
                 'future-horizon5-log_follower_count',
                 'future-horizon5-pct_change_follower_count',
                 'future-horizon5-user_impact_score', 'future-horizon6-follower_count',
                 'future-horizon6-log_follower_count',
                 'future-horizon6-pct_change_follower_count',
                 'future-horizon6-user_impact_score', 'future-horizon7-follower_count',
                 'future-horizon7-log_follower_count',
                 'future-horizon7-pct_change_follower_count',
                 'future-horizon7-user_impact_score', 'future-horizon14-follower_count',
                 'future-horizon14-log_follower_count',
                 'future-horizon14-pct_change_follower_count',
                 'future-horizon14-user_impact_score', 'future-horizon21-follower_count',
                 'future-horizon21-log_follower_count',
                 'future-horizon21-pct_change_follower_count',
                 'future-horizon21-user_impact_score', 'future-horizon28-follower_count',
                 'future-horizon28-log_follower_count',
                 'future-horizon28-pct_change_follower_count',
                 'future-horizon28-user_impact_score', 'past-mean_friendsPerDay',
                 'past-mean_followersPerDay'
                 ]


def compute_stats(df):
    rows = []
    for tw in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        tw_df = df[df['history_agg_window'] == tw]
        
        with PdfPages(DIST_PATH_FMT.format(tw)) as pdf:
            for f in stat_features:
                new_row = [tw, f, tw_df.shape[0], tw_df[f].isna().sum()]
                
                try:
                    new_row += tw_df[f].quantile(q=[0.0, 0.1, 0.2, 0.3, 0.4,
                                                    0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).tolist()
                    new_row += [tw_df[f].mean(), tw_df[f].std()]
                except Exception as ex:
                    import pdb; pdb.set_trace()
                
                rows.append(new_row)
                
                if tw_df[f].dtype != tw_df['sampled_datetime'].dtype:
                    try:
                        sns.distplot(tw_df[f].dropna(), kde=False)
                    except Exception as ex:
                        print('Cannot plot:', f)
                
                plt.title('Dist. of "{}" with tw {}'.format(f, tw))
                plt.xlabel('Feature: "{}"'.format(f))
                pdf.savefig()
                plt.close()
        
        print('Finished window {}'.format(tw))
    
    stat_df = pd.DataFrame(rows, columns=['history_agg_window', 'feature_name',
                                          'num_total', 'num_null', 'min'] +
                                         ['q01', 'q02', 'q03', 'q04',
                                          'q05', 'q06', 'q07', 'q08',
                                          'q09'] + ['max', 'mean', 'stdev'])
    stat_df.to_csv(DESC_STAT_PATH, header=True, index=False, sep='\t')


def run_sanity_checks(df):
    # make sure there are more messages for longer horizons
    
    #
    
    pass


def main():
    df = pd.read_table(IN_PATH)
    
    #df['past-MSG_COUNT_PER_DAY_DIST'] = df['past-MSG_COUNT_PER_DAY_DIST'].map(eval)
    #df['past-MSG_COUNT_PER_HOUR_DIST'] = df['past-MSG_COUNT_PER_HOUR_DIST'].map(eval)
    
    compute_stats(df)
    run_sanity_checks(df)


if __name__ == '__main__':
    main()
