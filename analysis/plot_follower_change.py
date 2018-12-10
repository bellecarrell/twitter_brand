import numpy as np
import os
import pandas as pd

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

FOLLOWER_PERCENTILES = [(9, 20), (17, 30), (30, 40), (50, 50), (84, 60),
                        (152, 70), (295, 80), (739, 90), (1480, 95),
                        (5060, 99), (193981, 100)]

IN_PATH = 'promoting_users/static_info/static_user_info.csv'

OUT_DIR = 'plots'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

OUT_PATH = os.path.join(OUT_DIR, 'follower_count_change_plots.pdf')

df = pd.read_table(IN_PATH, sep=',')

filt_df = df[['user_id', 'category_most_index-mace_label',
              'classify_account-mace_label', 'followers_count', 'percentile',
              'percent_change']]
filt_df.columns = ['user_id', 'domain', 'account_type',
                   'start_follower_count', 'start_follower_count_bin',
                   'percent_change']

filt_df = filt_df[(~np.isnan(filt_df['percent_change'])) &
                  (filt_df['account_type']=='promoting') &
                  (filt_df['start_follower_count']>0.)]

filt_df['end_follower_count'] = filt_df['start_follower_count'] * (1. + filt_df['percent_change'])

filt_df['end_follower_count_bin'] = filt_df['end_follower_count'].map(
    lambda x: ([pct for t, pct in FOLLOWER_PERCENTILES if x <= t])[0])

filt_df = filt_df.sort_values(['start_follower_count_bin', 'domain'], ascending=True)
filt_df['rank'] = list(range(filt_df.shape[0]))

filt_df['diff_follower_count'] = filt_df['end_follower_count'] - \
                                 filt_df['start_follower_count']

with PdfPages(OUT_PATH) as pdf:
    for start_bin in [[b] for b in filt_df['start_follower_count_bin'].unique()] + \
                     [filt_df['start_follower_count_bin'].unique().tolist()]:
        filt_df_subset = filt_df[filt_df['start_follower_count_bin'].isin(start_bin)]
        
        start_cnts = filt_df_subset['start_follower_count'].tolist()
        end_cnts = filt_df_subset['end_follower_count'].tolist()
        colors = filt_df_subset['domain'].tolist() + filt_df_subset['domain'].tolist()
        shapes = ['Jul 2018' for _ in range(filt_df_subset.shape[0])] + ['Nov 2018' for _ in range(filt_df_subset.shape[0])]
        idx = list(range(filt_df_subset.shape[0], 0, -1)) + list(range(filt_df_subset.shape[0], 0, -1))
        
        flat_df = pd.DataFrame({'count': start_cnts + end_cnts,
                                'domain': colors,
                                'sample time': shapes,
                                'example_rank': idx})
        flat_df['log_10(Follower Count)'] = np.log10(flat_df['count'])

        sns.set(style='whitegrid', font_scale=0.8)
        g = sns.relplot(x='log_10(Follower Count)', y='example_rank',
                        hue='domain', data=flat_df, style='sample time')
        plt.title('Bin: {}'.format(start_bin))
        
        for i, (_, row) in enumerate(filt_df_subset.iterrows()):
            plt.plot([np.log10(row['start_follower_count']),
                      np.log10(row['end_follower_count'])],
                     [filt_df_subset.shape[0] - i,
                      filt_df_subset.shape[0] - i], 'k--', alpha=0.2)
            print(filt_df_subset.shape[0] - i)
        
        plt.xticks( [np.log10(t) for t, _ in FOLLOWER_PERCENTILES],
                    ['{} ({}%)'.format(t, pct) for t, pct in FOLLOWER_PERCENTILES] )
        
        g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=30)
        g.ax.set_ylabel('')
        g.ax.set_xlabel('Follower Count (log-scale)')
        
        #plt.legend(loc='upper right')
        #import pdb; pdb.set_trace()
        #plt.setp(g.ax.get_legend().get_texts(), fontsize='10') # for legend text
        
        plt.tight_layout()
        pdf.savefig()

