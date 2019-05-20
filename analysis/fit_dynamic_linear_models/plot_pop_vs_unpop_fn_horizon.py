## Sample code to plot signed p-value as a function of horizon and history windows

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set_style('whitegrid')

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 5.0})

ivs_to_plot = {'past-PCT_FRIDAYS_WITH_TWEET': 'Fridays (%)', 'past-PCT_MSGS_9TO12_ET': 'Posted 9-12 (%)', 'past-MEAN_TWEETS_PER_DAY': 'Tweets/day (mean)', 'past-PCT_MSGS_REPLIES': 'Replies (%)', 'past-IS_INTERACTIVE': 'Is interactive', 'past-PCT_MSGS_WITH_URL': 'Has URL (%)', 'past-MEDIAN_SENTIMENT': 'Sentiment (median)', 'past-TOPIC_DIST_ENTROPY_ADD1': 'Topic dist. $\delta=1$ (entropy)', 'past-PCT_MSGS_WITH_PLURALITY_TOPIC_ADD1': 'Topic dist. $\delta=1$ (% plurality topic)'}

bottom_df = pd.read_table('/Users/abenton10/additional_projects/twitter_brand/dynamic_models_baseline_tables_bottom80pct/all_model_runs.tsv.gz')
top_df = pd.read_table('/Users/abenton10/additional_projects/twitter_brand/dynamic_models_baseline_tables_top10pct/all_model_runs.tsv.gz')

kept_labels = ['Fridays (%)', 'Posted 9-12 (%)', 'Tweets/day (mean)',
               'Replies (%)', 'Is interactive', 'Has URL (%)',
               'Sentiment (median)', 'Topic dist. $\delta=1$ (entropy)',
               'Topic dist. $\delta=1$ (% plurality topic)']

bottom_df['Popularity Percentile'] = '$\leq$ 80%'
top_df['Popularity Percentile'] = '$\geq$ 90%'
all_df = pd.concat([bottom_df, top_df])
all_df['Strategy'] = all_df['iv'].map(lambda x: ivs_to_plot[x] if x in ivs_to_plot else None)
all_df = all_df[all_df['Strategy'].isin(set(kept_labels))]


def pvalue_to_thresh(p):
    p = float(p)
    if p >= 0.05:
        return 1.0
    elif p >= 0.01:
        return 0.05
    elif p >= 0.001:
        return 0.01
    else:
        return 0.001

all_df['p < 0.01'] = all_df['iv_pvalue'].map(pvalue_to_thresh)

all_df = all_df[all_df['ctrl'] == "['current-log_follower_count', 'current-user_impact_score', 'geo_enabled', 'primary_domain-mace_label[arts]', 'primary_domain-mace_label[travel]', 'primary_domain-mace_label[other]', 'primary_domain-mace_label[health]', 'primary_domain-mace_label[business]', 'primary_domain-mace_label[politics]', 'primary_domain-mace_label[style]', 'primary_domain-mace_label[beauty]', 'primary_domain-mace_label[books]', 'primary_domain-mace_label[gastronomy]', 'primary_domain-mace_label[sports]', 'primary_domain-mace_label[science and technology]', 'primary_domain-mace_label[family]', 'primary_domain-mace_label[games]']"]

all_df['iv_wt'] = all_df['iv_wt'].map(lambda x: float(x))

with PdfPages('/Users/abenton10/Desktop/pop_vs_unpop_weights_fn_horizon.pdf') as pdf:
    horizon_df = all_df[all_df['history']==14]
    
    for i, l in enumerate(kept_labels):
        subset_df = horizon_df[horizon_df['Strategy']==l]
        
        ax = sns.lineplot(x='horizon', y='iv_wt', hue='Popularity Percentile',
                          data=subset_df,
                          dashes=False, legend='brief' if (i==2) else False,
                          #legend = 'brief' if (i == 0) else False,
                          alpha=0.8)
        
        sns.scatterplot(x='horizon', y='iv_wt', hue='Popularity Percentile',
                        style='p < 0.01', markers={1.0: '.', 0.05: '^', 0.01: 'p', 0.001: '*'},
                        data=subset_df, ax=ax, legend=False, s=1000., alpha=0.8)
        
        ax.set_title('{}'.format(l), fontsize=25)

        #if i == 2:
        #    plt.legend()
        
        if i == 3 or i == 2:
            ax.set_ylabel('Strategy Weight', fontsize=20)
        else:
            plt.ylabel('')
        
        if i == 7:
            ax.set_xlabel('Horizon', fontsize=20)
        else:
            plt.xlabel('')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()

with PdfPages('/Users/abenton10/Desktop/pop_vs_unpop_weights_fn_history.pdf') as pdf:
    history_df = all_df[all_df['horizon']==14]
    
    for i, l in enumerate(kept_labels):
        subset_df = history_df[history_df['Strategy']==l]
        
        ax = sns.lineplot(x='history', y='iv_wt', hue='Popularity Percentile',
                          data=subset_df, dashes=False,
                          legend='brief' if (i==2) else False,
                          # legend = 'brief' if (i == 0) else False,
                          alpha=0.8)
        
        sns.scatterplot(x='history', y='iv_wt', hue='Popularity Percentile',
                        style='p < 0.01', markers={1.0: '.', 0.05: '^', 0.01: 'p', 0.001: '*'},
                        data=subset_df, ax=ax, legend=False, s=1000., alpha=0.8)
        
        ax.set_title('{}'.format(l), fontsize=25)

        #if i == 0:
        #    plt.legend()
        
        if i == 3 or i == 2:
            ax.set_ylabel('Strategy Weight', fontsize=20)
        else:
            plt.ylabel('')
        
        if i == 7:
            ax.set_xlabel('History', fontsize=20)
        else:
            plt.xlabel('')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
