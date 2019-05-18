## Sample code to plot signed p-value as a function of horizon and history windows

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set_style('whitegrid')

ivs_to_plot = {'past-PCT_FRIDAYS_WITH_TWEET': 'Fridays (%)', 'past-PCT_MSGS_9TO12_ET': '9-12 (%)', 'past-MEAN_TWEETS_PER_DAY': 'tweets/day (mean)', 'past-PCT_MSGS_REPLIES': 'replies (%)', 'past-IS_INTERACTIVE': 'is interactive', 'past-PCT_MSGS_WITH_URL': 'has url (%)', 'past-MEAN_SENTIMENT': 'sentiment (mean)', 'past-PCT_MSGS_WITH_PLURALITY_TOPIC_ADD1': 'topic dist. (% plurality topic)', 'past-TOPIC_DIST_ENTROPY_ADD01': 'topic dist. (entropy)'}

df = pd.read_table('all_model_runs.tsv.gz')
iv_df = df[df['iv'].isin(ivs_to_plot)]
iv_df['iv'] = iv_df['iv'].map(lambda x: ivs_to_plot[x])
iv_df = iv_df[iv_df['model']=='ols']
iv_df['iv_signed_pvalue'] = [float(p)-1. if float(w) < 0. else 1.-float(p) for p, w in zip(iv_df['iv_pvalue'].tolist(), iv_df['iv_wt'].tolist())]
iv_df = iv_df[iv_df['ctrl']=="['current-log_follower_count', 'current-user_impact_score', 'geo_enabled', 'primary_domain-mace_label[arts]', 'primary_domain-mace_label[travel]', 'primary_domain-mace_label[other]', 'primary_domain-mace_label[health]', 'primary_domain-mace_label[business]', 'primary_domain-mace_label[politics]', 'primary_domain-mace_label[style]', 'primary_domain-mace_label[beauty]', 'primary_domain-mace_label[books]', 'primary_domain-mace_label[gastronomy]', 'primary_domain-mace_label[sports]', 'primary_domain-mace_label[science and technology]', 'primary_domain-mace_label[family]', 'primary_domain-mace_label[games]']"]

pdf = PdfPages('/Users/abenton10/iv_pvalue_line_plots_bottom80pct.pdf')
ax = sns.lineplot(x='horizon', y='iv_signed_pvalue', hue='iv', data=iv_df)
plt.title('Signed $[1-p]$ ~ horizon'); plt.xlabel('Horizon (days)'); plt.ylabel('Signed $[1 - p]$'); plt.legend(loc=2, borderaxespad=0., fontsize=8)
plt.tight_layout()
pdf.savefig()
plt.close()
ax = sns.lineplot(x='history', y='iv_signed_pvalue', hue='iv', data=iv_df, legend=False)
plt.title('Signed $[1-p]$ ~ history'); plt.xlabel('History (days)'); plt.ylabel('Signed $[1-p]$')
plt.tight_layout()
pdf.savefig()
plt.close()
pdf.close()

old_labels = ['Fridays (%)', '9-12 (%)', 'days 1 tweet (%)', 'tweets/day (mean)', 'replies (%)', 'replies/day (mean)', 'is interactive', 'has url (%)', 'sentiment (% positive)', 'topic dist. (entropy)', 'sentiment (mean)', 'topic dist. (% plurality topic)', 'topic dist. (sqrt % plurality topic)']

kept_labels = ['Fridays (%)', 9-12 (%)', 'tweets/day (mean)', 'replies (%)', 'is interactive', 'has url (%)', 'sentiment (mean)', 'topic dist. (entropy)', 'topic dist. (% plurality topic)']
