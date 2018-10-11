import argparse
from configs.config import *
import pandas as pd

def mean_median(df, col, where_col, where_val):
    if where_col:
        df = df.loc[df[where_col] == where_val]

    vals = df[col]

    return vals.mean(), vals.median()

def add_per(df_with_count, total=None):

    if total:
        per = 100 * (df_with_count['count'] / total)
    else:
        per = 100 * (df_with_count['count'] / df_with_count['count'].sum())

    df_with_count['percentage'] = per
    return df_with_count

if __name__ == '__main__':
    """
    Calculates relevant statistics for HIT results from the pre-study. 
    To be used to flag bad TURKers.
    """
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    args = parser.parse_args()
    total_promoting_hits = 189

    in_file = args.input
    out_dir = args.output

    df = pd.read_csv(in_file)
    output_dfs = []

    answer_cols_to_statify = ['classify_account', 'category_most_index', 'category_all_style', 'category_all_travel', 'category_all_beauty', 'category_all_gastronomy', 'category_all_politics', 'category_all_family', 'category_all_sports', 'category_all_other']
    all_df = pd.DataFrame(columns=['category', 'count', 'percentage'])

    for col in answer_cols_to_statify:
        all = False
        if 'all' in col:
            all = True

        # separate annotations
        full_col_name = 'Answer.' + col

        separate = df.groupby(full_col_name).HITId.count().to_frame(name='count').reset_index()
        if not all:
            separate = add_per(separate)
            separate_fname = 'separate_' + col
            output_dfs.append((separate_fname, separate))
        #else:
        #    separate = add_per(separate, total_promoting_hits)
        #    separate_fname = 'separate_' + col
        #    output_dfs.append((separate_fname, separate))

        ac_gp = df.groupby(['HITId', full_col_name]).size()
        # majority vote
        if not all:
            mj = ac_gp.loc[ac_gp.values >= 2].groupby(full_col_name).size().to_frame(name='count').reset_index()
            mj = add_per(mj)
            mj_fname = 'majority_' + col
            output_dfs.append((mj_fname, mj))

        # unanimous
        if not all:
            un = ac_gp.loc[ac_gp.values >=3].groupby(full_col_name).size().to_frame(name='count').reset_index()
            un = add_per(un)

            un_fname = 'unanimous_' + col
            output_dfs.append((un_fname, un))

    workers = df.WorkerId.unique()
    worker_hit_counts = df.groupby('WorkerId').count().HITId
    worker_df = pd.DataFrame(columns=['WorkerId', 'mean', 'median', 'non_promoting'])
    i = -1

    for worker in workers:
        mean, median = mean_median(df, 'WorkTimeInSeconds', 'WorkerId', worker)
        number_assignments = worker_hit_counts.get(worker)

        classifications = df.loc[df['WorkerId'] == worker].groupby('Answer.classify_account').size().to_frame(name='count').reset_index()
        classifications = add_per(classifications)
        p = classifications.loc[df['Answer.classify_account'] == 'promoting']['percentage']
        non_promoting = 100 - classifications.loc[df['Answer.classify_account'] == 'promoting']['percentage']

        worker_df.loc[0] = [worker, mean, median, 'tbd']
        worker_df.index = worker_df.index + 1

    output_dfs.append(('worker_stats', worker_df))

    for name_df in output_dfs:
        name = name_df[0]
        df = name_df[1]
        fname = out_dir + name + '.csv'
        df.to_csv(fname)

