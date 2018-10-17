import argparse
from configs.config import *
import pandas as pd
import math

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


def inter_annotator(df, worker, workers, answers):
    worker_hits = df.loc[df['WorkerId'] == worker]['HITId']
    totals = [0 for answer in answers]
    agrees = [0 for answer in answers]

    for hit in worker_hits:
        hit_rows = df.loc[df['HITId'] == hit]
        hit_workers = hit_rows.WorkerId.unique()

        for answer in answers:
            worker_response = str(hit_rows.loc[df['WorkerId'] == worker][answer].iloc[0])

            for other_worker in hit_workers:
                if worker != other_worker:
                    other_response = str(hit_rows.loc[df['WorkerId'] == other_worker][answer].iloc[0])
                    if worker_response is not 'nan':
                        totals[answers.index(answer)] += 1
                        if worker_response == other_response:
                            agrees[answers.index(answer)] += 1
                    else:
                        if other_response is 'nan':
                            totals[answers.index(answer)] += 1
                            agrees[answer.index(answer)] += 1

    return [worker] + [100 * (agree / total) if total != 0 else 'N/A' for agree, total in zip(agrees, totals) ]

def other_no_text(df, worker):
    worker_hits = df.loc[df['WorkerId'] == worker]
    total_worker_hits = len(worker_hits.index)
    total_other_no_text = len(df.loc[(worker_hits['Answer.other_text'] == '{}') & (df['Answer.category_most_index'] == 'other')].index)
    return 100 * total_other_no_text / total_worker_hits

def did_not_complete(df, worker):
    worker_hits = df.loc[df['WorkerId'] == worker]
    total_worker_hits = len(worker_hits.index)
    did_not_complete = len(df.loc[(worker_hits['Answer.classify_account'] == 'promoting') & (df['Answer.category_most_index'].isnull()==True) & (df['Answer.category_all_arts'].isnull()==True) & (df['Answer.category_all_beauty'].isnull()==True) & (df['Answer.category_all_family'].isnull()==True)& (df['Answer.category_all_gastronomy'].isnull()==True)& (df['Answer.category_all_health'].isnull()==True) & (df['Answer.category_all_other'].isnull()==True) & (df['Answer.category_all_politics'].isnull()==True) & (df['Answer.category_all_sports'].isnull()==True) & (df['Answer.category_all_style'].isnull()==True) &(df['Answer.category_all_travel'].isnull()==True)].index)
    return 100 * did_not_complete / total_worker_hits

if __name__ == '__main__':
    """
    Calculates relevant statistics for HIT results from the pre-study. 
    To be used to flag bad TURKers.
    """
    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    args = parser.parse_args()

    in_file = args.input
    out_dir = args.output

    df = pd.read_csv(in_file)
    output_dfs = []

    answer_cols_to_statify = ['classify_account', 'category_most_index', 'category_all_style', 'category_all_travel',
                              'category_all_beauty', 'category_all_gastronomy', 'category_all_politics',
                              'category_all_family', 'category_all_sports', 'category_all_other']

    # some summary stats
    ss_df = pd.DataFrame(columns=['promotional', 'category_not_other'])
    stats = []

    total_promoting_hits = len(df.loc[df['Answer.classify_account'] == 'promoting'].index)
    total_hits = len(df.index)
    other_hits = len(df.loc[df['Answer.category_most_index'] == 'other'].index)
    per_promotional = 100 * (total_promoting_hits / total_hits)
    per_not_other = 100* (total_promoting_hits - other_hits) / total_hits

    ss_df.loc[0] = [per_promotional, per_not_other]
    ss_df.index = ss_df.index + 1

    output_dfs.append(('summary_stats', ss_df))

    separate_all_df = pd.DataFrame(columns=['category', 'count', 'percentage'])
    mj_all_df = pd.DataFrame(columns=['category', 'count', 'percentage'])
    un_all_df = pd.DataFrame(columns=['category', 'count', 'percentage'])

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
        else:
            separate = add_per(separate, total_promoting_hits)
            separate_all_df.loc[0] = [col, separate['count'].iloc[0], separate['percentage'].iloc[0]]
            separate_all_df.index = separate_all_df.index + 1

        ac_gp = df.groupby(['HITId', full_col_name]).size()
        mj = ac_gp.loc[ac_gp.values >= 2].groupby(full_col_name).size().to_frame(name='count').reset_index()
        # majority vote
        if not all:
            mj = add_per(mj)
            mj_fname = 'majority_' + col
            output_dfs.append((mj_fname, mj))
        else:
            mj = add_per(mj, total_promoting_hits)
            mj_all_df.loc[0] = [col, mj['count'].iloc[0], mj['percentage'].iloc[0]]
            mj_all_df.index = mj_all_df.index + 1

        # unanimous
        un = ac_gp.loc[ac_gp.values >= 3].groupby(full_col_name).size().to_frame(name='count').reset_index()
        if not all:
            un = add_per(un)
            un_fname = 'unanimous_' + col
            output_dfs.append((un_fname, un))
        else:
            un = add_per(un, total_promoting_hits)
            if not un.empty:
                un_all_df.loc[0] = [col, un['count'].iloc[0], un['percentage'].iloc[0]]
            else:
                un_all_df.loc[0] = [col, '0', '0']

            un_all_df.index = un_all_df.index + 1

    output_dfs.append(('separate_all', separate_all_df))
    output_dfs.append(('mj_all', mj_all_df))
    output_dfs.append(('un_all', un_all_df))

    workers = df.WorkerId.unique()
    worker_hit_counts = df.groupby('WorkerId').count().HITId
    worker_df = pd.DataFrame(columns=['WorkerId', 'mean', 'median', 'num_assignments', 'non_promoting', 'other_no_text', 'did_not_complete'])
    # inter-annotator
    answers = [row for row in df.columns.values if 'Answer' in row]
    ia_cols = ['selected_worker'] + answers
    ia_df = pd.DataFrame(columns=ia_cols)

    for worker in workers:
        mean, median = mean_median(df, 'WorkTimeInSeconds', 'WorkerId', worker)
        number_assignments = worker_hit_counts.get(worker)

        classifications = df.loc[df['WorkerId'] == worker].groupby('Answer.classify_account').size().to_frame(
            name='count').reset_index()
        classifications = add_per(classifications)
        # hack -- actual indexing isn't working for some reason?
        non_promoting = 100 - classifications.tail(1)['percentage'].iloc[0]

        other_no_txt = other_no_text(df, worker)

        not_complete = did_not_complete(df, worker)

        worker_df.loc[0] = [worker, mean, median, number_assignments, non_promoting, other_no_txt, not_complete]
        worker_df.index = worker_df.index + 1

        inter_annotator_agreement = inter_annotator(df, worker, list(workers), answers)
        ia_df.loc[0] = inter_annotator_agreement
        ia_df.index = ia_df.index + 1

    output_dfs.append(('worker_stats', worker_df))
    output_dfs.append(('inter-annotator', ia_df))

    for name_df in output_dfs:
        name = name_df[0]
        df = name_df[1]
        fname = out_dir + name + '.csv'
        df.to_csv(fname)
