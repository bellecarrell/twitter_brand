import argparse
from configs.config import *
import pandas as pd
from utils.file_io import *

if __name__ == '__main__':
    """
    Generates MACE input file from HIT result file
    """

    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    parser.add_argument("mode", help="mode")
    args = parser.parse_args()

    in_file_path = args.input
    out_dir = args.output
    mode = args.mode

    df = pd.read_csv(in_file_path)

    user_ids = df['Input.user_id'].unique().tolist()
    workers = df.WorkerId.unique().tolist()
    cols_to_get = []
    if mode == "classify":
        cols_to_get.append('Answer.classify_account')
    elif mode == "binary":
        cols_to_get = [col for col in df.columns if col.startswith('Answer.category_all')]
    elif mode == "kway":
        cols_to_get.append('Answer.category_most_index')

    rows = []

    for id in user_ids:
        user_row = ['' for w in workers]
        id_assignments = df.loc[df['Input.user_id']==id]
        id_workers = id_assignments.WorkerId.unique().tolist()

        for w in id_workers:
            col_val = '/'.join([id_assignments.loc[df.WorkerId==w][col].values[0] for col in cols_to_get])
            w_index = workers.index(w)
            user_row[w_index] = col_val


        rows.append(user_row)

    in_file_name = in_file_path.split('/')[-1]
    out_name = out_dir + 'mace_input_' + mode + '_' + in_file_name
    write_rows_to_csv(out_name, rows)