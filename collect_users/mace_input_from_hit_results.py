import argparse
from configs.config import *
import os
import pandas as pd
from utils.file_io import *

def labels_to_mace_fmt(in_file_path, out_dir, mode):
    df = pd.read_csv(in_file_path, dtype=object)
    
    user_ids = df['Input.user_id'].unique().tolist()
    user_ids = [uid for uid in user_ids if uid and (not pd.isna(uid))]
    workers = df.WorkerId.unique().tolist()

    cols_to_get = []
    
    if mode == "classify":
        cols_to_get.append('Answer.classify_account')
    elif mode == "binary":
        cols_to_get = [col for col in df.columns if col.startswith('Answer.category_all')]
    elif mode == "kway":
        cols_to_get.append('Answer.category_most_index')
    
    for col in cols_to_get:
        rows = []
        for id in user_ids:
            user_row = ['' for w in workers]
            id_assignments = df.loc[df['Input.user_id']==id]
            id_workers = id_assignments.WorkerId.unique().tolist()
            
            for w in id_workers:
                col_val = id_assignments.loc[df.WorkerId==w][col].values[0]
                clazz = id_assignments.loc[df.WorkerId==w]['Answer.classify_account'].values[0]
                
                if mode == 'binary':
                    if (clazz == 'promoting') and (mode == 'binary'):
                        col_val = 'y' if type(col_val)==str else 'n'
                    else:
                        col_val = ''
                else:
                    if type(col_val) != str:
                        col_val = ''
                
                #col_val = '/'.join([id_assignments.loc[df.WorkerId==w][col].values[0]
                #                    for col in cols_to_get])
                w_index = workers.index(w)
                user_row[w_index] = col_val
            
            rows.append(user_row)
        
        in_file_name = os.path.basename(in_file_path)
        out_name = os.path.join(out_dir, 'mace_input_{}-{}-{}'.format(mode,
                                                                      col,
                                                                      in_file_name))
        write_rows_to_csv(out_name, rows)
        
        # need to save these to join back to users
        out_name = os.path.join(out_dir, 'userids_{}-{}-{}'.format(mode,
                                                                   col,
                                                                   in_file_name))
        pd.DataFrame({'user_ids': user_ids}).to_csv(out_name, index=False, header=True)

        print('Wrote out mode "{}" for column "{}"'.format(mode, col))

if __name__ == '__main__':
    """
    Generates MACE input file from HIT result file
    """

    parser = argparse.ArgumentParser()
    parser = add_input_output_args(parser)
    parser.add_argument("mode", help="which label columns to prep, defaults to preparing all columns",
                        default='all', choices=['all', 'classify', 'binary', 'kway'])
    args = parser.parse_args()

    in_file_path = args.input
    out_dir = args.output
    mode = args.mode

    modes = ['classify', 'binary', 'kway'] if mode == 'all' else [mode]
    for mode in modes:
        labels_to_mace_fmt(in_file_path, out_dir, mode)
