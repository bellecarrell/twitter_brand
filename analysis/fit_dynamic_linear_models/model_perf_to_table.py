'''
Collect dynamic model performance into tables by horizon and history (e.g. Table 3).

Adrian Benton
5/13/2019
'''

import os
import pandas as pd
import numpy as np

MODEL_RUN_DIR = '/exp/abenton/twitter_brand_workspace_20190417/dynamic_models_baseline_strategies/'
TABLE_DIR = '/exp/abenton/twitter_brand_workspace_20190417/dynamic_models_baseline_tables/'


def main():
    ps = [os.path.join(MODEL_RUN_DIR, p) for p in os.listdir(MODEL_RUN_DIR) if p.endswith('.npz')]
    
    df_map = {'dv': [], 'iv': [], 'ctrl': [],
              'model': [], 'iv_wt': [],
              'iv_pvalue': [], 'iv_tvalue': [],
              'ctrl_wt': [], 'ctrl_pvalue': [],
              'ctrl_tvalue': [], 'bic': [],
              'horizon': [], 'history': []}
    
    for metric in ['r2', 'mae', 'mse', 'accuracy', 'f1', 'precision', 'recall']:
        for fold in ['train', 'dev', 'test']:
            df_map['{}_{}'.format(fold, metric)] = []
    
    for pidx, p in enumerate(ps):
        d = np.load(p)
        
        model = d['model_class'].item().encode('ascii')
        
        dv = d['dv'].item().encode('ascii')
        ivs = [v.encode('ascii') for v in d['ivs']]
        ctrls = [v.encode('ascii') for v in d['controls']]
        
        params  = d['name_to_param'].item()
        pvalues = d['name_to_pvalue'].item()
        tvalues = d['name_to_tvalue'].item()
        
        df_map['dv'].append(dv)
        df_map['iv'].append(ivs[0] if len(ivs) == 1 else ivs)
        df_map['ctrl'].append(ctrls[0] if len(ctrls) == 1 else ctrls)
        df_map['model'].append(model)
        df_map['bic'].append(d['bic'].item())
        df_map['history'].append(d['history'].item())
        df_map['horizon'].append(int(dv.split('-')[1].replace('horizon', '')))
        
        if len(ivs) == 1:
            df_map['iv_wt'].append(params[ivs[0]])
            df_map['iv_pvalue'].append(pvalues[ivs[0]])
            df_map['iv_tvalue'].append(tvalues[ivs[0]])
        else:
            df_map['iv_wt'].append([params[v] for v in ivs])
            df_map['iv_pvalue'].append([pvalues[v] for v in ivs])
            df_map['iv_tvalue'].append([tvalues[v] for v in ivs])
        
        if len(ctrls) == 1:
            df_map['ctrl_wt'].append(params[ctrls[0]])
            df_map['ctrl_pvalue'].append(pvalues[ctrls[0]])
            df_map['ctrl_tvalue'].append(tvalues[ctrls[0]])
        else:
            df_map['ctrl_wt'].append([params[v] for v in ctrls])
            df_map['ctrl_pvalue'].append([pvalues[v] for v in ctrls])
            df_map['ctrl_tvalue'].append([tvalues[v] for v in ctrls])
        
        for metric in ['r2', 'mae', 'mse', 'accuracy', 'f1', 'precision', 'recall']:
            for fold in ['train', 'dev', 'test']:
                k = '{}_{}'.format(fold, metric)
                if k in d:
                    df_map[k].append(d[k].item())
                else:
                    df_map[k].append(None)
        
        if not (pidx % 200):
            print('{}/{} paths read; {} rows'.format(pidx, len(ps), len(df_map['model'])))
    
    df = pd.DataFrame(df_map)
    df.drop_duplicates(subset=['dv', 'iv',
                               'ctrl', 'model',
                               'horizon', 'history'], inplace=True)
    df.to_csv(os.path.join(TABLE_DIR, 'all_model_runs.tsv.gz'),
              sep='\t', header=True,
              index=False, compression='gzip')
    
    # collect rows by model class, horizon, and history and build "table 3" for each
    metric_to_report = {'logreg': 'f1', 'ols': 'r2', 'qr': 'r2'}
    
    rows = []
    
    for m in df['model'].unique():
        metric = metric_to_report[m]
        for horizon in df['horizon'].unique():
            for history in df['history'].unique():
                sub_df = df[(df['model'] == m) &
                            (df['horizon'] == horizon) &
                            (df['history'] == history)]
                
                
                fixed_perf_tbl = pd.DataFrame(rows)
                
                fixed_perf_tbl.to_csv(os.path.join(TABLE_DIR,
                                                   'table_model-{}_horizon-{}_history-{}'.format(
                                                           m, horizon, history)
                                                   ), sep='\t', header=True, index=False,
                                      )
                
                print('Saved table for {} {} {}'.format(m, horizon, history))


if __name__ == '__main__':
    if not os.path.exists(TABLE_DIR):
        os.mkdir(TABLE_DIR)
    
    main()
