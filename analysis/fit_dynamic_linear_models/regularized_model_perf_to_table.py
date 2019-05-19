'''
Collect dynamic model performance into tables by horizon and history (e.g. Table 3).

Adrian Benton
5/13/2019
'''

import os
import pandas as pd
import numpy as np

MODEL_RUN_DIR = '/exp/abenton/twitter_brand_workspace_20190417/dynamic_full_models/'
TABLE_DIR = '/exp/abenton/twitter_brand_workspace_20190417/dynamic_full_tables/'

PRIMARY_DOMAINS = ['arts', 'travel', 'other', 'health', 'business', 'politics',
                   'style', 'beauty', 'books', 'gastronomy', 'sports',
                   'science and technology', 'family', 'games']


def main():
    ps = [os.path.join(MODEL_RUN_DIR, p) for p in os.listdir(MODEL_RUN_DIR) if p.endswith('.npz')]
    
    df_map = {'dv': [], 'alpha': [], 'weights': [],
              'horizon': [], 'history': []}
    
    for metric in ['r2', 'mae', 'mse']:
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
            print('{}/{} paths read'.format(pidx, len(ps)))
    
    df = pd.DataFrame(df_map)
    df['iv_str'] = df['iv'].map(str)
    df['ctrl_str'] = df['ctrl'].map(str)
    df.drop_duplicates(subset=['dv', 'iv_str',
                               'ctrl_str', 'model',
                               'horizon', 'history'], inplace=True)
    df.to_csv(os.path.join(TABLE_DIR, 'all_model_runs.tsv.gz'),
              sep='\t', header=True,
              index=False, compression='gzip')


if __name__ == '__main__':
    if not os.path.exists(TABLE_DIR):
        os.mkdir(TABLE_DIR)
    
    main()
