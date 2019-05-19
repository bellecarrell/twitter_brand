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
    
    df_map = {'dv': [], 'features': [], 'alpha': [], 'weights': [],
              'horizon': [], 'history': []}
    
    for metric in ['r2', 'mae', 'mse']:
        for fold in ['train', 'dev', 'test']:
            df_map['{}_{}'.format(fold, metric)] = []
    
    for pidx, p in enumerate(ps):
        d = np.load(p, allow_pickle=True)
        
        model = d['model_class'].item()
        
        dv = d['dv'].item()
        features = [v for v in d['features']]
        
        params  = d['name_to_param'].item()
        
        df_map['dv'].append(dv)
        df_map['features'].append(features[0] if len(features) == 1 else features)
        df_map['history'].append(d['history'].item())
        df_map['horizon'].append(int(dv.split('-')[1].replace('horizon', '')))
        
        df_map['weights'].append([params[v] for v in features])
        df_map['alpha'].append(d['alpha'].item())
        
        for metric in ['r2', 'mae', 'mse']:
            for fold in ['train', 'dev', 'test']:
                k = '{}_{}'.format(fold, metric)
                if k in d:
                    df_map[k].append(d[k].item())
                else:
                    df_map[k].append(None)
        
        if not (pidx % 200):
            print('{}/{} paths read'.format(pidx, len(ps)))
    
    df = pd.DataFrame(df_map)
    df['feature_str'] = df['features'].map(str)
    df.drop_duplicates(subset=['dv', 'feature_str',
                               'horizon', 'history', 'alpha'], inplace=True)
    df.to_csv(os.path.join(TABLE_DIR, 'all_model_runs.tsv.gz'),
              sep='\t', header=True,
              index=False, compression='gzip')


if __name__ == '__main__':
    if not os.path.exists(TABLE_DIR):
        os.mkdir(TABLE_DIR)
    
    main()
