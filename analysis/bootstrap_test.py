import numpy as np
import pandas as pd
import random
from typing import Callable, Iterable

def bootstrap_test(y_hat1: Iterable[float],
                   y_hat2: Iterable[float],
                   y: Iterable[float],
                   batch_size: int,
                   n: int,
                   eval_func: Callable[[Iterable[float], Iterable[float]], float],
                   seed=12345,
                   with_replacement=False,
                   gt_is_better: bool=True, f1=False) -> (float, (Iterable[float], Iterable[float])):
    '''
    Compare performance of two models according to a bootstrap sampling test.
    Does model 2 outperform model 1 across different subsets of the data?
    
    :param y_hat1: predictions for model 1
    :param y_hat2: predictions for model 2
    :param y: gold targets
    :param batch_size: size of batches to subsample
    :param n: number of samples to take
    :param eval_func: maps from y and y_hat to a score
    :param with_replacement: sample with replacement
    :param seed: random seed to sample examples
    :param gt_is_better: higher scores -> better performance
    :return p, scores: p-value that model 2 is better than model 1
                       performance of each model for each sample
    '''
    
    perf_tally = []
    
    scores = ([], [])
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Collect targets and predictions in a single table.  Easier to subsample over rows.
    df = pd.DataFrame({'model1': y_hat1, 'model2': y_hat2, 'gold': y})
    
    for i in range(n):
        subsample_df = df.sample(n=batch_size, replace=with_replacement)

        if f1:
            perf1 = eval_func(subsample_df['gold'], subsample_df['model1'], average='macro')
            perf2 = eval_func(subsample_df['gold'], subsample_df['model2'],average='macro')
        else:
            perf1 = eval_func(subsample_df['gold'], subsample_df['model1'])
            perf2 = eval_func(subsample_df['gold'], subsample_df['model2'])
        
        scores[0].append(perf1)
        scores[1].append(perf2)
        
        if gt_is_better:
            perf_tally.append(1 * (perf1 < perf2))
        else:
            perf_tally.append(1 * (perf2 < perf1))
    
    p = 1. - np.nanmean(perf_tally)
    
    return p, scores
