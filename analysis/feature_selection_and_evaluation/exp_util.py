from analysis.file_data_util import *
import sklearn.linear_model
import numpy as np
from analysis.bootstrap_test import bootstrap_test
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

def filter_X_by_selected_features(X, us, sf, vocab):
    n_sf = len(sf)
    filt_X = X[:,sf]
    return filt_X

def train_lr_with_salient_features(in_dir,out_dir, vocab,static_info,sf, y_method, majority_label, tw, low=True,future=False, sp=False):

    if sp:
        sf = [s[1] for s in sf['other']]
    else:
        sf = [s[1] for s in sf['low']]
    sf = [vocab[s] for s in sf]
    ret_eval = {}

    max_l2 = None
    max_acc = 0

    for i, l2 in enumerate([1, 10, 50, 100, 500, 1000]):
        lr = sklearn.linear_model.LogisticRegression(C=l2)

        X, us = load_fold(in_dir, vocab, static_info, 'train', future=future)
        X = X[:, sf]
        if sp:
            y_l = y_method(us, static_info)
        else:
            X, y_l, y_h = y_method(X, us, static_info, len(sf))

        if low:
            lr.fit(X,y_l)
        else:
            lr.fit(X,y_h)

        X, us = load_fold(in_dir, vocab, static_info, 'dev',future=future)
        X = X[:, sf]
        if sp:
            y_l = y_method(us, static_info)
        else:
            X, y_l, y_h = y_method(X, us, static_info, len(sf))

        yhat = np.argmax(lr.predict_proba(X), axis=1)
        mj_yhat = [majority_label for y in yhat]

        if low:
            y = y_l
        else:
            y = y_h

        e = {}
        e['model'] = 'linear_{}'.format(l2)
        e['model_dir'] = 'linear_sf'
        e['f1'] = f1_score(y, yhat, average='macro')
        acc = accuracy_score(y, yhat)
        e['accuracy'] = acc
        if acc > max_acc:
            max_acc = acc
            max_l2 = l2
        e['confusion_matrix'] = confusion_matrix(y, yhat)
        e['vs_majority'] = bootstrap_test(mj_yhat, yhat, y, 50, 1000, accuracy_score)[0]
        ret_eval[i] = e

    X, us = load_fold(in_dir, vocab, static_info, 'train', future=future)
    X = X[:, sf]
    if sp:
        y_l = y_method(us, static_info)
    else:
        X, y_l, y_h = y_method(X, us, static_info, len(sf))
    lr = sklearn.linear_model.LogisticRegression(C=max_l2)
    lr.fit(X,y_l)

    X, us = load_fold(in_dir, vocab, static_info, 'test', future=future)
    X = X[:, sf]
    if sp:
        y_l = y_method(us, static_info)
    else:
        X, y_l, y_h = y_method(X, us, static_info, len(sf))

    yhat = np.argmax(lr.predict_proba(X), axis=1)
    mj_yhat = [majority_label for y in yhat]

    y = y_l

    e = {}
    e['model'] = 'linear_{}_test'.format(max_l2)
    e['model_dir'] = 'linear_sf'
    e['f1'] = f1_score(y, yhat, average='macro')
    acc = accuracy_score(y, yhat)
    e['accuracy'] = acc
    e['confusion_matrix'] = confusion_matrix(y, yhat)
    e['vs_majority'] = bootstrap_test(mj_yhat, yhat, y, 50, 1000, accuracy_score)[0]

    ret_eval[6] = e

    out_path = os.path.join(out_dir,'{}_eval_linear_sf'.format(tw))

    df = pd.DataFrame(ret_eval).T
    df.to_csv(out_path, sep='\t', header=True, index=False)
