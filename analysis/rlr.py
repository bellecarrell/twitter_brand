'''
Implementation of Randomized Linear/Logistic Regression.

Adrian Benton
11/26/2018
'''
import gzip
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix, accuracy_score
import time
from analysis.bootstrap_test import bootstrap_test

SEED = 12345
VERBOSE = False


class RandomizedRegression:
    def __init__(self, model_type='lr', model_dir=None, log_l1_range=False, min_absolute_threshold=0.0):
        self.model_type = model_type  # fit linear regression instead of logistic
        self.model_dir = model_dir  # where to save model weights
        self.log_l1_range = log_l1_range  # pick l1 logarithmically from random
        
        if self.model_type == 'lasso':
            self._build_model = lambda l1: sklearn.linear_model.Lasso(alpha=l1,
                                                                      selection='random',
                                                                      random_state=np.random.randint(0, 123456))
        elif self.model_type == 'lr':
            self._build_model = lambda l1: sklearn.linear_model.LogisticRegression(penalty='l1', C=l1)
        else:
            self._build_model = lambda l1: sklearn.svm.SVC(C=l1, kernel='linear', probability=True)
        
        self.min_absolute_threshold = min_absolute_threshold
        
        self.neg_feature_counts = {}
        self.pos_feature_counts = {}
        self.abs_feature_counts = {}
    
    def get_salient_features(self, feature_key, target_key, n=100, salience_type='pos'):
        if not self.abs_feature_counts:
            print('Need to fit model first')
            return None
        else:
            if salience_type == 'pos':
                feature_counts = self.pos_feature_counts
            elif salience_type == 'neg':
                feature_counts = self.neg_feature_counts
            elif salience_type == 'abs':
                feature_counts = self.abs_feature_counts
            else:
                raise Exception('Do not recognize count type: "{}"'.format(salience_type))

            print('{} feature counts'.format(feature_counts.keys()))
            print('{} target key'.format(target_key.keys()))
            rev_cnts = [[(v, feature_key[k]) for k, v in feature_counts[i].items()]
                        for i, target in target_key.items() if i in feature_counts.keys()]
            for cnts in rev_cnts:
                cnts.sort(reverse=True)
            
            salient_features_per_target = {target: rev_cnts[i][:n] for i, target in target_key.items()
                                           if i in feature_counts.keys()}
            
            return salient_features_per_target
    
    def fit(self, X, y, n_batches=100, prop_per_batch=0.1, l1_range=[0.0, 10.0]):
        np.random.seed(SEED)
        
        if self.log_l1_range:
            l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
        else:
            l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]
        
        start = time.time()
        
        # subsample examples to create new batches
        for b, l1 in enumerate(l1s):
            filt = np.random.random(X.shape[0]) <= prop_per_batch
            self.fit_batch(X[filt, :], y[filt], l1, b)
            
            if VERBOSE:
                print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))

    def fit_batches(self, Xs, ys, l1_range=[0.0, 1.0]):
        np.random.seed(SEED)
        
        n_batches = len(Xs)

        if self.log_l1_range:
            l1s = np.power(l1_range[1] - l1_range[0], np.random.random(n_batches)) - 1 + l1_range[0]
        else:
            l1s = (l1_range[1] - l1_range[0]) * np.random.random(n_batches) + l1_range[0]

        start = time.time()
        
        # fit each batch
        for b, (X, y, l1) in enumerate(zip(Xs, ys, l1s)):
            self.fit_batch(X, y, l1, b)
            if VERBOSE:
                print('Finished {}/{} ({}s)'.format(b, len(l1s), int(time.time() - start)))

    def fit_batch(self, X, y, l1, batch_idx):
        if all([v == y[0] for v in y]):
            print('Skipping batch, all labels = {}'.format(y[0]))
            return
        
        model = self._build_model(l1)
        
        model.fit(X, y)
        
        if 'sparse_coef_' in dir(model):
            wts = model.sparse_coef_
        else:
            if self.model_type != 'svm':
                model = model.sparsify()
            wts   = model.coef_
        
        if self.model_dir is not None:
            #if not os.path.exists(self.model_dir):
            #    os.mkdir(self.model_dir)

            print('self model dir {}'.format(self.model_dir))
            out_path = os.path.join(self.model_dir, 'model_{}_{}.pickle.gz'.format(batch_idx, int(time.time())))
            
            with gzip.open(out_path, 'w') as model_file:
                pickle.dump(model, model_file)
        
        # count which features are positive, negative, or have large absolute value
        if len(wts.shape) == 1:
            if 0 not in self.abs_feature_counts:
                self.pos_feature_counts[0] = {}
                self.neg_feature_counts[0] = {}
                self.abs_feature_counts[0] = {}
            
            nz_idxes = wts.nonzero()[0]
            
            for i in nz_idxes:
                v = wts[i]
                
                if i not in self.abs_feature_counts[0]:
                    self.abs_feature_counts[0][i] = 0
                    self.pos_feature_counts[0][i] = 0
                    self.neg_feature_counts[0][i] = 0

                if np.abs(v) > self.min_absolute_threshold:
                    self.abs_feature_counts[0][i] += 1
                if v > self.min_absolute_threshold:
                    self.pos_feature_counts[0][i] += 1
                if v < -self.min_absolute_threshold:
                    self.neg_feature_counts[0][i] += 1
        else:
            for target in range(wts.shape[0]):
                if target not in self.abs_feature_counts:
                    self.pos_feature_counts[target] = {}
                    self.neg_feature_counts[target] = {}
                    self.abs_feature_counts[target] = {}

                z = wts[target].nonzero()
                if self.model_type != 'svm':
                    nz_idxes = wts[target].nonzero()[1]
                else:
                    nz_idxes = wts[target].nonzero()[0]
                
                for i in nz_idxes:
                    v = wts[target, i]

                    if i not in self.abs_feature_counts[target]:
                        self.abs_feature_counts[target][i] = 0
                        self.pos_feature_counts[target][i] = 0
                        self.neg_feature_counts[target][i] = 0
    
                    if np.abs(v) > self.min_absolute_threshold:
                        self.abs_feature_counts[target][i] += 1

                    if v > self.min_absolute_threshold:
                        self.pos_feature_counts[target][i] += 1
                    if v < -self.min_absolute_threshold:
                        self.neg_feature_counts[target][i] += 1


class Ensemble:
    def __init__(self, model_dir):
        self.load_models(model_dir)
    
    def load_models(self, model_dir):
        self._model_dir = model_dir
        
        ps = sorted([os.path.join(self._model_dir, p) for p in os.listdir(model_dir)])
        self._models = []
        
        for p in ps:
            with gzip.open(p, 'r') as f:
                try:
                    self._models.append(pickle.load(f))
                except Exception as ex:
                   print('Error loading "{}" -- {}'.format(p, ex))
        
        self.model_weighting = np.ones((len(self._models),))

        if type(self._models[0]) == sklearn.linear_model.LogisticRegression:
            self.model_type = 'lr'
        elif type(self._models[0]) == sklearn.linear_model.Lasso:
            self.model_type = 'lasso'
        else:
            self.model_type = 'svm'

        if self.model_type == 'lasso':
            self.is_continuous = True
        else:
            self.is_continuous = False

    def predict(self, X, int_to_label=None, return_all_model_preds=False):
        if self.model_type == 'lasso':
            if return_all_model_preds:
                return [m.predict(X) for m in self._models]
            else:
                all_preds = [wt * m.predict(X) for wt, m in zip(self.model_weighting, self._models)]
                preds = sum(all_preds) / sum(self.model_weighting)
        # elif self.model_type == 'svm':
        #     if return_all_model_preds:
        #         preds = [m.predict(X) for m in self._models]
        #         preds = [[0 if p < 0.5 else 1 for p in ps] for ps in preds]
        #     else:
        #         all_preds = [wt * m.predict(X) for wt, m in zip(self.model_weighting, self._models)]
        #         preds = sum(all_preds) / sum(self.model_weighting)
        #         preds = [0 if p < 0.5 else 1 for p in preds]
        else:
            if return_all_model_preds:
                #return [np.argmax(m.predict_log_proba(X), axis=1) for m in self._models]
                return [np.argmax(m.predict_proba(X), axis=1) for m in self._models]
            else:
                #all_scores = [wt * m.predict_log_proba(X) for wt, m in zip(self.model_weighting, self._models)]
                all_scores = [wt * m.predict_proba(X) for wt, m in zip(self.model_weighting, self._models)]
                mean_scores = sum(all_scores) / sum(self.model_weighting)
                preds = np.argmax(mean_scores, axis=1)
            
            if int_to_label is not None:
                preds = [int_to_label[p] for p in preds]
        
        return preds

    def score(self, X, return_all_model_scores=False):
        if self.model_type == 'lasso':  # same as prediction
            if return_all_model_scores:
                return [m.predict(X) for m in self._models]
            else:
                all_preds = [wt * m.predict(X) for wt, m in zip(self.model_weighting, self._models)]
                scores = sum(all_preds) / sum(self.model_weighting)
        elif self.model_type == 'svm':  # same as prediction
            if return_all_model_scores:
                return [m.predict(X) for m in self._models]
            else:
                all_preds = [wt * m.predict(X) for wt, m in zip(self.model_weighting, self._models)]
                scores = sum(all_preds) / sum(self.model_weighting)
        else:
            if return_all_model_scores:
                return [m.predict_log_proba(X) for m in self._models]
            else:
                all_scores = [wt * m.predict_log_proba(X) for wt, m in zip(self.model_weighting, self._models)]
                scores = sum(all_scores) / sum(self.model_weighting)
    
        return scores

    def eval(self, X, y, majority_label, int_to_label=None,replace=False):
        yhats = [self.predict(X, int_to_label, False)]
        yhats += self.predict(X, int_to_label, True)

        mj_yhat = [majority_label for y in yhats[0]]

        ret_eval = {}

        #majority
        if not self.is_continuous:
            e = {}
            e['model'] = 'majority'
            e['model_dir'] = 'mj'
            e['f1'] = f1_score(y, mj_yhat, average='macro')
            e['accuracy'] = accuracy_score(y, mj_yhat)
            e['confusion_matrix'] = confusion_matrix(y, mj_yhat)
            e['vs_majority'] = 0
            ret_eval[0] = e

        for idx, yhat in enumerate(yhats):
            e = {}
            
            if idx == 0:
                l1 = None
            else:
                l1 = self._models[idx-1].alpha if self.is_continuous else self._models[idx-1].C
            
            e['model'] = 'ensemble' if idx == 0 else 'model-{}-l1_{:.2f}'.format(idx, l1)
            e['model_dir'] = self._model_dir
            
            if self.model_type == 'lasso':
                e['mae'] = mean_absolute_error(y, yhat)
                e['mse'] = mean_squared_error(y, yhat)

            else:
                e['f1'] = f1_score(y, yhat, average='macro')
                e['accuracy'] = accuracy_score(y, yhat)
                e['confusion_matrix'] = confusion_matrix(y, yhat)
            ret_eval[idx+1] = e

            e['acc_p'] = bootstrap_test(mj_yhat,yhat,y,50,1000,accuracy_score,with_replacement=replace)[0]
            e['f1_p'] = bootstrap_test(mj_yhat, yhat, y, 50, 1000, f1_score, with_replacement=replace,f1=True)[0]

        return ret_eval
    
    def eval_to_file(self, X, y, majority_label, out_path, int_to_label=None,replace=False):
        ret_eval = self.eval(X, y, majority_label, int_to_label,replace=replace)
        
        df = pd.DataFrame(ret_eval).T
        df.to_csv(out_path, sep='\t', header=True, index=False)


def gen_data(n=200, p=200, noise=0.01):
    # Figure 4 in Meinshausen 2009 paper.  p predictors, all independent, but
    # correlated with predictor 3.  y is generated as a sum of the first
    # two predictors with added noise.
    
    rho = 0.5  # covariance between predictors 1 & 2 and 3
    cov = np.eye(p, p) + 0.001
    cov[0, 2] = rho
    cov[1, 2] = rho
    
    beta = np.zeros(p)
    beta[0] = 1.0
    beta[1] = 1.0
    X = np.random.multivariate_normal(np.zeros(p), cov, size=(n,))
    
    y = X.dot(beta) + np.random.normal(0.0, noise, size=(n,))
    
    return X, y
    

def test_continuous():
    print('==== Test continuous randomized regression ====')
    X, y = gen_data(n=500, p=200, noise=0.01)
    
    lin_reg = RandomizedRegression(is_continuous=True, model_dir=None, log_l1_range=True)
    lin_reg.fit(X, y, n_batches=100, prop_per_batch=0.05, l1_range=[0.0, 10.0])
    
    salient_features = lin_reg.get_salient_features({i: 'X{}'.format(i) for i in range(200)}, {0: 'target'}, n=5)
    
    # should be first 2 predictors more often than not
    print('+ Salient features (should be X0 & X1): {}'.format(salient_features))


def test_discrete():
    print('==== Test discrete randomized regression ====')
    X, y = gen_data(n=500, p=200, noise=0.01)
    y = np.array([1 if v > 0. else 0 for v in y])
    
    log_reg = RandomizedRegression(is_continuous=False, model_dir='./log_reg_models_discrete', log_l1_range=True)
    log_reg.fit(X, y, n_batches=100, prop_per_batch=0.05, l1_range=[0.0, 10.0])

    salient_features = log_reg.get_salient_features({i: 'X{}'.format(i) for i in range(200)}, {0: 'target'}, n=5)

    # should be first 2 predictors more often than not
    print('+ Salient features (should be X0 & X1): {}'.format(salient_features))
    
    discrete_ensemble = Ensemble('./log_reg_models_discrete')
    
    df = pd.DataFrame(discrete_ensemble.eval(X, y)).T
    
    acc = df['accuracy']
    ensemble_acc = acc[0]
    
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    individual_quants = [np.quantile(acc[1:], q=q) for q in quantiles]
    
    print('Model accuracy by quantile: {}'.format(['(q={:.2f}, {:.2f})'.format(q, acc)
                                                   for q, acc
                                                   in zip(quantiles, individual_quants)]))
    print('Ensemble accuracy: {:.3f}'.format(ensemble_acc))


def main():
    pass


if __name__ == '__main__':
    test_discrete()
    test_continuous()
