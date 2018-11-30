'''
Implementation of Randomized Linear/Logistic Regression.

Adrian Benton
11/26/2018
'''

import gzip
import numpy as np
import os
import pickle
import sklearn.linear_model
import time

SEED = 12345
VERBOSE = False

class RandomizedRegression:
    def __init__(self, is_continuous=False, model_dir=None, log_l1_range=False, min_absolute_threshold=0.0):
        self.is_continuous = is_continuous  # fit linear regression instead of logistic
        self.model_dir = model_dir  # where to save model weights
        self.log_l1_range = log_l1_range  # pick l1 logarithmically from random
        
        if self.is_continuous:
            self._build_model = lambda l1: sklearn.linear_model.Lasso(alpha=l1,
                                                                      selection='random',
                                                                      random_state=np.random.randint(0, 123456))
        else:
            self._build_model = lambda l1: sklearn.linear_model.LogisticRegression(penalty='l1', C=l1)
        
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

            print('{} feature counts'.format(feature_counts))
            print('{} target key'.format(target_key))
            rev_cnts = [[(v, feature_key[k]) for k, v in feature_counts[i].items()]
                        for i, target in target_key.items()]
            for cnts in rev_cnts:
                cnts.sort(reverse=True)
            
            salient_features_per_target = {target: rev_cnts[i][:n] for i, target in target_key.items()}
            
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
            model = model.sparsify()
            wts   = model.coef_
        
        if self.model_dir is not None:
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
            
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

                nz_idxes = wts[target].nonzero()[1]
                
                for i in nz_idxes:
                    v = wts[target, i]
                    
                    if i not in self.abs_feature_counts[0]:
                        self.abs_feature_counts[target][i] = 0
                        self.pos_feature_counts[target][i] = 0
                        self.neg_feature_counts[target][i] = 0
    
                    if np.abs(v) > self.min_absolute_threshold:
                        self.abs_feature_counts[target][i] += 1
                    if v > self.min_absolute_threshold:
                        self.pos_feature_counts[target][i] += 1
                    if v < -self.min_absolute_threshold:
                        self.neg_feature_counts[target][i] += 1


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
    
    log_reg = RandomizedRegression(is_continuous=False, model_dir='./log_reg_models', log_l1_range=True)
    log_reg.fit(X, y, n_batches=100, prop_per_batch=0.05, l1_range=[0.0, 10.0])

    salient_features = log_reg.get_salient_features({i: 'X{}'.format(i) for i in range(200)}, {0: 'target'}, n=5)

    # should be first 2 predictors more often than not
    print('+ Salient features (should be X0 & X1): {}'.format(salient_features))


def main():
    pass


if __name__ == '__main__':
    test_discrete()
    test_continuous()