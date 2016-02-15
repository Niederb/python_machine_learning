# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:22:33 2015

@author: thoma
"""
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from random import sample
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from scipy.stats import mode
from sklearn.preprocessing import Imputer

digits = load_digits()
print(digits.data.shape)
test_percentage = 0.5

f_t, f_v, c_t, c_v = train_test_split(digits.data, digits.target, test_size=test_percentage)

nan_prob_t = 0.0
nan_mask_t = np.random.binomial(n=1, p=nan_prob_t, size=np.shape(f_t))
nan_f_t = f_t
nan_f_t[nan_mask_t==1] = np.nan
nan_prob_v = 0.0
nan_mask_v = np.random.binomial(n=1, p=nan_prob_v, size=np.shape(f_v))
nan_f_v = f_v
nan_f_v[nan_mask_v == 1] = np.nan

class HalfRF:
    def __init__(self, data, classes, tree_features, n_trees=100):
        self.n_features = np.shape(data)[1]
        n_rows = np.shape(data)[0]
        n_nans = np.sum(np.isnan(data), 0)
        data = data[:, n_nans < n_rows]
        self.n_features = np.shape(data)[1]
        
        n_nans = np.sum(np.isnan(data), 1)
        data = data[n_nans < self.n_features, :]
        self.n_rows = np.shape(data)[0]
        
        if (tree_features > self.n_features):
            tree_features = self.n_features
        
        self.col_list = np.zeros((n_trees, tree_features), dtype='int')
        self.n_trees = n_trees
        self.bags = []
        for i in range(n_trees):
            cols = sample(range(self.n_features), tree_features)
            cols.sort()
            self.col_list[i, :] = cols
            data_temp = data[:, cols]
            n_nans = np.sum(np.isnan(data_temp), 1)
            data_temp = data_temp[n_nans == 0, :]
            classes_temp = classes[n_nans == 0]
            #bag = BaggingClassifier(n_estimators=1, max_features=tree_features)
            bag = RandomForestClassifier(n_estimators=1, max_features=tree_features)
            bag.fit(data_temp, classes_temp)
            self.bags.append(bag)
            print(np.shape(data_temp))
        
    def classify(self, data):
        nan_cols = np.arange(self.n_features)[np.isnan(data)]
        decisions = []
        s1 = set(nan_cols)
        for i in range(self.n_trees):
            cols = self.col_list[i]
            s2 = set(cols)
            
            if len(s1.intersection(s2)) > 0:
                #decisions[i] = -1
                continue
            decisions.append(self.bags[i].predict(data[cols]))
        if (len(decisions) == 0):
            return (-1, 0, 0)
        return (mode(decisions)[0][0][0], mode(decisions)[1][0][0], len(decisions))
          
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(f_t)
imp_f_t = imp.transform(f_t)  
imp_f_v = imp.transform(f_v)        
          
n_trees = 300
tree_features = 64
clf = HalfRF(imp_f_t, c_t, tree_features, n_trees)
n_validation = np.shape(f_v)[0]
results = np.zeros((n_validation, 3))
for i in range(n_validation):
    v_item = imp_f_v[i, :]
    (prediction, votes, total_votes) = clf.classify(v_item)
    results[i, :] = (prediction, votes, total_votes)
    #print("%f/%f" % (prediction, c_v[i]))
print(1.0* sum(results[:, 0] == c_v)/n_validation)
print(sum(results[:, 2] == 0))
print(np.mean(results[:, 2]))


imp_clf = RandomForestClassifier(n_trees, max_features=tree_features)
imp_clf.fit(imp_f_t, c_t)
imp_prediction = imp_clf.predict(imp_f_v)
print(1.0*sum(imp_prediction == c_v)/n_validation)
print("Hello World")