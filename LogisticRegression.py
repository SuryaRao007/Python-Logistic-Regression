#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:36:07 2021

@author: surya pg. 66, see page 53 for training data and standardization
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X=iris.data[:,[2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1, stratify=y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_train_01_subset = X_train[(y_train==0)|(y_train==1)]
y_train_01_subset = y_train[(y_train==0)|(y_train==1)]
lrgd = LogisticRegressionGD(eta=0.05,n_iter=1000,random_state=1)
plot_decision_regions(X=X_train_01_subset,y=y_train_01_subset,classifier=lrgd)
"""
import numpy as np

class LogisticRegressionGD(object):
    
    def __init__(self,eta=0.05,n_iter=100, random_state=1):
        self.eta =eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale =0.01, size = 1+X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0]+= self.eta * errors.sum()
            #compute logistic cost
            cost = (-y.dot(np.log(output))-((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)
            return self
        
    def net_input(self, X):
        return np.dot(X,self.w_[1:] + self.w_[0])
    
    def activation(self,z):
        #compute logistic sigmoid activation
        return 1./(1. + np.exp(-np.clip(z,-250,250)))
    
    def predict(self,X):
        #return class label after unit step
        return np.where(self.net_input(X) >= 0.0,1,0)