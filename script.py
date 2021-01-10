#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 10:27:28 2021

@author: surya
"""

from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import LogisticRegression
import plot_decision_regions
iris = datasets.load_iris()
X=iris.data[:,[2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_train_01_subset = X_train[(y_train==0)|(y_train==1)]
y_train_01_subset = y_train[(y_train==0)|(y_train==1)]
lrgd = LogisticRegression.LogisticRegressionGD(eta=0.01,n_iter=1000,random_state=1)
lrgd.fit(X_train_01_subset,y_train_01_subset)
plot_decision_regions.plot_decision_regions(X=X_train_01_subset,y=y_train_01_subset,classifier=lrgd)