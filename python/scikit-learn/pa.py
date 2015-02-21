#!/usr/bin/python

import numpy as np
import pylab as pl

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits()

X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)

# classify small against large digits
y = (y > 4).astype(np.int)

clf = PassiveAggressiveClassifier( loss='squared_hinge', shuffle = True, n_jobs = 2, warm_start = True)
clf.fit(X, y)

label = y
p_label = clf.predict(X)
print "Confusion Matrix for Digits Dataset (for training sample)"
print confusion_matrix(label, p_label)








