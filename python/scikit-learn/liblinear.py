#!/usr/bin/env python
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix


# loading
filename = "/home/kzk/datasets/news20/news20.dat"
st = time.time()
(X, y) = load_svmlight_file(filename)
et = time.time()
print "ellapsed time: %f [s]" % (et - st)

# learning
svc = LinearSVC()
st = time.time()
svc.fit(X, y)
et = time.time()
print "ellapsed time: %f [s]" % (et - st)


y_pred = svc.predict(X)
cm = confusion_matrix(y, y_pred)
#print cm
print "accurary: %d [%%]" % (np.sum(cm.diagonal()) * 100.0 / np.sum(cm))

