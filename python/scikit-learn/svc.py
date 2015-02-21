#!/usr/bin/python

from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)  
# prediction
print "prediction"
print clf.predict([[2., 2.]])

# get support vectors
print "get support vectors"
print clf.support_vectors_

# get indices of support vectors
print "get indices of support vectors"
print clf.support_ 

# get number of support vectors for each class
print "get number of support vectors for each class"
print clf.n_support_ 


