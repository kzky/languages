#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared

print "##### Basic Examples #####"
print "Scala Addition"
x = T.dscalar("x")
y = T.dscalar("y")
z = x + y
f = function([x, y], z)
print f(2, 1.3)

print "Vector Manipulation"
x = T.dvector("x")
y = T.dvector("y")
z = x.dot(y)
f = function([x, y], z)
x_ = np.random.rand(10)
y_ = np.random.rand(10)
print f(x_, y_)

print "Logistic Function for Vector"
x = T.dvector("x")
s = 1 / (1 + T.exp(x))
f = function([x], s)
x_ = np.random.rand(10)
print f(x_)

print "Matrix Vecotr Product"
A = T.dmatrix("A")
b = T.dvector("b")
c = A.dot(b)
f = function([A, b], c)
A_ = np.random.rand(10, 10)
b_ = np.random.rand(10)
print f(A_, b_)

print "Broadcast"
A = T.dmatrix("A")
f = function([A], A * 5)
A_ = np.ones((5, 5))
print f(A_)

print "Shared Variables"
state = shared(0)
x = T.lscalar("x")
accumulator = function([x], state, updates=[(state, state + x)])
for i in xrange(0, 10):
    accumulator(i)
    pass
print state.get_value()


print "##### Derivatives #####"
print "Derivative of Vector-Vector Product w.r.t Vector"
x = T.dvector("x")
y = T.dvector("y")
z = T.dot(x, y)
gz_wrt_x = T.grad(z, x)
f = function([x, y], gz_wrt_x)  # d f(x) / dx
x_ = np.random.rand(5)
y_ = np.random.rand(5)
print f(x_, y_)
print y_  # recalculation

print "Derivative of Quadratic Form w.r.t. Vector"
x = T.dvector("x")
A = T.dmatrix("A")
z = x.dot(A).dot(x)
gz_wrt_x = T.grad(z, x)
f = function([x, A], gz_wrt_x)  # d f(x) / dx
x_ = np.random.rand(5)
A_ = np.random.rand(5, 5)
A_ = A_ + A_.T
print f(x_, A_)
print 2 * np.dot(A_, x_)  # recalculation

print "Derivative of 2-norm"
w = T.dvector("w")
z = T.grad(T.square(w.norm(L=2)) / 2, w)
f = function([w], z)
w_ = np.random.rand(5)
print f(w_, )
print w_  # recalculation

print "Derivative Trace of Matrix-Matrix Product"
A = T.dmatrix("A")
X = T.dmatrix("X")
z = A.dot(X).trace()
gz_wrt_X = T.grad(z, X)
f = function([A, X], gz_wrt_X)
A_ = np.random.rand(5, 5)
X_ = np.random.rand(5, 5)
print f(A_, X_)
print A_  # recalculation

print "Derivative of Log Determinant of Matrix"
X = T.dmatrix("X")
z = T.log(theano.sandbox.linalg.det(X))
gz_wrt_X = T.grad(z, X)
f = function([X], gz_wrt_X)
X_ = np.random.rand(5, 5)
X_ = (X_ + X_) / 2 + np.diag(np.ones(5) * 10)
X_ = np.random.rand(5, 5)
print f(X_)
print np.linalg.inv(X_)  # recalculation

print "##### Logistic Regression solved with Gradient Descent #####"
# Loading dataset
data = np.loadtxt("/home/kzk/datasets/uci_csv/breast_cancer.csv")
y_ = data[:, 0]
X_ = data[:, 1:]
set_y = list(set(y_))
for i, label in enumerate(y_):
    if label == set_y[0]:
        y_[i] = 1
    else:
        y_[i] = -1
    pass

# Learning setting
## param
max_iter = 1000
w_threshold = 0.001
step_size = 0.01
d = X_.shape[1]
n = X_.shape[0]

## variables
w = shared(np.random.rand(d), name="w")
b = shared(np.random.rand(1)[0], name="b")
X = T.dmatrix("X")
y = T.dvector("y")
c = 0.1  # lambda
regularizer = (w ** 2).sum() / 2
loss = T.log((1 + T.exp(-y * (X.dot(w) + b)))).sum()
obj_func = c * regularizer + loss / n
grad_obj_func_wrt_w = T.grad(obj_func, w)
grad_obj_func_wrt_b = T.grad(obj_func, b)

## train/predict
train = function(
    inputs=[X, y],
    outputs=[w, b],
    updates=[(w, w - step_size * grad_obj_func_wrt_w), (b, b - step_size * grad_obj_func_wrt_b)]
)

x = T.dvector("x")
y_pred = T.dscalar("y")
prob = 1 / (1 + T.exp(- y_pred * (w.dot(x) + b)))
predict = function(
    inputs=[x, y_pred],
    outputs=prob
)

print "Learn"
w_prev = w.get_value()
cnt = 0
while True:
    cnt += 1
    train(X_, y_)
    w_cur = w.get_value()
    diff_w = np.linalg.norm(w_cur) - np.linalg.norm(w_prev)
    print cnt, np.abs(diff_w)
    if np.abs(diff_w) < w_threshold or cnt > max_iter:
        break
        pass
    w_prev = w_cur

print "Predict"
hit = 0
for i, z in enumerate(X_):
    pred_1 = predict(z, 1)
    pred__1 = predict(z, -1)
    
    print "1 probability is", pred_1
    print "-1 probability is", pred__1
    print "label is ", y_[i]
    pred_value = 1 if pred_1 >= pred__1 else -1
    hit += 1 if pred_value == y_[i] else 0

print "Accuracy is ", (100.0 * hit/len(y_)), " %"

