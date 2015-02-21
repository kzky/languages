import time
from sklearn.datasets import load_svmlight_file

filename = "/home/kzk/datasets/news20/news20.dat"
st = time.time()
(X, y) = load_svmlight_file(filename)
et = time.time()
print "ellapsed time: %f [s]" % (et - st)

