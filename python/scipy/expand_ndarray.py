import numpy as np
import scipy as sp

fin = "/home/kzk/datasets/uci_csv/iris.csv"
data = np.loadtxt(fin)
nrow = data.shape[0]
ncol = data.shape[1]

print np.row_stack((data, np.ones(ncol)))[-1, :]
print np.column_stack((data, np.ones(nrow)))[:, -1]

