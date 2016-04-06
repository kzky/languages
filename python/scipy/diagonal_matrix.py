#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.sparse

"""
Create sprase diagonal matrix
"""

x = np.ones(10)
y = scipy.sparse.spdiags(x, 0, len(x), len(x))
print type(sp.sparse.csr_matrix(y))
print sp.sparse.csr_matrix(y)

