#!/usr/bin/env python

import scipy.stats as sst
import numpy as np
import scipy.special as ssp

gdist = sst.gamma(a=3, loc=0, scale=1)
data = gdist.rvs(10000)

# estimation by what?
print sst.gamma.fit(data)

# moment estimation
m = np.mean(data)
v = np.var(data)
shape = (m/v) ** 2
scale = v**2 / m

print shape
print scale








