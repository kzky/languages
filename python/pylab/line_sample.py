"""
Demo of a simple plot with a custom dashed line.

A Line object's ``set_dashes`` method allows you to specify dashes with
a series of on/off lengths (in points).
"""
import numpy as np
import matplotlib.pyplot as plt

## setup components
x = np.linspace(0, 10, 100)
#line, = plt.plot(x, np.sin(x), "--", linewidth=2)
line = plt.plot(x, np.sin(x), "-", linewidth=2)
dashes = [10, 5, 100, 2]
#line.set_dashes(dashes)

## show
plt.show()

