#!/usr/bin/env python

# Sed definitly http://matplotlib.org/faq/usage_faq.html

import matplotlib.pylab as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# The easiest way to create a new figure is with pyplot:
## fig = plt.figure()  # an empty figure with no axes
## fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes


# sample 1
fig, ax = plt.subplots(1, 1)  # a figure with a 1x1 grid of Axes

x = np.array([1, 2, 3, 4, 5])
y = np.power(x, 2)
e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])

## integer x-axias value
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.errorbar(x, y, e, linestyle='None', marker='^')
fig.canvas.draw()

# sample 2
x = np.arange(-2, 2, 0.01) * np.pi
y1 = np.sin(x)
y2 = np.cos(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y1)
ax.plot(x, y2)
fig.canvas.draw()
