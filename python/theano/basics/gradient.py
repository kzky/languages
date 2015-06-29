#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
import theano
from theano import function
from theano import shared

w = T.dvector("w")
w_L2_norm_2 = T.square(w.norm(L=2))
g_w_L2_norm_2 = T.grad(w_L2_norm_2, w)

print "To see the graph, difficult to see the result in math form..."
theano.pp(g_w_L2_norm_2)


