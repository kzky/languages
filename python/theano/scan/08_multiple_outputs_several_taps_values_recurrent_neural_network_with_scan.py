import theano
import numpy as np
from theano import tensor as T

def oneStep(u_tm4, u_t, x_tm3, x_tm1, y_tm1, W, W_in_1, W_in_2, W_feedback, W_out):

    x_t = T.tanh(theano.dot(x_tm1, W) +
                 theano.dot(u_t, W_in_1) +
                 theano.dot(u_tm4, W_in_2) +
                 theano.dot(y_tm1, W_feedback))
    y_t = theano.dot(x_tm3, W_out)

    return [x_t, y_t]

W_ = np.random.rand(10, 10).astype(np.float32)
W_in_1_ = np.random.rand(10, 10).astype(np.float32)
W_in_2_ = np.random.rand(10, 10).astype(np.float32)
W_feedback_ = np.random.rand(10, 10).astype(np.float32)
W_out_ = np.random.rand(10, 10).astype(np.float32)
    
W = theano.shared(W_)
W_in_1 = theano.shared(W_in_1_)
W_in_2 = theano.shared(W_in_2_)
W_feedback = theano.shared(W_feedback_)
W_out = theano.shared(W_out_)

u = T.matrix()    # it is a sequence of vectors
x0 = T.matrix()  # initial state of x has to be a matrix, since it has to cover x[-3]
y0 = T.vector()  # y0 is just a vector since scan has only to provide y[-1]

([x_vals, y_vals], updates) = theano.scan(fn=oneStep,
                                          sequences=dict(input=u, taps=[-4, -0]),
                                          outputs_info=[dict(initial=x0, taps=[-3, -1]), y0],
                                          non_sequences=[W, W_in_1, W_in_2, W_feedback, W_out],
                                          strict=True)
# for second input y, scan adds -1 in output_taps by default

func = theano.function(inputs=[u, x0, y0], outputs=[x_vals, y_vals], updates=updates)

u_ = np.random.rand(50, 10).astype(np.float32)
x0_ = np.random.rand(5, 10).astype(np.float32)
y0_ = np.random.rand(10).astype(np.float32)

print func(u_, x0_, y0_)
