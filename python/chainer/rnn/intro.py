import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

# Sampls
print("# Sampls")
l = L.LSTM(100, 50)
l.reset_state()
x = Variable(np.random.randn(10, 100).astype(np.float32))
y = l(x)

print(l.c.data)  # internal memory
print(l.h.data)  # internal state
print(y.data)
print(np.all(l.h.data == y.data))

# RNN model
print("# RNN model for one-step computation")
class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__(
            embed=L.EmbedID(1000, 100),
            mid=L.LSTM(100, 50),
            out=L.Linear(50, 1000)
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y

rnn = RNN()
model = L.Classifier(rnn)
optimizer = optimizers.SGD()
optimizer.setup(model)

# Loss
#def compute_loss(x_list):
#    loss = 0
#    for cur_word, next_word in zip(x_list, x_list[1:]):
#        loss += model(cur_word, next_word)
#    return loss

# Optimization for one sequence
## Suppose we have a list of word variables x_list.
#rnn.reset_state()
#model.zerograds()
#loss = compute_loss(x_list)
#loss.backward()
#optimizer.update()
##or
#rnn.reset_state()
#optimizer.update(compute_loss, x_list)

# Truncated backprop
#loss = 0
#count = 0
#seqlen = len(x_list[1:])
# 
#rnn.reset_state()
#for cur_word, next_word in zip(x_list, x_list[1:]):
#    loss += model(cur_word, next_word)
#    count += 1
#    if count % 30 == 0 or count == seqlen:
#        model.zerograds()
#        loss.backward()
#        loss.unchain_backward()
#        optimizer.update()


