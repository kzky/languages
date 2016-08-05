# Recurrent Nets and their Computational Graph

## Recurrent Nets
- Normally, one case define a class one-step RNN inheriting Chain.
- Compute an objective over sequence which length can be either a variable length or a fixed-size length.
- Internal state can be accessed by l.h and intermel memory by l.c
- On step optimization is usullay as follows.

```python
rnn.reset_state()  # rnn is the RNN class inheriting Chain as the example
model.zerograds()
loss = compute_loss()  # equal to the objective function
optimizer.update()
```

## Truncate the Graph by Unchaining
- Truncated backprop is necessary because too long sequence is hard to be optimized and/or can not fit into the computer memory.
- Variable.unchain_backward() chops the computation history, and chopped variables are disposed if there is no reference to those.
- Even if the unchain_backward is called, internal state are referred by RNN instance and can be accessed.

## Network Evaluation without Storing the Computation History
- Volatile variable does not hold computational history, so it can be useful when the forward computation only occurs, i.e., the evaluation phase.
- Varialbe.volatie is also useful to create a predictor network based on a pre-trained feature extractor network by switching volatle flags as the following, which is for stopping the backward computation.

```python
x = Variable(x_data, volatile="on")
feet = feature_extractor(x)
feet.volatile = "off"
y = predictor_network(feet)
y.backword()
```

## Making it with Trainer
- If data is represented as very long sequence, use a customized iterator.
- Actually, the implementation is up to one, and if one wants to use Iterator, inherite chainer.dataset.Iterator and overwrite necessary functions.
