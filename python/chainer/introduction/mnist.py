"""Example of MNIST
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
                    
class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({"loss": loss, "accuracy": accuracy}, self)
        return loss

if __name__ == '__main__':
    # Model and Optimizer
    model = Classifier(MLP())
    try: 
        model.to_gpu(0)
    except Exception as e:
        print(e)

    # Optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model)
     
    # Dataset
    train, test = datasets.get_mnist()
     
    # Iterators
    train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=100,
                                         repeat=False, shuffle=False)
     
    # Updater and Trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (20, "epoch"), out="result")
     
    # Add extensions
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/accuracy", "validation/main/accuracy"]))
    trainer.extend(extensions.ProgressBar())
     
    # Run
    trainer.run()
