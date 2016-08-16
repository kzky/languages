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

class CNN(Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            cn1=L.Convolution2D(1, 64, (3, 3)),
            cn2=L.Convolution2D(64, 32, (3, 3)),
            # initialization is deferred until the first forward pass
            l1=L.Linear(1152, 100),
            #l1=L.Linear(None, 100),
            l2=L.Linear(100, 10),
            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(100),
        )
        self.train = train

    def __call__(self, x):
        # Reshpae
        x0 = F.reshape(x, (len(x.data), 1, 28, 28))

        # 2 layers conv
        h = F.relu(F.max_pooling_2d(self.bn1(self.cn1(x0)), (2, 2), (2, 2)))
        h = F.relu(F.max_pooling_2d(self.bn2(self.cn2(h)), (2, 2), (2, 2)))

        # 2 layers affine
        h = F.relu(self.bn3(self.l1(h)))
        y = self.l2(h)

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
    model = Classifier(CNN())
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
    train_iter = iterators.SerialIterator(train, batch_size=128, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=128,
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
