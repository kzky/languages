from __future__ import division
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class RNN(Chain):
    def __init__(self, train=True):
        super(RNN, self).__init__(
            embed=L.EmbedID(10000, 1000),
            mid=L.LSTM(1000, 100),
            out=L.Linear(100, 10000)
        )

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train
        
    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y

class ParallelSequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.offsets = [i * len(dataset) // batch_size for i in range(batch_size)]
        self.iteration = 0

    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, bprop_len, device=None):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.zerograds()  # Initialize the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    # Model
    rnn = RNN()
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    
    # Optimizer
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    
    # Dataset
    train, val, test = chainer.datasets.get_ptb_words()

    # Iterator
    train_iter = ParallelSequentialIterator(train, 20)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Updater
    updater = BPTTUpdater(train_iter, optimizer, bprop_len=35)
    trainer = training.Trainer(updater, (10, "epoch"), out="result")

    # Add extensions
    interval = 10
    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=None,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
        update_interval=1))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    
    # Run
    trainer.run()

    # Evaluate the final model
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_model, device=None)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))

if __name__ == "__main__":
    main()
