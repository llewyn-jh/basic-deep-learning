"""Recurrent Neural Network.
Study principals of a recurrent neural newtwork."""

import tensorflow as tf
import numpy as np

class RecurrentNetwork:
    """A network is made up varialbles initialize, forward,
       backward, train, fit, calcualte validation loss,
       activation, dataset batch, predict, accuracy methods."""

    def __init__(self, n_cells=10, batch_size=32, learning_rate=0.1):
        """Initialize all of variables except for kernels.
        n_cells: the number of neurals in a hidden layer, default: 10
        batch_size: batch size of train, validation dataset, default: 32
        learning_rate: 0~1, updating rate of weights, default: 0.1"""

        self.n_cells = n_cells
        self.batch_size = batch_size
        self.w1h = None
        self.w1x = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.h = None
        self.losses = []
        self.val_losses = []
        self.lr = learning_rate

    def init_weights(self, n_features, n_classes):
        """Set initial kernels of hidden layers.
        We use Orthogonal, GlorotUniform of tensorflow initializers and numpy zeros.
        n_features: the number of cells
        n_classes: the number of classes what you want to detect"""

        orth_init = tf.initializers.Orthogonal()
        glorot_init = tf.initializers.GlorotUniform()
        self.w1h = orth_init((self.n_cells, self.n_cells)).numpy()
        self.w1x = glorot_init((n_features, self.n_cells)).numpy()
        self.b1 = np.zeros(self.n_cells)
        self.w2 = glorot_init((self.n_cells, n_classes)).numpy()
        self.b2 = np.zeros(n_classes)

    def forpass(self, x):
        """Calculate a forward in a network. Return values before final activation in a network.
        x: input data, training or validation dataset.
        You have to encode a dataset to one hot encoding.
        A shape of data is (self.batch_size, n_features, n_features)"""

        self.h = [np.zeros((x.shape[0], self.n_cells))]
        seq = np.swapaxes(x, 0, 1)
        for sample in seq:
            z1 = np.dot(sample, self.w1x) + np.dot(self.h[-1], self.w1h) + self.b1
            h = np.tanh(z1)
            self.h.append(h)
            z2 = np.dot(h, self.w2) + self.b2
        return z2

    def backprop(self, x, err):
        """Calculate a backword in a network. Return gradients of kernels ans biases
        x: input data, training or validation dataset.
        You have to encode a dataset to one hot encoding.
        A shape of data is (self.batch_size, n_features, n_features)
        err: differences between targets and activations, "- (targets - activations)" """

        m = len(x)

        w_2_grad = np.dot(self.h[-1], err) / m
        b_2_grad = np.sum(err) / m

        seq = np.swapaxes(x, 0, 1)
        w1h_grad = w1x_grad = b_1_grad = 0
        err2cell = np.dot(err, self.w2.T) * (1 - self.h[-1] ** 2)

        for sample, h in zip(seq[::-1][:10], self.h[:-1][::-1][:10]):
            w1h_grad += np.dot(h.T, err2cell)
            w1x_grad += np.dot(sample.T, err2cell)
            b_1_grad += np.sum(err2cell, axis=0)
            err2cell = np.dot(err2cell, self.w1h) * (1 - h ** 2)

        w1h_grad /= m
        w1x_grad /= m
        b_1_grad /= m

        return w1h_grad, w1x_grad, b_1_grad, w_2_grad, b_2_grad

    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        """Train a network. Not save all of kernels and biases in a network.
        x, x_val: input data, training or validation dataset.
        You have to encode a dataset to one hot encoding.
        A shape of data is (self.batch_size, n_features, n_features)
        y, y_val: input data, target of train or validation dataset.
        if you want to classfy multi classes, you have to do one hot encoding and modify some code.
        epochs: the number of training loop, default=100"""

        y = y.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        np.random.seed(42)
        self.init_weights(x.shape[2], y.shape[1])
        for i in range(epochs):
            print('Epoch', i, end=' ')
            batch_losses = []
            for x_batch, y_batch in self.gen_batch(x, y):
                print('.', end='')
                z = self.forpass(x_batch)
                a = 1 / (1 + np.exp(-z))
                err = - (y_batch - a)
                w1h_grad, w1x_grad, b_1_grad, w_2_grad, b_2_grad = self.backprop(x_batch, err)
                self.w1h -= self.lr * w1h_grad
                self.w1x -= self.lr * w1x_grad
                self.b1 -= self.lr * b_1_grad
                self.w2 -= self.lr * w_2_grad
                self.b2 -= self.lr * b_2_grad
                a = np.clip(a, 1e-10, 1-1e-10)
                loss = np.mean(-(y_batch * np.log(a) + (1 - y_batch) * np.log(1 - a)))
                batch_losses.append(loss)
            print()
            self.losses.append(np.mean(batch_losses))
            self.update_val_losses(x_val, y_val)

    def gen_batch(self, x, y):
        """Generate batch of dataset
        x: input data, training or validation dataset.
        y: input data, target of train or validation dataset."""

        length = len(x)
        bins = length // self.batch_size
        if length % self.batch_size:
            bins += 1
        indexes = np.random.permutation(np.arange(len(x)))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]

    def update_val_losses(self, x_val, y_val):
        """Calculate and update losses of validation dataset.
        x_val: input data, training or validation dataset.
        You have to encode a dataset to one hot encoding.
        A shape of data is (self.batch_size, n_features, n_features)
        y_val: input data, target of train or validation dataset.
        if you want to classfy multi classes,
        you have to do one hot encoding and modify some code."""

        z = self.forpass(x_val)
        a = 1 / (1 + np.exp(-z))
        a = np.clip(a, 1e-10, 1-1e-10)
        loss = np.mean(-(y_val * np.log(a) + (1 - y_val) * np.log(1 - a)))
        self.val_losses.append(loss)

    def predict(self, x):
        """Predict a result. Return True or False.
        x: input data, training or validation dataset.
        You have to encode a dataset to one hot encoding.
        A shape of data is (self.batch_size, n_features, n_features)
        y: input data, target of train or validation dataset."""

        z = self.forpass(x)
        a = 1 / (1 + np.exp(-z))
        return a > 0.5

    def score(self, x, y):
        """Calculate accuracy of a network.
        x_val: input data, training or validation dataset.
        You have to encode a dataset to one hot encoding.
        A shape of data is (self.batch_size, n_features, n_features)
        y_val: input data, target of train or validation dataset."""

        return np.mean(self.predict(x) == y.reshape(-1, 1))