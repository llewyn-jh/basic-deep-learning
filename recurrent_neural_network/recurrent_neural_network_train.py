"""Train a recurrent neural network.
Dataset is the IMDB from Keras.
"""

import argparse
import warnings
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from recurrent_neural_network import RecurrentNetwork

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a recurrent neural network")
    parser.add_argument("--num_words",
                        required=False,
                        default=100,
                        help="The number of word that will be used to train a network.")
    parser.add_argument("--skip_top",
                        required=False,
                        default=20,
                        help="The number of word that will be not used to train a network.")
    parser.add_argument("--n_cells",
                        required=False,
                        default=32)
    parser.add_argument("--batch_size",
                        required=False,
                        default=32)
    parser.add_argument("--learning_rate",
                        required=False,
                        default=0.01)

    args = parser.parse_args()

    print("A kind of train word: ", args.num_words)
    print("A kind of non train word: ", args.skip_top)
    print("The number of cells: ", args.n_cells)
    print("Batch size: ", args.batch_size)
    print("Learning rate: ", args.learning_rate)

    # Load tensorflow keras imdb dataset.
    print("Loading...")
    (x_train_all, y_train_all), (x_test, y_test) = \
        imdb.load_data(skip_top=args.skip_top, num_words=args.num_words)

    # Remove all paddings 0, indents 1, no words 2.
    for i, sample in enumerate(x_train_all):
        x_train_all[i] = [n for n in sample if n > 2]

    # Load a dictionary.
    word2index = imdb.get_word_index()
    index2word = {word2index[k]: k for k in word2index}

    # Shffle dataset.
    np.random.seed(42)
    random_index = np.random.permutation(25000)
    x_train = x_train_all[random_index[:20000]]
    y_train = y_train_all[random_index[:20000]]
    x_val = x_train_all[random_index[20000:]]
    y_val = y_train_all[random_index[20000:]]

    # Equalize all length of a sentences.
    x_train = sequence.pad_sequences(x_train, maxlen=args.num_words)
    x_val = sequence.pad_sequences(x_val, maxlen=args.num_words)

    # One hot encoding.
    x_train_onehot = to_categorical(x_train)
    x_val_onebot = to_categorical(x_val)

    # Train a recurrent_neral_network
    print("Training a network...")
    rnn = RecurrentNetwork(n_cells=32, batch_size=32, learning_rate=0.01)
    rnn.fit(x_train_onehot, y_train, epochs=20, x_val=x_val_onebot, y_val=y_val)
    accuracy = rnn.score(x_val_onebot, y_val)
    print("Accuracy: {}".format(accuracy))
