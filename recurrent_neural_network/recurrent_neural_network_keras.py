"""Train a recurrent neural network.
Dataset is the IMDB from Keras.
"""

import argparse
import warnings
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, BatchNormalization, Dropout, Conv1D, GlobalMaxPooling1D

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a recurrent neural network")
    parser.add_argument("--num_words",
                        required=False,
                        default=1000,
                        type=int,
                        help="Set the number of word that will be used to train a network.")
    parser.add_argument("--skip_top",
                        required=False,
                        type=int,
                        default=20,
                        help="Set the number of word that will be not used to train a network.")
    parser.add_argument("--max_len",
                        required=False,
                        type=int,
                        default=100,
                        help="Set an explicit maximum sequence length.")
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        default=32)
    parser.add_argument("--epochs",
                        required=False,
                        type=int,
                        default=10)

    args = parser.parse_args()

    print("A kind of train word: ", args.num_words)
    print("A kind of non train word: ", args.skip_top)
    print("Maxium sequence length: ", args.max_len)
    print("Batch size: ", args.batch_size)
    print("Epochs: ", args.epochs)

    # Load IMDB dataset
    (x_train_all, y_train_all), (x_test, y_test) = \
        imdb.load_data(num_words=args.num_words, skip_top=args.skip_top)

    # Remove a digit 0, 1, and 2
    for i, sample in enumerate(x_train_all):
        x_train_all[i] = [n for n in sample if n > 2]

    # Shuffle dataset
    random_index = np.random.permutation(np.arange(25000))
    x_train = x_train_all[random_index[:20000]]
    y_train = y_train_all[random_index[:20000]]
    x_val = x_train_all[random_index[20000:]]
    y_val = y_train_all[random_index[20000:]]

    # Equalize all length of sentences
    x_train_seq = sequence.pad_sequences(x_train, args.max_len)
    x_val_seq = sequence.pad_sequences(x_val, args.max_len)

    # Build a model
    model = Sequential()
    model.add(Embedding(args.num_words, 128))
    model.add(LSTM(64))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Compile a model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train_seq,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val_seq, y_val),
        verbose=2
        )

    # Check a result
    loss, accuracy = model.evaluate(x_val_seq, y_val, verbose=2)
