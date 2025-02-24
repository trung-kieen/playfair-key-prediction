import numpy as np

import pandas as pd
from pandas.core.internals.blocks import NumpyBlock
import tensorflow

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, RepeatVector, TimeDistributed, Input
from keras.callbacks import ModelCheckpoint

units = 10001
MAX_DECRYPT_SEQUENCES_LEN = 40
MAX_KEY_SEQUENCES_LEN = 8


def load_data(filename , feature_cols, label_col: str):
    df = pd.read_excel(filename)
    return df[feature_cols], df[label_col]



def tokenize_normalize(tokenizer, maxlen, corpus):
    """
    Perform tokenizer and padding message
    Required tokenizer fit on corpus before proccessing this action
    Example:
        tokenizer = Tokenizer(char_level = True)
        tokenizer.fit_on_texts(features)

    Padding ensure all sequence same length, so maxlen have higher length
    text length in corpus to garuantee not loss of data.

    Return tuple of tokenizered_corpus and padded_corpus
    """
    tokenizered_corpus = tokenizer.texts_to_sequences(corpus)
    padded_corpus = pad_sequences(tokenizered_corpus, maxlen= maxlen, padding="post", truncating="post")
    return tokenizered_corpus, padded_corpus

def rnn_machine_translate_model(src_seq_len, tar_seq_len, n_units):
    vocab_size = 10000  # Estimated vocabulary size
    embedding_dim = 128  # Size of word embeddings
    sequence_length = 40  # Number of words in each input sequence
    lstm_units = 256  # Number of LSTM cells

    model = Sequential([
        # Convert word indices into dense vectors
        # LSTM(32, input_shape=(40 , 1 )), # 40 vector of 1 features
        # # RepeatVector(...),
        # LSTM(..., return_sequences=True),
        # TimeDistributed(Dense(11, activation = "")),
        LSTM(128, input_shape=(40, 1), activation ="relu"),
        RepeatVector(8),
        # LSTM(100, activation ="sigmoid", return_sequences = True),
        # LSTM(100, activation ="sigmoid", return_sequences = True),
        LSTM(64, return_sequences=True, activation ="sigmoid"),
        LSTM(64, return_sequences=True, activation ="softmax"),
        TimeDistributed(Dense(1, activation='relu'))
    ])



    return model

def dim_adjust(lst):
    """
    Array to 3D for dim as numpy
    """
    n = np.array(lst)
    return np.expand_dims(n, axis = 2)
def main():
    filename = "PLAYFAIR_CIPHER_DATASET_RANDOM_KEY.xlsx"
    feature_names= "Decrypted Text"
    label_name = "Key"
    features, labels = load_data(filename, feature_names, label_name)


    # print(features)
    # print(labels.shape)
    feature_tokenizer = Tokenizer(char_level = True)
    feature_tokenizer.fit_on_texts(features)
    label_tokenizer = Tokenizer(char_level = True)
    label_tokenizer.fit_on_texts(labels)

    feature_padded = tokenize_normalize(feature_tokenizer, MAX_DECRYPT_SEQUENCES_LEN, features)[1]
    labels_padded = tokenize_normalize(label_tokenizer, MAX_KEY_SEQUENCES_LEN, labels)[1]

    X_train, X_test, y_train, y_test = train_test_split(feature_padded, labels_padded, test_size=0.2, random_state=42)
    X_train = np.array(X_train).reshape(8000, 40, 1)
    y_train = np.array(y_train).reshape(8000, 8, 1)
    X_test = np.array(X_test).reshape(2000, 40, 1)
    y_test = np.array(y_test).reshape(2000, 8, 1)
    print(X_train.shape)
    print(X_test)
    N_UNITS = 256
    model = rnn_machine_translate_model( MAX_DECRYPT_SEQUENCES_LEN, MAX_KEY_SEQUENCES_LEN, N_UNITS)
    print(model.summary())
    model.compile(optimizer = "adam", loss = "binary_crossentropy")
    # model.compile(optimizer = "adam", loss = "binary_crossentropy")
    # model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)
    saved_model= 'model.h5'
    # checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=2)
    # model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=2)
    # plot_model(model, to_file='model.png', show_shapes=True)


if __name__ == '__main__':
    main()
