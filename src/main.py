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

SIZE = 10000
TRAIN_SIZE = 10000 * 0.8
TEST_SIZE = 10000 * 0.8
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
        Input(shape = (sequence_length, 1)),
        # Convert word indices into dense vectors
        # LSTM(32, input_shape=(40 , 1 )), # 40 vector of 1 features
        # # RepeatVector(...),
        # LSTM(..., return_sequences=True),
        # TimeDistributed(Dense(11, activation = "")),
        LSTM(128, input_shape=(MAX_DECRYPT_SEQUENCES_LEN, 1),
             activation = "relu"
             ),
        RepeatVector(8),
        LSTM(100, activation ="relu", return_sequences = True),
        # LSTM(100, activation ="sigmoid", return_sequences = True),
        LSTM(64, return_sequences=True,
             # activation = "softmax"
             ),
        # LSTM(64, return_sequences=True,
        #      activation = "softmax"
        #      ),
        Dense(128, activation="relu"),
        Dropout(0.5),

    # Add an output layer with output units for all 10 digits
        Dense(10, activation="softmax"),

        # LSTM(64, return_sequences=True,
        #      activation = "softmax"
        #      ),
        TimeDistributed(1, Dense(1,
                              activation="relu"
                              ))
    ])



    return model

def dim_adjust(lst):
    """
    Array to 3D for dim as numpy
    """
    n = np.array(lst)
    return np.expand_dims(n, axis = 2)


def evaluate(model, X_test, y_test, tokenizer):
    predictions = model.predict(X_test, batch_size=64, verbose=0)
    for predict, actual in zip(predictions[:2], y_test[:2]):
        print("Predict" )
        print(predict)
        print("Actual value is")
        print(actual)

def tensor_dim_normalize(lst):
    """
    LSTM require to work with dim is 3 instead of 2
    """
    n = np.array(lst)
    n_tensor = np.expand_dims(n, axis = 2)
    return n_tensor
def main():
    filename = "PLAYFAIR_CIPHER_DATASET.xlsx"
    # filename = "PLAYFAIR_CIPHER_DATASET_RANDOM_KEY.xlsx"
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

    feature_padded = tensor_dim_normalize(feature_padded)
    labels_padded = tensor_dim_normalize(labels_padded)
    print(feature_padded)
    print(feature_padded.shape)
    X_train, X_test, y_train, y_test = train_test_split(feature_padded, labels_padded, test_size=0.2, random_state=42)
    N_UNITS = 256
    model = rnn_machine_translate_model( MAX_DECRYPT_SEQUENCES_LEN, MAX_KEY_SEQUENCES_LEN, N_UNITS)
    print(model.summary())
    # model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.compile(optimizer = "adam", loss = "binary_crossentropy")
    # model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)
    # saved_model= 'model.h5'
    # checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train, y_train, epochs=5, batch_size=40, validation_data=(X_test, y_test), verbose=2)
    # model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=2)
    # plot_model(model, to_file='model.png', show_shapes=True)


    evaluate(model, X_test, y_test, label_tokenizer)

if __name__ == '__main__':
    main()
