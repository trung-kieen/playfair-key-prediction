import pandas as pd
import tensorflow

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, RepeatVector, TimeDistributed, Input, Activation, Lambda
from keras.callbacks import ModelCheckpoint

ONE_HOT_SIZE= 50 # Max size of number by one hot encoder
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

    vocab_size = 10000  # Estimated vocabulary size
    embedding_dim = 128  # Size of word embeddings
    sequence_length = 40  # Number of words in each input sequence
    lstm_units = 256  # Number of LSTM cells

    inputs = Input(name='inputs',shape=[src_seq_len])
    layer = Embedding(vocab_size, 1 ,input_length=src_seq_len)(inputs)
    layer = LSTM(1, input_shape=(src_seq_len, 1), return_sequences = True)(layer)
    layer = LSTM(1, input_shape=(src_seq_len, 1))(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(tar_seq_len,name='out_layer'  )(layer)

    # This layer result an interger
    layer = Activation('sigmoid')(layer)
    # layer = Activation('relu')(layer)
    layer = Dense(tar_seq_len)(layer)
    # layer = Dense(tar_seq_len,activation = "relu")(layer)
    model = Model(inputs=inputs,outputs=layer)

    model.compile(optimizer = 'rmsprop',loss =  'mse', metrics = ["accuracy"])

    return model

def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return np.array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [np.argmax(vector) for vector in encoded_seq]

# def tensor_dim(lst):
#     """
#     Array to 3D for dim as numpy
#     """
#     return one_hot_encode(lst, ONE_HOT_SIZE)
    # n = np.array(lst)
    # return np.expand_dims(n, axis = 2)

def key_seq_normalization(seq):
    """
    Assume prediction value will have shape (None, 8)
    Normalize to (None, 8, 1), round floating point convert to int
    Prediction will not follow original style, we need to add more demension to decode
    """
    seq = np.expand_dims(seq, axis = 2)
    normal = np.abs(np.round(seq)).astype(np.int64)
    return normal

def decode_seq(seq, tokenizer):
    """
    Assume input shape is (8, 1) from one vector
    Recontructure origin text base on input seq vector
    """
    return "".join(tokenizer.sequences_to_texts(seq)).strip()

def evaluate(model, X_test, y_test, tokenizer):
    predictions = model.predict(X_test, batch_size=64, verbose=0)
    print(predictions)
    print(predictions.shape)
    predictions = key_seq_normalization(predictions)

    data = dict()
    data["predict"] = []
    data["actual"] = []

    # predictions = tensor_to_2d_and_round(predictions)
    for predict, actual in zip(predictions, y_test):
        predict = decode_seq(predict, tokenizer)
        actual = decode_seq(actual, tokenizer)
        data["predict"].append(predict)
        data["actual"].append(actual)
    # Save in file
    df = pd.DataFrame(data)
    df.to_csv("result.csv")
    print(df)


def tensor_to_2d_and_round(n):
    # Reduct dim
    n = np.squeeze(n, axis=2)
    n = np.round(n)
    return n


def tensor_dim_normalize(seq):
    """
    LSTM require to work with dim is 3 instead of 2
    """
    # return one_hot_encode(lst, ONE_HOT_SIZE)
    n = np.array(seq)
    # n_tensor = n.reshape(1, seq.shape[0], seq.shape[1])
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
    X_train, X_test, y_train, y_test = train_test_split(feature_padded, labels_padded, test_size=0.2, random_state=42)
    N_UNITS = 256
    model = rnn_machine_translate_model( src_seq_len = MAX_DECRYPT_SEQUENCES_LEN, tar_seq_len = MAX_KEY_SEQUENCES_LEN, n_units = N_UNITS)
    print(model.summary())


    model.fit(X_train, y_train, epochs=5, batch_size=40, validation_data=(X_test, y_test), verbose=2)


    evaluate(model, X_test, y_test, label_tokenizer)

if __name__ == '__main__':
    main()
