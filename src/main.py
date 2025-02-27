import pandas as pd
import tensorflow

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, RepeatVector, TimeDistributed, Input, Activation, Lambda, Flatten
from keras.callbacks import ModelCheckpoint

"""
Convention:
Vector: 1D
Matrix: 2D
Tensor: 3D



Example of dataset read by pandas
Plain Text	Key	Cipher Text	Encrypted Text
taeniform	SECRET	SGSOLBQEKZ	TAENIFORM
fitting	SECRET	BLCZEMUA	FITTING
"""



EPORCHS = 10
LSTM_N_UNITS = 64
SIZE = 10000
TRAIN_SIZE = 10000 * 0.8
TEST_SIZE = 10000 * 0.2


# Maximize number of token in each vector
# => Convention fix in 40 character for input and 8 character for key
MAX_DECRYPT_SEQUENCES_LEN = 100 # Number of word, character to decrypt word
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

def decoder_off_set(decoder_input_data, offset):
    decoder_target_data = np.zeros_like(decoder_input_data)
    # Shift decoder_input_data by 'offset' to get decoder_target_data
    decoder_target_data[:, :-offset, :] = decoder_input_data[:, offset:, :]
    return decoder_target_data


def rnn_machine_translate_model(input_vocab, output_vocab, input_timesteps, output_timesteps, n_units, n_features):

    model = Sequential()
    # model.add(LSTM(units=n_units, return_sequences=True, input_shape=(input_timesteps, 1)))
    model.add(Embedding(input_vocab, n_units,input_length = input_timesteps, mask_zero=True))
    model.add(Dropout(0.2))
    model.add(LSTM(n_units))
    model.add(RepeatVector(output_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(Dense(output_vocab, activation='softmax'))
    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy',
    metrics=[
        tensorflow.keras.metrics.MeanSquaredError(name="mse"),
        tensorflow.keras.metrics.MeanAbsoluteError(name="msa"),
        tensorflow.keras.metrics.BinaryAccuracy(name="accuracy"),
    ]
    )

    print(model.summary())
    return model

def one_hot_encode(matrix, n_unique):
    one_hot_tensor = tensorflow.one_hot(matrix, depth=n_unique)
    return one_hot_tensor.numpy().astype(np.float32)

# decode a one hot encoded string
def one_hot_decode(tensor):
	# return [np.argmax(vector) for vector in matrix]
    decoded_arr = np.argmax(tensor, axis=-1)
    return decoded_arr

# def tensor_dim(lst):
#     """
#     Array to 3D for dim as numpy
#     """
#     return one_hot_encode(lst, ONE_HOT_SIZE)
    # n = np.array(lst)
    # return np.expand_dims(n, axis = 2)

def tensor_post_proccess(tensor):
    """
    Assume prediction value will have shape (None, 8)
    Normalize to (None, 8, 1), round floating point convert to int
    Prediction will not follow original style, we need to add more demension to decode
    """
    # seq = np.expand_dims(seq, axis = 2)
    normal_tensor = np.argmax(tensor, axis=-1)
    # matrix = one_hot_decode(normal_tensor)


    return normal_tensor


def decode_seq(matrix, tokenizer):
    """
    Assume input shape is (8, 1) from one vector
    Recontructure origin text base on input seq vector
    """
    return "".join(tokenizer.sequences_to_texts(matrix)).strip()

def evaluate(model, X_test, y_test, tokenizer):
    predictions = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1])), batch_size=64, verbose=0)
    predictions = tensor_post_proccess(predictions)
    # targets = tensor_post_proccess(y_test)

    targets = y_test

    # Reverser tokenizer and remove whitespace
    predictions = detokenizer(predictions, tokenizer)
    targets= detokenizer(y_test ,tokenizer )
    data = dict()
    data["predict"] = predictions
    data["actual"] = targets

    df = pd.DataFrame(data)
    df.to_csv("result.csv")
    print(df)
def detokenizer(matrix, tokenizer):
    return [word.replace(" ", "") for word in  tokenizer.sequences_to_texts(matrix)]


def tensor_to_2d_and_round(n):
    # Reduct dim
    n = np.squeeze(n, axis=2)
    n = np.round(n)
    return n

def train_test(model, X_train, y_train ,
               X_test, 	y_test, epochs=100,
							        verbose=0, patience=5):
	# patient early stopping
	#es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, patience=20)
	es = EarlyStopping(monitor='val_loss', mode='min',
	                   verbose=1, patience=patience)
	# train model
	print('training for ',epochs,
	      ' epochs begins with',
				' EarlyStopping(monitor= val_loss ',
				' patience=',patience,')....')
	history=model.fit(X_train, y_train, validation_split= 0.1, epochs=epochs,  verbose=verbose, callbacks=[es])
	print(epochs,' epoch training finished...')

	# report training
	# list all data in history
	#print(history.history.keys())
	# evaluate the model
	_, train_acc = model.evaluate(X_train, y_train, verbose=0)
	_, test_acc = model.evaluate(X_test, 	y_test, verbose=0)
	print('\nPREDICTION ACCURACY (%):')
	print('Train: %.3f, Test: %.3f' % (train_acc*100, test_acc*100))
	# summarize history for accuracy
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title(model.name+' accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(model.name+' loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()


def tensor_dim_normalize(matrix):
    """
    LSTM require to work with dim is 3 instead of 2
    This function convert matrix to tensor
    """
    # return one_hot_encode(lst, ONE_HOT_SIZE)
    n = np.array(matrix)
    # n_tensor = n.reshape(1, seq.shape[0], seq.shape[1])
    n_tensor = np.expand_dims(n, axis = 2)
    return n_tensor

def max_token_value(matrix):
    """
    Return highest value text token could be
    => Use to determine n_features by one_hot_encode
    """
    highest_value = 0
    for vector in matrix:
        highest_value = max(highest_value, max(vector))

    return highest_value



def main():
    filename = "PLAYFAIR_CIPHER_DATASET_RANDOM_KEY_10000.xlsx"
    feature_names= "Encrypted Text"
    label_name = "Key"
    features, labels = load_data(filename, feature_names, label_name)

    feature_tokenizer = Tokenizer(char_level=True)
    feature_tokenizer.fit_on_texts(features)

    label_tokenizer = Tokenizer(char_level=True)
    label_tokenizer.fit_on_texts(labels)

    # Tokenize and pad each separately
    features_padded = tokenize_normalize(feature_tokenizer, MAX_DECRYPT_SEQUENCES_LEN, features)[1]
    labels_padded = tokenize_normalize(label_tokenizer, MAX_KEY_SEQUENCES_LEN, labels)[1]


    # one_hot_depth = np.max(feature_padded) + 1

    # features_one_hot = one_hot_encode(feature_padded, one_hot_depth)
    # labels_one_hot = one_hot_encode(labels_padded, one_hot_depth)




    # Use ascii alphabert => About 26 character for encode in vectorizer
    encrypt_vocab_size = len(feature_tokenizer.word_index) + 4
    key_vocab_size = len(label_tokenizer.word_index) + 4



    # How freq they could be



    # X_train, X_test, y_train, y_test = train_test_split(features_one_hot, labels_one_hot, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(features_padded, labels_padded, test_size=0.2, random_state=42)

    # X_train, X_test, y_train, y_test = train_test_split(feature_padded, labels_padded , test_size=0.1, random_state=42)

    # model = rnn_machine_translate_model( src_seq_len = MAX_DECRYPT_SEQUENCES_LEN, tar_seq_len = MAX_KEY_SEQUENCES_LEN, n_units = LSTM_N_UNITS, n_features = one_hot_depth)
    model = rnn_machine_translate_model( input_vocab = encrypt_vocab_size, output_vocab = key_vocab_size, input_timesteps= MAX_DECRYPT_SEQUENCES_LEN, output_timesteps= MAX_KEY_SEQUENCES_LEN  ,n_units = LSTM_N_UNITS, n_features = 1)

    save_model_file = "translate.keras"
    checkpoint = ModelCheckpoint(save_model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1),
          epochs=EPORCHS, batch_size=1,
          validation_split = 0.2,
          callbacks=[checkpoint], verbose=1)

    # decoder_output = decoder_off_set(y_train, 1)
    # model.fit(X_train, y_train, epochs=EPORCHS, batch_size=64,  verbose=2)

    # model.save('s2s.h5')
    # train_test(model, X_train, y_train , X_test, 	y_test, epochs = EPORCHS)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train','validation'])
    plt.show()

    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = history.history['val_accuracy'][-1]

    print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
    print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

    evaluate(model, X_test, y_test, label_tokenizer)

if __name__ == '__main__':
    main()
