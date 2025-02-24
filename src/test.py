import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import pandas as pd
import openpyxl
# import keras




def read_cipher_text(file_path):
    df = pd.read_excel(file_path)
    return df['Cipher Text'], df['Cipher Label']

cipher_texts, labels_text = read_cipher_text("PLAYFAIR_CIPHER.py")


tokenizer = keras.preprocessing.text.Tokenizer(char_level = True)
print("HI")
