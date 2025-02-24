import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, label_ranking_average_precision_score
from sklearn.feature_extraction.text import CountVectorizer


dataset_file = "PLAYFAIR_CIPHER_DATASET.xlsx"

df = pd.read_excel(dataset_file)
dataset_columns = ["Plain Text", "Key", "Cipher Text", "Decrypted Text"]



x = df["Plain Text"]
df.to_numpy().reshape(df.shape[0], df.shape[1], 1)
print(df)
# print(x )
# x.reshape(len(x), 1)
# print(x)
