import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, label_ranking_average_precision_score
from sklearn.feature_extraction.text import CountVectorizer


dataset_file = "PLAYFAIR_CIPHER_DATASET.xlsx"

df = pd.read_excel(dataset_file)
dataset_columns = ["Plain Text", "Key", "Cipher Text", "Decrypted Text"]

def vectorize(corpus, dataset_columns):

    rs = dict()
    for col in dataset_columns:
        vectorizer = CountVectorizer()
        result = vectorizer.fit_transform(corpus[col])
        rs[col] = result.toarray()

        print(rs )
    return rs


corpus = vectorize(df, dataset_columns)



"""

       Plain Text     Key     Cipher Text  Decrypted Text
0       taeniform  SECRET      SGSOLBQEKZ       TAENIFORM
1         fitting  SECRET        BLCZEMUA         FITTING
2       ozocerite  SECRET      UWPECTMECW       OZOCERITE
3  interventralia  SECRET  HOSCSYSOSTFHHB  INTERVENTRALIA
4        orthopod  SECRET        QESMPQPB        ORTHOPOD


"""
predict_cols = ["Plain Text", "Key", "Cipher Text"]
label_col = "Decrypted Text"





# X = dict()
# for feature in predict_cols:
#     X = corpus[feature]
# y = corpus[label_col]



# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.10, random_state=42)
# model = LogisticRegression()
# model.fit(X_train, y_train)


# predictions = model.predict(X_test)
# predictions["Label"]= y[label_col]
# # print(predictions)


# print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
