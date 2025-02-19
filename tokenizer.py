"""
Try to encode and decode textual message with Bag Of Word
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# How to actually convert textual data


corpus = ["This is the first document.",
          "This document is the second document",
          "And this is the third one.",
          "Is this the first document?",
]


# Tokenizer textual data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)


bag_of_words = vectorizer.get_feature_names_out()
print("Bad of word create for toeknizer", bag_of_words)
print("Numberic vector data to matrix A")
print(bag_of_words)
print(X.toarray())


print("First row value repersent in numberic")
print(X[0])


# Result just reconstruct word base not meaning
print("Revserse sentence form row 1")
reversed_value =np.array(X[0].toarray())
print(reversed_value * bag_of_words)
