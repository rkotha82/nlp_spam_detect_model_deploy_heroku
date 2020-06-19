import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.externals import joblib
import joblib
import pickle

df = pd.read_csv('spam.csv', encoding = 'latin-1')  # Read the dataset
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)  # dropped these cols

# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

# Extract Feature with CountVectorizer
cv = CountVectorizer()  # We use this library to create the vectors from the text
X = cv.fit_transform(X)  # Fit the data

pickle.dump(cv, open('transform.pkl', 'wb'))  # Create a pickle file to be used in app.py, Flask framework

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
# Implement a multinomial algorithm
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
filename = 'nlp_model.pkl'  # Deploy this file in Heroku platform
pickle.dump(clf, open(filename, 'wb'))

# Alternative usage of Saved Model
joblib.dump(clf, 'NB_spam_model.pkl')
NB_spam_model = open('NB_spam_model.pkl', 'rb')
clf = joblib.load(NB_spam_model)

# y_pred = clf.predict(X_test)  # My test - First line
#
# from sklearn.metrics import accuracy_score   # My test - Sec line
# score = accuracy_score(y_test, y_pred)   # My test - Third line
# print(score)  # My test - Third line
# print(X)  # My test - Last line



