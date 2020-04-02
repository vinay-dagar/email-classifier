import pandas as pd
import numpy as np
import pickle

df = pd.read_table('SMSSpamCollection', header=None, encoding='utf-8', names=["label", "message"])
import spacy
nlp = spacy.load('en_core_web_sm')

stop_words = nlp.Defaults.stop_words

import re
import nltk
ps = nltk.PorterStemmer()
stop_words = nlp.Defaults.stop_words

corpus = []
corpus = df['message'].apply(lambda x: ' '.join(ps.stem(term) for term in x.split() if not term in stop_words))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = TfidfVectorizer(min_df=1,stop_words='english')
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# pickle.dump(spam_detect_model, open('spam_classifier.pkl','wb'))

# model = pickle.load(open('spam_classifier.pkl','rb'))
# print('modal trained')