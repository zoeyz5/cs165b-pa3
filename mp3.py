# Starter code for CS 165B MP3
import random
import nltk
import pandas as pd
import json
from sklearn import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def clean(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]

    clean_text = " ".join(stemmed_tokens)
    return clean_text

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2))

#mutinomial naive bayes
from sklearn.naive_bayes import MultinomialNB
mn = MultinomialNB()

#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=20, random_state=0)

def run_train_test(training_data, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[{"text": the utterance,\
                             "label": the label, can be 0(negative), 1(neutral),or 2(positive),\
                             "speaker": the name of the speaker,\
                             "dialogue_id": the id of dialogue,\
                             "utterance_id": the id of the utterance}]
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    train_feature = [_.pop('text') for _ in training_data]
    train_label = [_.pop('label') for _ in training_data]
    test_feature = [_.pop('text') for _ in testing_data]

    x_clean = [clean(i) for i in train_feature]
    xt_clean = [clean(i) for i in test_feature]

    x_vec = cv.fit_transform(x_clean).toarray()
    xt_vec = cv.transform(xt_clean).toarray()

    rf.fit(x_vec, train_label)
    text_predict = rf.predict(xt_vec)

    return text_predict

    #TODO implement your model and return the prediction
