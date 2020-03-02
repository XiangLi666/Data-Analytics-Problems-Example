# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:39:24 2019

@author: Li Xiang
"""
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import sentiwordnet as swn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import re
import os
import time
import argparse


def prepare_training_data():
    """prepare the training data"""
    """get the raw text&label combinations list"""
    print("start training, the first run will take several minutes")
    documents_label = [
            (" ".join([w for w in movie_reviews.words(fileid)
                      if w.isalpha()]), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)
            ]
    documents = [document for (document, label) in documents_label]
    dump(documents, 'documents.jbl')
    labels = [label for (document, label) in documents_label]
    labels_array = np.array(labels).reshape(len(labels), 1)
    dump(labels_array, 'labels_array.jbl')

    """get the text with the sentiment, the label vector would be the same as
    the original one"""
    senti_documents = documents[:]
    for i in range(len(senti_documents)):
        senti_documents[i] = [word for word in senti_documents[i].split()
                              if list(swn.senti_synsets(word))]
        senti_documents[i] = " ".join([
                word for word in senti_documents[i]
                if list(swn.senti_synsets(word))[0].pos_score() > 0.5
                or list(swn.senti_synsets(word))[0].neg_score() > 0.5
                ])
    dump(senti_documents, 'senti_documents.jbl')

    """get the text with only the words in MPQA"""
    with open(
        './data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff',
        'r'
    )as f:
        words_lines = f.read()
    mpqa_words = re.findall(r'word1=(\w+)', words_lines)
    mpqa_documents = documents[:]
    for i in range(len(mpqa_documents)):
        mpqa_documents[i] = " ".join([
                word for word in mpqa_documents[i].split()
                if word in mpqa_words
                ])
    dump(mpqa_documents, 'mpqa_documents.jbl')

    """replace the negation part a text with a single word"""
    neg_documents = documents[:]
    for i in range(len(neg_documents)):
        neg_words = re.findall(r'not\s\w+', neg_documents[i])
        for j in range(len(neg_words)):
            neg_words[j] = re.sub(r'\s', '_', neg_words[j])
        neg_documents[i] = re.sub(r'not\s\w+', '', neg_documents[i])
        neg_documents[i] = neg_documents[i]+' '+" ".join(neg_words)
    dump(neg_documents, 'neg_documents.jbl')


def get_training_data():
    """get the training data"""
    documents = load('documents.jbl')
    labels_array = load('labels_array.jbl')
    senti_documents = load('senti_documents.jbl')
    mpqa_documents = load('mpqa_documents.jbl')
    neg_documents = load('neg_documents.jbl')
    return documents, labels_array, senti_documents, mpqa_documents, neg_documents


"""clean the classifier folder"""
try:
    os.mkdir('./classifiers')
except FileExistsError:
    pass
"""get the training data from local directory"""
try:
    documents, labels_array, senti_documents, mpqa_documents, neg_documents = get_training_data()
except FileNotFoundError:
    prepare_training_data()
    documents, labels_array, senti_documents, mpqa_documents, neg_documents = get_training_data()


def train_model1_NB(doc=documents, lab=labels_array):
    """raw count with naive bayes"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    feature1_matrix = vectorizer.transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/bayes-all-words-raw-counts.jbl')
    print("""
    Creating Bayes classifier in classifiers/bayes-all-words-raw-counts.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model1_DT(doc=documents, lab=labels_array):
    """raw count with decision tree"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/tree-all-words-raw-counts.jbl')
    print("""
    Creating Tree classifier in classifiers/tree-all-words-raw-counts.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model2_NB(doc=documents, lab=labels_array):
    """binary with naive bayes"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english", binary=True)
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/bayes-all-words-binary.jbl')
    print("""
    Creating Bayes classifier in classifiers/bayes-all-words-binary.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model2_DT(doc=documents, lab=labels_array):
    """binary with decision tree"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english", binary=True)
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/tree-all-words-binary.jbl')
    print("""
    Creating Tree classifier in classifiers/tree-all-words-binary.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model3_NB(doc=senti_documents, lab=labels_array):
    """sentiwordnet with naive bayes"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/bayes-SentiWordNet-words.jbl')
    print("""
    Creating Bayes classifier in classifiers/bayes-SentiWordNet-words.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model3_DT(doc=senti_documents, lab=labels_array):
    """sentiwordnet with decision tree"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/tree-SentiWordNet-words.jbl')
    print("""
    Creating Tree classifier in classifiers/tree-SentiWordNet-words.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model4_NB(doc=mpqa_documents, lab=labels_array):
    """mpqa words with naive bayes"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/bayes-Subjectivity-Lexicon-words.jbl')
    print("""
    Creating Bayes classifier in classifiers/bayes-Subjectivity-Lexicon-words.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model4_DT(doc=mpqa_documents, lab=labels_array):
    """mpqa words with decision tree"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/tree-Subjectivity-Lexicon-words.jbl')
    print("""
    Creating Tree classifier in classifiers/tree-Subjectivity-Lexicon-words.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model5_NB(doc=neg_documents, lab=labels_array):
    """all words plus negation with naive bayes"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/bayes-all-words-plus-Negation.jbl')
    print("""
    Creating Bayes classifier in classifiers/bayes-all-words-plus-Negation.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def train_model5_DT(doc=neg_documents, lab=labels_array):
    """all words plus negation with decision tree"""
    start_time = time.time()
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    feature1_matrix = vectorizer.fit_transform(doc)
    data = np.concatenate((feature1_matrix.toarray(), lab), axis=1)
    data = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    dump(clf, './classifiers/tree-all-words-plus-Negation.jbl')
    print("""
    Creating Tree classifier in classifiers/tree-all-words-plus-Negation.jbl""")
    print("    Elapsed time:%ss" % (time.time() - start_time))
    print("    Accuracy:%s" % accuracy_score(y_test, clf.predict(X_test)))


def use_model1_NB(text_path, doc=documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/bayes-all-words-raw-counts.jbl')
    print(model.predict(text_matrix)[0])


def use_model1_DT(text_path, doc=documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/tree-all-words-raw-counts.jbl')
    print(model.predict(text_matrix)[0])


def use_model2_NB(text_path, doc=documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english", binary=True)
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/bayes-all-words-binary.jbl')
    print(model.predict(text_matrix)[0])


def use_model2_DT(text_path, doc=documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english", binary=True)
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/tree-all-words-binary.jbl')
    print(model.predict(text_matrix)[0])


def use_model3_NB(text_path, doc=senti_documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/bayes-SentiWordNet-words.jbl')
    print(model.predict(text_matrix)[0])


def use_model3_DT(text_path, doc=senti_documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/tree-SentiWordNet-words.jbl')
    print(model.predict(text_matrix)[0])


def use_model4_NB(text_path, doc=mpqa_documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/bayes-Subjectivity-Lexicon-words.jbl')
    print(model.predict(text_matrix)[0])


def use_model4_DT(text_path, doc=mpqa_documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/tree-Subjectivity-Lexicon-words.jbl')
    print(model.predict(text_matrix)[0])


def use_model5_NB(text_path, doc=neg_documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/bayes-all-words-plus-Negation.jbl')
    print(model.predict(text_matrix)[0])


def use_model5_DT(text_path, doc=neg_documents):
    with open(text_path, "r") as f:
        text = [f.read()]
    vectorizer = CountVectorizer(max_features=2500, stop_words="english")
    vectorizer.fit(doc)
    text_matrix = vectorizer.transform(text).toarray()
    text_matrix = pd.DataFrame(text_matrix)
    model = load('./classifiers/tree-all-words-plus-Negation.jbl')
    print(model.predict(text_matrix)[0])


def return_result(model, index, text_path):
    """a function used to show the result.
    e.g.:
    >>>return_result("bayes","1","./data/emma.txt")
    >>>pos
    """
    if model == "bayes" and index == "1":
        use_model1_NB(text_path)
    if model == "tree" and index == "1":
        use_model1_DT(text_path)
    if model == "bayes" and index == "2":
        use_model2_NB(text_path)
    if model == "tree" and index == "2":
        use_model2_DT(text_path)
    if model == "bayes" and index == "3":
        use_model3_NB(text_path)
    if model == "tree" and index == "3":
        use_model3_DT(text_path)
    if model == "bayes" and index == "4":
        use_model4_NB(text_path)
    if model == "tree" and index == "4":
        use_model4_DT(text_path)
    if model == "bayes" and index == "5":
        use_model5_NB(text_path)
    if model == "tree" and index == "5":
        use_model5_DT(text_path)


"""Create the parser with required arguments."""
parser = argparse.ArgumentParser(
        description="Training and Using Sentiment Classifier.",
        prog='sentiment classifier', usage='%(prog)s [options]'
        )
parser.add_argument(
        "--train",
        dest="train",
        help="Train all classifiers.",
        action="store_true"
        )
parser.add_argument(
        "--run",
        dest="run",
        help="Test sentiment on a given file.",
        type=str,
        nargs="+"
        )
args = parser.parse_args()
"""Train all models"""
if args.train:
    train_model1_NB()
    train_model1_DT()
    train_model2_NB()
    train_model2_DT()
    train_model3_NB()
    train_model3_DT()
    train_model4_NB()
    train_model4_DT()
    train_model5_NB()
    train_model5_DT()

"""Test the sentiment of sentence"""
if args.run:
    path = "./" + args.run[1]
    if len(args.run) == 2:
        print("""Choose a model:
        1 - all words raw counts
        2 - all words binary
        3 - SentiWordNet words
        4 - Subjectivity Lexicon words
        5 - all words plus Negation""")
        model_index = input("Type a number:\n")
        return_result(args.run[0], model_index, path)
