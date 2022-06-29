from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from IPython import embed
from sklearn.metrics import classification_report

"""
python eval_classical.py --trn_data ../data/sc-stance/L2c/sc-stance-neu-ner-train.csv --test_data ../data/sc-stance/L2/sc-stance-neu-test.csv --model LR
python eval_classical.py --trn_data ../data/sc-stance/L2/sc-stance-neu-train.csv --test_data ../data/sc-stance/L2/sc-stance-neu-test.csv --model LR


python eval_classical.py --trn_data ../data/sc-stance/L2/sc-stance-neu-train.csv --test_data ../data/sc-stance/L2/sc-stance-neu-dev.csv --model LR

"""

def get_features(texts, targets, vectorizer=None):
    corpus = []
    for i in range(len(texts)):
        corpus.append(targets[i] + ' -- ' + texts[i])
    X = None
    if vectorizer == None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
    else:
        X = vectorizer.transform(corpus)
    return X, vectorizer



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--trn_data', dest='trn_data', help='Name of the train data file', required=True)
    parser.add_argument('--test_data', dest='test_data', help='Name of the test data file', required=True)
    parser.add_argument('--model', dest='model',help='whats the model', required=False, default = 'LR') #False, default = torch.load('../models/vanilla_3.pt')
    args = parser.parse_args()


    df_train = pd.read_csv(args.trn_data)
    df_test = pd.read_csv(args.test_data)

    X_train, vectorizer = get_features(df_train['text'].values, df_train['target'].values)
    y_train = df_train['label'].values

    X_test, __ = get_features(df_test['text'].values, df_test['target'].values, vectorizer)
    y_test = df_test['label'].values


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if args.model == 'LR':
        clf = LogisticRegression(random_state=0)
    else:
        clf = DummyClassifier(strategy="most_frequent")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

main()