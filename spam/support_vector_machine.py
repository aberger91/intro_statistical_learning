import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from spam_feature_labels import features

def prepare_svm_data():
    '''
    X_train, X_test, Y_train, Y_test = prepare_bayesian_data()
    '''
    scaler = StandardScaler()

    xs = pd.read_csv('../data/spambase.data', names=features)
    ys = xs.pop('is_spam')

    X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.4)

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test


def support_vector_spam_filter():
    X_train, X_test, Y_train, Y_test = prepare_svm_data()
    svm_model = svm.SVC()

    svm_fit = svm_model.fit(X_train, Y_train)

    svm_predicts = [1 if x > 0.5 else 0 for x in svm_fit.predict(X_test)]
    return (svm_fit, Y_test, svm_predicts)

if __name__ == '__main__':
    support_vector_spam_filter()
