import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from spam import features


def bayesian_spam_filter():
    dat = pd.read_csv('../data/spambase.data', names=features)
    dat = dat.replace('?', np.nan).dropna()

    ys = dat.pop('is_spam')

    X_train, X_test, Y_train, Y_test = train_test_split(dat, ys, test_size=0.4)

    model = GaussianNB()
    fit = model.fit(X_train, Y_train)

    predictions = fit.predict(X_test)

    print(confusion_matrix(Y_test, predictions))
    print(accuracy_score(Y_test, predictions))


if __name__ == '__main__':
    bayesian_spam_filter()
