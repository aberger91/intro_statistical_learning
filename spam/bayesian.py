from sklearn.naive_bayes import GaussianNB, BernoulliNB
from prepare_spambase_data import standardized_split
from spam_feature_labels import features

def bernoulli_bayes_spam_filter():
    X_train, X_test, Y_train, Y_test = standardized_split()

    model = BernoulliNB()
    fit = model.fit(X_train, Y_train)

    predictions = fit.predict(X_test)
    return (fit, Y_test, predictions)

def gaussian_bayes_spam_filter():
    X_train, X_test, Y_train, Y_test = standardized_split()

    model = GaussianNB()
    fit = model.fit(X_train, Y_train)

    predictions = fit.predict(X_test)
    return (fit, Y_test, predictions)


if __name__ == '__main__':
    bayesian_spam_filter()
