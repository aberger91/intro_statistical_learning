'''
#Spam Filtering with Classifiers
  - ordinary least squares regression
  - logistic regression
  - naive bayesian classifiers
  - multilayer perceptron neural network

*References*:
  - https://archive.ics.uci.edu/ml/datasets/Spambase
  - Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt
    Hewlett-Packard Labs, 1501 Page Mill Rd., Palo Alto, CA 94304
'''

from prepare_spambase_data import (
                                   standardized_train_test_split, 
                                   train_test_split_data
                                   )  # X_train, X_test, Y_train, Y_test
from sklearn.metrics import (
                            confusion_matrix, 
                            classification_report, 
                            accuracy_score
                            ) 
from support_vector_machine import support_vector_spam_filter
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from statsmodels.api import Logit

def get_predictions(model):
    X_train, X_test, Y_train, Y_test = standardized_train_test_split()
    fit = model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return (fit, Y_test, predictions)

def random_forest_spam_filter():
    model = RandomForestClassifier(n_estimators=100, max_features=57)
    predicts = get_predictions(model)
    return predicts

def k_nearest_neighbors_spam_filter(n_neighbors=1):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    predicts = get_predictions(model)
    return predicts

def bernoulli_bayes_spam_filter():
    model = BernoulliNB()
    predicts = get_predictions(model)
    return predicts

def gaussian_bayes_spam_filter():
    model = GaussianNB()
    predicts = get_predictions(model)
    return predicts

def neural_network_spam_filter():
    model = MLPClassifier(hidden_layer_sizes=(57, 57, 57), max_iter=1000)
    predicts = get_predictions(model)
    return predicts

def logit_spam_filter():
    X_train, X_test, Y_train, Y_test = train_test_split_data()
    logit_model = Logit(Y_train, X_train)
    logit_fit = logit_model.fit()
    logit_predicts = [1 if x > 0.5 else 0 for x in logit_fit.predict(X_test)]
    return (logit_fit, Y_test, logit_predicts)


if __name__ == '__main__':
    '''
    TODO reduce false positives
    '''
    ML_functions = [ 
                    gaussian_bayes_spam_filter,
                    bernoulli_bayes_spam_filter,
                    logit_spam_filter,
                    neural_network_spam_filter,
                    support_vector_spam_filter,
                    k_nearest_neighbors_spam_filter,
                    random_forest_spam_filter
                   ] 

    #  functions for model diagnostics
    metrics = (confusion_matrix, classification_report, accuracy_score)
    
    for function in ML_functions:
        print('=' * 56)
        model = function.__name__
        print('%s model summary' % model)
        
        fitted_model, y_true, y_hats = function()

        for metric in metrics:
            print(' %s %s ' % (model, metric.__name__))
            print(metric(y_true, y_hats))
