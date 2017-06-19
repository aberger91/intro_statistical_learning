'''
#Spam Filtering with Classifiers
  - ordinary least squares regression
  - logistic regression
  - naive bayesian classifiers
  - multilayer perceptron neural network

# Attribute Information
  - 48 continuous real [0,100] attributes of type word_freq_WORD
  - = percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.

  - 6 continuous real [0,100] attributes of type char_freq_CHAR]
  - = percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail

  - 1 continuous real [1,...] attribute of type capital_run_length_average
  - = average length of uninterrupted sequences of capital letters

  - 1 continuous integer [1,...] attribute of type capital_run_length_longest
  - = length of longest uninterrupted sequence of capital letters

  - 1 continuous integer [1,...] attribute of type capital_run_length_total
  - = sum of length of uninterrupted sequences of capital letters
  - = total number of capital letters in the e-mail

  - 1 nominal {0,1} class attribute of type spam
  - = denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail. 

*References*:
  - https://archive.ics.uci.edu/ml/datasets/Spambase
  - Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt
    Hewlett-Packard Labs, 1501 Page Mill Rd., Palo Alto, CA 94304
'''
from support_vector_machine import support_vector_spam_filter
from prepare_spambase_data import standardized_split, split
from sklearn.metrics import (
                            confusion_matrix, 
                            classification_report, 
                            accuracy_score
                            ) 


class SpamBaseModels:
    def random_forest_spam_filter():
        X_train, X_test, Y_train, Y_test = standardized_split()

        # 57 -> number of features
        rf = RandomForestClassifier(n_estimators=100, max_features=57)

        fit = rf.fit(X_train, Y_train)
        random_forest_predicts = rf.predict(X_test)

        return (fit, Y_test, random_forest_predicts)

    def k_nearest_neighbors_spam_filter(n_neighbors=1):
        X_train, X_test, Y_train, Y_test = standardized_split()

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        fit = model.fit(X_train, Y_train)

        #  optimize_k(X_train, Y_train)
        predictions = fit.predict(X_test)
        return (fit, Y_test, predictions)

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

    def neural_network_spam_filter():
        X_train, X_test, Y_train, Y_test = standardized_split()

        # 57 -> number of features
        mlp = MLPClassifier(hidden_layer_sizes=(57, 57, 57), max_iter=1000)

        fit = mlp.fit(X_train, Y_train)
        mlp_predicts = mlp.predict(X_test)

        return (fit, Y_test, mlp_predicts)

    def logit_spam_filter():
        X_train, X_test, Y_train, Y_test = split()
        logit_model = sm.Logit(Y_train, X_train)

        logit_fit = logit_model.fit()

        logit_predicts = [1 if x > 0.5 else 0 for x in logit_fit.predict(X_test)]
        return (logit_fit, Y_test, logit_predicts)


if __name__ == '__main__':
    '''
    TODO reduce false positives
    '''
    spambase_models = SpamBaseModels()
    ML_functions = [ 
                    spambase_models.gaussian_bayes_spam_filter,
                    spambase_models.bernoulli_bayes_spam_filter,
                    spambase_models.logit_spam_filter,
                    spambase_models.neural_network_spam_filter,
                    spambase_models.support_vector_spam_filter,
                    spambase_models.k_nearest_neighbors_spam_filter,
                    spambase_models.random_forest_spam_filter
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
