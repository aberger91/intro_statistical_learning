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
from regression import logit_spam_filter
from bayesian import bernoulli_bayes_spam_filter, gaussian_bayes_spam_filter
from neural_net import neural_network_spam_filter
from support_vector_machine import support_vector_spam_filter
from k_nearest_neighbors import k_nearest_neighbors_spam_filter
from random_forest import random_forest_spam_filter
from sklearn.metrics import (
                            confusion_matrix, 
                            classification_report, 
                            accuracy_score
                            ) 

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
