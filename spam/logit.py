import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from spam import features


def logit_spam_filter():
    dat = pd.read_csv('../data/spambase.data', names=features)

    ys = dat.pop('is_spam')
    xs = sm.add_constant(dat)

    X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.4)

    logit_model = sm.Logit(Y_train, X_train)

    logit_fit = logit_model.fit()

    #print(logit_fit.summary())
    #print('Logit Confidence Intervals\n%s\n' % logit_fit.conf_int())
    #
    #b0 = logit_fit.params[0] # intercept coeficent
    #
    #significant_values = logit_fit.pvalues.loc[logit_fit.pvalues < 0.0005]
    #coefficients = logit_fit.params.loc[significant_values.index]
    #odds_ratios = coefficients.apply(lambda x: 100 * math.exp(x) / (1 + math.exp(x)))
    #
    #print("Significant Values\n%s\n" % significant_values)
    #print("Coefficients\n%s\n" % coefficients)
    #print("Odds Ratios\n%s\n" % odds_ratios)

    logit_predicts = [1 if x > 0.5 else 0 for x in logit_fit.predict(X_test)]

    print("Logit Confusion Matrix\n%s\n" % confusion_matrix(Y_test, logit_predicts))
    print("Logit Classification Report\n%s\n" % classification_report(Y_test, logit_predicts))

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.scatter(logit_fit.predict(), logit_fit.resid_response)
    ax.set_xlabel('predictions')
    ax.set_ylabel('residuals')
    plt.show()


if __name__ == '__main__':
    logit_spam_filter()
