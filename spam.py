import math
import pandas as pd
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style='white')

features = [
	    'word_freq_make',
	    'word_freq_address',
	    'word_freq_all',
	    'word_freq_3d',
	    'word_freq_our',
	    'word_freq_over',
	    'word_freq_remove',
	    'word_freq_internet',
	    'word_freq_order',
	    'word_freq_mail',
	    'word_freq_receive',
	    'word_freq_will',
	    'word_freq_people',
	    'word_freq_report',
	    'word_freq_addresses',
	    'word_freq_free',
	    'word_freq_business',
	    'word_freq_email',
	    'word_freq_you',
	    'word_freq_credit',
	    'word_freq_your',
	    'word_freq_font',
	    'word_freq_000',
	    'word_freq_money',
	    'word_freq_hp',
	    'word_freq_hpl',
	    'word_freq_george',
	    'word_freq_650',
	    'word_freq_lab',
	    'word_freq_labs',
	    'word_freq_telnet',
	    'word_freq_857',
	    'word_freq_data',
	    'word_freq_415',
	    'word_freq_85',
	    'word_freq_technology',
	    'word_freq_1999',
	    'word_freq_parts',
	    'word_freq_pm',
	    'word_freq_direct',
	    'word_freq_cs',
	    'word_freq_meeting',
	    'word_freq_original',
	    'word_freq_project',
	    'word_freq_re',
	    'word_freq_edu',
	    'word_freq_table',
	    'word_freq_conference',
	    'char_freq_;',
	    'char_freq_(',
	    'char_freq_[',
	    'char_freq_!',
	    'char_freq_$',
	    'char_freq_#',
	    'capital_run_length_average',
	    'capital_run_length_longest',
	    'capital_run_length_total',
	    'is_spam'
	]

def logistic_function(b0, b1, x):
    '''
    given the coefficients of a logit_model
    calculate y_hat for an x value
    '''
    odds = math.exp(b0 + b1 * x)
    f = odds / (1 + odds)
    return f

dat = pd.read_csv('spambase.data', names=features)

ys = dat.pop('is_spam')
xs = sm.add_constant(dat)

X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.4)

logit_model = sm.Logit(Y_train, X_train)
bayes_model = BernoulliNB()

logit_fit = logit_model.fit()
bayes_fit = bayes_model.fit(X_train, Y_train)

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
bayes_predicts = [1 if x > 0.5 else 0 for x in bayes_fit.predict(X_test)]

print("Logit Confusion Matrix\n%s\n" % confusion_matrix(Y_test, logit_predicts))
print("Logit Classification Report\n%s\n" % classification_report(Y_test, logit_predicts))

print("Bayes Confusion Matrix\n%s\n" % confusion_matrix(Y_test, bayes_predicts))
print("Bayes Classification Report\n%s\n" % classification_report(Y_test, bayes_predicts))

f = plt.figure()
ax = f.add_subplot(111)
ax.scatter(logit_fit.predict(), logit_fit.resid_response)
ax.set_xlabel('predictions')
ax.set_ylabel('residuals')
plt.show()

########## Plot decision boundary between two features ###########

features = ['word_freq_000', 'word_freq_george']

xs = np.array(dat[features])
X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.4)

logit_model = LogisticRegression()
logit_fit = logit_model.fit(X_train, Y_train)

x_min, x_max = xs[:, 0].min() - .5, xs[:, 0].max() + .5
y_min, y_max = xs[:, 1].min() - .5, xs[:, 1].max() + .5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))

f, ax = plt.subplots(figsize=(12, 10))

Z = logit_fit.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=.5)
cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=.5)

ax_c = f.colorbar(cs)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)

ax.plot(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], 'ro', label=features[0])
ax.plot(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], 'bo', label=features[1])

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
plt.show()
