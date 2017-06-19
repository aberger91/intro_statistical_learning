from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.api import add_constant
from pandas import read_csv
from spam_feature_labels import features

def standardized_split():
    '''
    X_train, X_test, Y_train, Y_test = standardized_split()
    '''
    xs = read_csv('../data/spambase.data', names=features)
    ys = xs.pop('is_spam')
    X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.4)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def split():
    '''
    X_train, X_test, Y_train, Y_test = split()
    '''
    dat = read_csv('../data/spambase.data', names=features)
    ys = dat.pop('is_spam')
    xs = add_constant(dat)
    return train_test_split(xs, ys, test_size=0.4)
