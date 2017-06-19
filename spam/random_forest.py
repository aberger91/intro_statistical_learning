from sklearn.ensemble import RandomForestClassifier 
from prepare_spambase_data import standardized_split

def random_forest_spam_filter():
    X_train, X_test, Y_train, Y_test = standardized_split()

    # 57 -> number of features
    rf = RandomForestClassifier(n_estimators=100, max_features=57)

    fit = rf.fit(X_train, Y_train)
    random_forest_predicts = rf.predict(X_test)

    return (fit, Y_test, random_forest_predicts)


if __name__ == '__main__':
    random_forest_spam_filter()
