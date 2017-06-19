from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from spam import features

def neural_network_spam_filter():
    scaler = StandardScaler()

    xs = pd.read_csv('../data/spambase.data', names=features)
    ys = xs.pop('is_spam')

    X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.4)

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(57, 57, 57), max_iter=1000)

    fit = mlp.fit(X_train, Y_train)
    mlp_predicts = mlp.predict(X_test)

    print(confusion_matrix(Y_test, mlp_predicts))
    print(classification_report(Y_test, mlp_predicts))


if __name__ == '__main__':
    neural_network_spam_filter()
