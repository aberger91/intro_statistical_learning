from sklearn.neural_network import MLPClassifier
from spam_feature_labels import features
from prepare_spambase_data import standardized_split

def neural_network_spam_filter():
    X_train, X_test, Y_train, Y_test = standardized_split()

    # 57 -> number of features
    mlp = MLPClassifier(hidden_layer_sizes=(57, 57, 57), max_iter=1000)

    fit = mlp.fit(X_train, Y_train)
    mlp_predicts = mlp.predict(X_test)

    return (fit, Y_test, mlp_predicts)


if __name__ == '__main__':
    neural_network_spam_filter()
