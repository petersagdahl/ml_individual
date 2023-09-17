import numpy as np
import pandas as pd

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:

    def __init__(self):
        self.weights = []

    def fit(self, xe, ye, learning_rate, num_iters):

        y = np.array(ye)[:, np.newaxis]
        x = np.array(xe)
        X = self.feature_engineering(x[:, 0], x[:, 1])

        weights = np.array(X).shape[1]
        weights = np.full((weights, 1), 0, dtype=np.float64)

        weights = self.adjust_parameters(
            X, y, weights, learning_rate, num_iters)
        self.weights = weights

    def feature_engineering(self, column_1, column_2):
        num_samples = len(column_1)
        # Initialize the feature matrix with a column of ones for the bias term.
        features = np.ones((num_samples, 1))

        # Add features that by trail and error seems to fit the datasets quite well. As well as the original features.
        new_feature = (column_2)
        new_feature = new_feature[:, np.newaxis]
        features = np.append(features, new_feature, axis=1)

        new_feature = column_1
        new_feature = new_feature[:, np.newaxis]
        features = np.append(features, new_feature, axis=1)

        new_feature = column_1**2/2**2 + column_2**2
        new_feature = new_feature[:, np.newaxis]
        features = np.append(features, new_feature, axis=1)

        return features

    def predict(self, xe):

        x = np.array(xe)
        X = self.feature_engineering(x[:, 0], x[:, 1])

        return (sigmoid(np.dot(X, self.weights)) > 0.5).ravel()

    def adjust_parameters(self, x, y, weights, learning_rate, num_iters):

        for i in range(num_iters):
            # Calculate predictions
            predictions = sigmoid(np.dot(x, weights))

            # Calculate the error (how wrong the predictions are)
            error = predictions - y

            # Calculate how much we need to adjust the parameters (weights)
            gradient = np.dot(x.T, error) / len(y)
            weights -= learning_rate * gradient

        return weights

# --- Some utility functions


def binary_accuracy(y_true, y_pred, threshold=0.5):

    assert np.array(y_true).shape == np.array(y_pred).shape
    y_pred_thresholded = (np.array(y_pred) >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):

    assert np.array(y_true).shape == np.array(y_pred).shape
    y_pred = np.clip(np.array(y_pred), eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        np.array(y_true) * np.log(np.array(y_pred)) +
        (1 - np.array(y_true)) * (np.log(1 - np.array(y_pred)))
    )


def sigmoid(x):

    return 1. / (1. + np.exp(-x))
