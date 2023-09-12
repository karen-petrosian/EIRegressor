# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from .dsgd import DSClassifierMultiQ
from .dsgd import DSRule
from .utils import replace_nan_median, bucketing

class EIRegressor:
    """
    Implementation of Embedded interpreter regression based on DS model
    """

    def __init__(self, regressor, n_buckets=3, bucketing_method="quantile",
                 reg_args={}, **cla_kwargs):
        """
        :param regressor: Regressor to be used for each bucket
        :param n_buckets: int, the amount of evenly-populated generated buckets of regressors
        :param bucketing_method: string, method to separate the target in different categories (ranged/quantile/max_score)
        :param reg_args: dict, Arguments for the regressor
        :param cla_kwargs: dict,  keyword arguments for the classifier
        """
        self.n_buckets = n_buckets
        self.bins = []
        self.bucketing_method = bucketing_method
        self.y_dtype = None
        self.training_medians = None
        self.classifier = DSClassifierMultiQ(
            num_classes=n_buckets, **cla_kwargs)
        self.regressors = [regressor(
            **reg_args) for i in range(n_buckets)]

    def fit(self, X_train, y_train, reg_args={}, **cla_kwargs):
        """
        Fits the model using the training data
        :param X: Features for training
        :param y: Labels of features
        :param reg_args: Arguments for the regressor fitting
        :param cla_kwargs: Arguments for the DS classifier fitting
        """

        self.y_dtype = y_train.dtype
        if self.bins == []:
            (buckets, bins) = bucketing(
                y_train, bins=self.n_buckets, type=self.bucketing_method)
            self.bins = bins  # To test classifier later
        else:
            buckets = pd.cut(y_train, self.bins)
        self.classifier.fit(X_train, buckets, **cla_kwargs)
        pred_bucket = self.classifier.predict(X_train)
        self.training_medians = replace_nan_median(X_train)
        for i in range(self.n_buckets):
            bucket_X = X_train[pred_bucket == i]
            bucket_y = y_train[pred_bucket == i]
            if len(bucket_X) == 0:
                bucket_X = X_train[buckets == i]
                bucket_y = y_train[buckets == i]
            self.regressors[i].fit(bucket_X, bucket_y, **reg_args)

    def predict(self, X_test, return_buckets=False):
        """
        Predict the classes for the feature vectors
        :param X: Feature vectors
        :param return_buckets: If true, it return buckets assigned to data
        :return: Value predicted for each feature vector. If return_buckets is true, it returns the buckets assigned to data
        """
        buck_pred = self.classifier.predict(X_test)
        y_pred = np.zeros(buck_pred.shape, dtype=self.y_dtype)

        replace_nan_median(X_test, self.training_medians)
        for i in range(self.n_buckets):
            if not (buck_pred == i).any():
                continue
            y_pred[buck_pred == i] = self.regressors[i].predict(
                X_test[buck_pred == i])
        if return_buckets:
            return buck_pred, y_pred
        return y_pred

    def get_bins(self):
        """
        Returns the bins used for bucketing the data
        """
        return self.bins

    def set_bins(self, bins):
        """
        Sets the bins used for bucketing the data
        :param bins: Array of bins
        """
        self.bins = bins

    def predict_proba(self, X):
        """
        Predict the score of belogning to all classes
        :param X: Feature vector
        :return: Class scores for each feature vector
        """
        return self.classifier.predict_proba(X)

    def predict_explain(self, X):
        """
        Predict the score of belogning to each class and give an explanation of that decision
        :param x: A single Feature vectors
        :return: Class scores for each feature vector and a explanation of the decision
        """
        return self.classifier.predict_explain(X)

    def add_rule(self, rule, caption="", m_sing=None, m_uncert=None):
        """
        Adds a rule to the model. If no masses are provided, random masses will be used.
        :param rule: lambda or callable, used as the predicate of the rule
        :param caption: Description of the rule
        :param m_sing: [optional] masses for singletons
        :param m_uncert: [optional] mass for uncertainty
        """
        self.classifier.model.add_rule(DSRule(rule, caption), m_sing, m_uncert)

    def find_most_important_rules(self, classes=None, threshold=0.2):
        """
        Shows the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return: A list containing the information about most important rules
        """
        return self.classifier.model.find_most_important_rules(classes=classes, threshold=threshold)

    def print_most_important_rules(self, classes=None, threshold=0.2):
        """
        Prints the most contributive rules for the classes specified
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :return:
        """
        self.classifier.model.print_most_important_rules(
            classes=classes, threshold=threshold)

    def evaluate_classifier(self, X_test, y_test):
        """
            Evaluates the classifier using the test data
            :param X_test: Features for test
            :param y_test: Labels of features
            :return: F1 score macro and accuracy score
        """
        bins = []
        if len(self.bins) == 3:
            bins = np.append(int(min(y_test)-1), np.append(self.bins[1], int(max(y_test)+1)))
        else:
            bins = np.append(
                min(y_test)-1, np.append(self.bins[1:-1], max(y_test)+1))
        y_test = pd.cut(y_test, bins, labels=False)
        y_pred = self.classifier.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
        return f1_macro, acc

    def rules_to_txt(self, filename, classes=None, threshold=0.2, results={}):
        """
        Write the most contributive rules for the classes specified in an output file
        :param filename: Output file name
        :param classes: Array of classes, by default shows all clases
        :param threshold: score minimum value considered to be contributive
        :param results: Dictionary with the results to print in txt 
        :return:
        """
        rules = self.classifier.model.find_most_important_rules(
            classes=classes, threshold=threshold)
        with open(filename, 'w') as file:
            for r in results:
                file.write(r + ": " + str(results[r]) + "\n\n\n")
            file.write(f"Most important rules\n-----------------------------\n")
            for key, rules_list in rules.items():
                file.write(f"\n---{key}---\n")
                for rule in rules_list:
                    file.write(
                        f"rule{rule [1]}: {rule[2]}\nprobabilities_array:{rule[4]}\n\n")
