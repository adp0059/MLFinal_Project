import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

"""
baselineNB: function for Naive Bayes Baseline
:param x_train: training data
:param y_train: training labels
:param x_test: testing data
:param y_test: testing labels
:param dist: set distribution
             0: Bernoulli
             1: Gaussian
             2: Multinomial
"""


def baselineNB(x_train, y_train, x_test, y_test, dist):
    # Generate Model

    if dist == 0:
        model = BernoulliNB()
        print("Running Bernoulli Naive Bayes")
    elif dist == 1:
        model = GaussianNB()
        print("Running Gaussian Naive Bayes")
    else:
        model = MultinomialNB()
        print("Running Multinomial Naive Bayes")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_acc = (metrics.accuracy_score(y_test, y_pred)) * 100
    print("NB Accuracy:", format(train_acc, '.3f'))

    # Report Test Accuracy, Classification Report, and Confusion Matrix

    print("\nNB Classification Report")
    report = metrics.classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_pred, y_test)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    if dist == 0:
        plt.savefig("Matrix/Confusion_matrix_NB_B")
    elif dist == 1:
        plt.savefig("Matrix/Confusion_matrix_NB_G")
    else:
        plt.savefig("Matrix/Confusion_matrix_NB_M")
    print("NB Confusion Matrix saved\n")

    return ()

def baselineDT(x_train, y_train, x_test, y_test):
    print("Running Decision Tree")

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=7)

    # Train Decision Tree Classifer
    clf = clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)

    # Model Accuracy, how often is the classifier correct?
    train_acc = metrics.accuracy_score(y_test, y_pred) * 100
    print("DT Accuracy:", format(train_acc, ".3f"))

    print("\nDT Classification Report")
    report = metrics.classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_pred, y_test)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("Matrix/Confusion_matrix_DT")

    print("DT Confusion Matrix saved")

    return()
