import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
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
             1: Multinomial
"""


def baselineNB(x_train, y_train, x_test, y_test, dist):
    # Generate Model

    if dist == 0:
        model = BernoulliNB()
        print("Running Bernoulli Naive Bayes")
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

    plot_Conf(y_pred, y_test)
    if dist == 0:
        plt.savefig("Matrix/Confusion_matrix_NB_B")
    else:
        plt.savefig("Matrix/Confusion_matrix_NB_M")
    print("NB Confusion Matrix saved")

    plot_ROC(model, x_test, y_test)
    if dist == 0:
        plt.savefig("ROC/ROC_NB_B")
    else:
        plt.savefig("ROC/ROC_NB_M")
    print("NB ROC Curve saved\n")

    return ()

"""
baselineDT: function for Decision Tree Baseline
:param x_train: training data
:param y_train: training labels
:param x_test: testing data
:param y_test: testing labels
"""
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

    plot_Conf(y_pred, y_test)
    plt.savefig("Matrix/Confusion_matrix_NB_B")
    print("NB Confusion Matrix saved")

    plot_ROC(clf, x_test, y_test)
    plt.savefig("ROC/ROC_DT")
    print("DT ROC Curve saved\n")

    return()

def plot_ROC(model, x_test, y_test):
    plt.clf()
    # define metrics
    y_pred_proba = model.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC Curve")
    plt.legend(loc=4)
    return ()

def plot_Conf(y_pred, y_test):
    plt.clf()
    cm = confusion_matrix(y_pred, y_test)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    return ()
