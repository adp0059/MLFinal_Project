import csv
import random
import numpy as np
from baselinefn import *
from sklearn.model_selection import ShuffleSplit, train_test_split

# https://github.com/Meenapintu/Spam-Detection

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


def ReLu(x):
    return max(0, x)


def dReLu(y):
    if y < 0:
        return 0.0
    elif y > 0:
        return 1.0
    else:
        return "undefined"


class NN:
    def __init__(self, input_node_num, hidden_node_num, output_node_num):
        self.LR = 0.5;
        self.MF = 0.1;

        self.input_node_num = input_node_num
        self.hidden_node_num = hidden_node_num
        self.output_node_num = output_node_num

        self.inv = np.zeros(self.input_node_num)
        self.hnv = np.zeros(self.hidden_node_num)
        self.onv = np.zeros(self.output_node_num)

        self.wi = np.zeros((self.input_node_num, self.hidden_node_num))
        self.wo = np.zeros((self.hidden_node_num, self.output_node_num))
        self.lrci = np.zeros((self.input_node_num, self.hidden_node_num))
        self.lrco = np.zeros((self.hidden_node_num, self.output_node_num))

        for i in range(self.input_node_num):
            for j in range(self.hidden_node_num):
                self.wi[i][j] = rand(-1.0, 1.0)

        for j in range(self.hidden_node_num):
            for k in range(self.output_node_num):
                self.wo[j][k] = rand(-1.0, 1.0)
                # self.lrco[j][k] = rand(-1.0, 1.0)

    def update(self, inputs):
        if len(inputs) != self.input_node_num - 1:
            raise ValueError('error update')

        for i in range(self.input_node_num - 1):
            self.inv[i] = sigmoid(inputs[i])

        for j in range(self.hidden_node_num):
            sum = 0.0
            for i in range(self.input_node_num):
                sum = sum + self.inv[i] * self.wi[i][j]
            self.hnv[j] = sigmoid(sum)

        for k in range(self.output_node_num):
            sum = 0.0
            for j in range(self.hidden_node_num):
                sum = sum + self.hnv[j] * self.wo[j][k]
            self.onv[k] = sigmoid(sum)

        return self.onv[:]

    def update_out_weight(self, output_deltas):
        for j in range(self.hidden_node_num):
            for k in range(self.output_node_num):
                change = output_deltas[k] * self.hnv[j]
                self.wo[j][k] = self.wo[j][k] + self.LR * change + self.MF * self.lrco[j][k]
                self.lrco[j][k] = change

    def update_input_weight(self, hidden_deltas):
        for i in range(self.input_node_num):
            for j in range(self.hidden_node_num):
                change = hidden_deltas[j] * self.inv[i]
                self.wi[i][j] = self.wi[i][j] + self.LR * change + self.MF * self.lrci[i][j]
                self.lrci[i][j] = change

    def backPropagate(self, targets):
        if len(targets) != self.output_node_num:
            raise ValueError('error backPropagate')

        output_deltas = [0.0] * self.output_node_num
        for k in range(self.output_node_num):
            error = targets[k] - self.onv[k]
            output_deltas[k] = dsigmoid(self.onv[k]) * error

        hidden_deltas = [0.0] * self.hidden_node_num
        for j in range(self.hidden_node_num):
            error = 0.0
            for k in range(self.output_node_num):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.hnv[j]) * error

        self.update_out_weight(output_deltas)
        self.update_input_weight(hidden_deltas)

    def predict(self, x_test):
        predictions = []
        # out_file=open("output.csv",'wb')
        # writer=csv.writer(out_file, dialect='excel')
        # writer.writerow(['Id','Label',])
        for p in x_test:
            if self.update(p)[0] > .5:
                # writer.writerow([count,1])
                # print(1)
                predictions.append(1)
            else:
                # writer.writerow([count,0])
                # print(0)
                predictions.append(0)
        return predictions

    def fit(self, x_train, y_train, epochs=30):
        for _ in range(epochs):
            for j in range(len(x_train)):
                # inputs = p[0]
                # targets = p[1]
                self.update(x_train[j])
                self.backPropagate([y_train[j]])


def load_data(path):
    with open(path, newline="") as file:
        arr = list(csv.reader(file))
    data = np.array(arr)

    X = data[:, :-1]
    Y = data[:, -1]

    return X.astype(np.float16), Y.astype(int)


# https://github.com/Meenapintu/Spam-Detection
def run(x_train, x_test, y_train, y_test):
    print("Running Multilayered Perceptron")
    n = NN(58, 2, 1)

    n.fit(x_train, y_train)

    y_pred = n.predict(x_test)

    train_acc = metrics.accuracy_score(y_test, y_pred) * 100
    print("MLP Accuracy:", format(train_acc, ".3f"))

    print("\nMLP Classification Report")
    report = metrics.classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_pred, y_test)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("Matrix/Confusion_matrix_MLP")

    print("MLP Confusion Matrix saved\n")


# Data processing
path = "spambase/spambase.data"
data, labels = load_data(path)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

for _ in range(1):
    run(x_train, x_test, y_train, y_test)

# Baseline Functions
baselineNB(x_train, y_train, x_test, y_test, 0)
baselineNB(x_train, y_train, x_test, y_test, 2)
baselineDT(x_train, y_train, x_test, y_test)

