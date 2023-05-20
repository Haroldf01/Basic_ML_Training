# class 9th = 2022, 1st Jan
# class 10th = 2022, 15th Jan

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv('./data/Iris.csv')
df = df.drop(['Id'], axis=1)

# output
target = df['Species']

s = set()
for val in target:
    s.add(val)
s = list(s)

# Removing one data point to make it binary classifier
rows = list(range(100, 150))
df = df.drop(df.index[rows])

# Reducing feature space from 4 to 2
# i.e Sepal length and Petal length
x = df['SepalLengthCm']
y = df['PetalLengthCm']

setosa_x = x[:50]
setosa_y = y[:50]

versicolor_x = x[50:]
versicolor_y = y[50:]

plt.figure(figsize=(6, 4))
plt.scatter(setosa_x, setosa_y, marker='+', color='green')
plt.scatter(versicolor_x, versicolor_y, marker='_', color='red')
plt.show()

# Drop rest of the features and extract the target values
df = df.drop(['SepalWidthCm', 'PetalWidthCm'], axis=1)
Y = []
target = df['Species']

for val in target:
    if(val == 'Iris-setosa'):
        Y.append(-1)
    else:
        Y.append(1)
df = df.drop(['Species'], axis=1)
X = df.values.tolist()

print('\ntypes of x\n{}\n'.format(type(X)))


# Shuffle and split the data into training and test set
X, Y = shuffle(X, Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(90, 1)
y_test = y_test.reshape(10, 1)


# Support Vector Machine
train_f1 = x_train[:, 0]
train_f2 = x_train[:, 1]

train_f1 = train_f1.reshape(90, 1)
train_f2 = train_f2.reshape(90, 1)

w1 = np.zeros((90, 1))
w2 = np.zeros((90, 1))

epochs = 1

# learning rate
alpha = 0.0001

print('training started')

while(epochs < 1000):
    y = w1 * train_f1 + w2 * train_f2
    prod = y * y_train
    count = 0
    for val in prod:
        if(val >= 1):
            cost = 0
            w1 = w1 - alpha * (2 * 1/epochs * w1)
            w2 = w2 - alpha * (2 * 1/epochs * w2)

        else:
            cost = 1 - val
            w1 = w1 + alpha * \
                (train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
            w2 = w2 + alpha * \
                (train_f2[count] * y_train[count] - 2 * 1/epochs * w2)
        count += 1
    epochs += 1


# Clip the weights
index = list(range(10, 90))
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

w1 = w1.reshape(10, 1)
w2 = w2.reshape(10, 1)

# Extract the test data features
test_f1 = x_test[:, 0]
test_f2 = x_test[:, 1]

test_f1 = test_f1.reshape(10, 1)
test_f2 = test_f2.reshape(10, 1)


# Predict
y_pred = w1 * test_f1 + w2 * test_f2
predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

accuracy = accuracy_score(y_test, predictions)
print('Accuracy {}'.format(accuracy))


##########################
## implement SVM using scikit-learn
##########################

# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# clf = SVC(kernel='linear')
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print(accuracy_score(y_test,y_pred))

# https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
