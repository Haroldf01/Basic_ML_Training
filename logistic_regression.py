# class 6th - octor

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , digits.data.shape)

# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)

# figsize = width and height in inches
plt.figure(figsize=(20, 4))

for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# modeling pattern
# model = LogisticRegression()
model = LogisticRegression(C=50.0, penalty="l1", solver="saga", tol=0.1)
model.fit(x_train, y_train)

# Returns a NumPy Array
# Predict for One Observation (image)
one_pred = model.predict(x_test[6].reshape(1,-1))
print(f'one_pred - {one_pred}')

all_pred = model.predict(x_test)
print(f'all_pred - {all_pred}')

# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
