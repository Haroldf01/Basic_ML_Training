import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/titanic_train_clean.csv')

def decision_tree():
    # drop Name feature
    df.drop(['Name', 'Ticket', 'Title', 'PassengerId'], axis=1, inplace=True)

    gender_map = {
        'male': 0,
        'female': 1
    }

    df['Sex'] = df['Sex'].map(gender_map)

    X = df.drop(columns=['Survived', 'Cabin', 'Embarked'])
    Y = df['Survived'].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, Y_train)
    dt_pred = dt.predict(X_train)

    # F1-Score is the harmonic mean of precision and recall given by the formula:
    print('Training Set Evaluation F1-Score=>', f1_score(Y_train, dt_pred))

    # Evaluating on Test set
    dt_pred_test = dt.predict(X_test)
    print('Testing Set Evaluation F1-Score=>', f1_score(Y_test, dt_pred_test))
