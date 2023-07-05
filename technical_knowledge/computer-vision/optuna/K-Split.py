from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from sklearn.model_selection import cross_validate

from sklearn import datasets
cancer = datasets.load_breast_cancer()

X = cancer.data
y = 1- cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=123)

model = LogisticRegression(solver='lbfgs',max_iter=10000)

# 訓練
model.fit(X_train, y_train)

# 予測
pred = model.predict(X_test)

# 正解率を出力
accuracy = 100.0 * accuracy_score(y_test, pred)
print("正解率: {}".format(accuracy))