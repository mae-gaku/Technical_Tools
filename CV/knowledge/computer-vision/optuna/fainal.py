from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

import numpy as np
import optuna
from sklearn import datasets
cancer = datasets.load_breast_cancer()

X = cancer.data
y = 1- cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=123)


class Objective:
    def __init__(self, X, y):
        # 変数X,yの初期化
        self.X = X
        self.y = y

    def __call__(self, trial):
        # ハイパーパラメータの設定
        params = {
            # 最適化アルゴリズムを指定
            'solver' : trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            # 正則化の強さを指定（0.0001から10まで）
            'C': trial.suggest_loguniform('C', 0.0001, 10),
            # 最大反復回数（＊ソルバーが収束するまで）
            'max_iter': trial.suggest_int('max_iter', 100, 100000)
            }

        model = LogisticRegression(**params)

        # 評価指標として正解率の最大化を目指す
        scores = cross_validate(model,
                                X=self.X, 
                                y=self.y,
                                scoring='accuracy', # 正解率を指定（https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter）
                                n_jobs=-1) # 並行して実行するジョブの数（-1は全てのプロセッサを使用）
        return scores['test_score'].mean()

# ハイパーパラメータの探索
objective = Objective(X_train, y_train)
study = optuna.create_study(direction='maximize') # 最大化
study.optimize(objective, timeout=60)

# ベストパラメータを出力
print('params:', study.best_params)
print(study.best_value)

model = LogisticRegression(
    # ハイパーパラメータ探索で特定した値を設定
    solver = study.best_params['solver'],
    C = study.best_params['C'],
    max_iter = study.best_params['max_iter']
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
# 正解率の出力
print("Accuracy: {:.5f}".format(100 * accuracy_score(y_test, pred)))

# 混同行列の出力
print(confusion_matrix(y_test, pred))

# 適合率
print("Precision: {:.5f}".format(100 * precision_score(y_test, pred)))
# 再現率
print("Recall: {:.5f}".format(100 * recall_score(y_test, pred)))