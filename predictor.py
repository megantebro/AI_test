from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_and_predict(df, target_column,model_id=3):
    # 特徴量と目的変数に分ける
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 数値データだけを使う（object型などを除外）
    X = X.select_dtypes(include=['float64', 'int64'])

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルの作成と学習
    if model_id == 1:
        model = Perceptron()
    elif model_id == 2:
        model = SVC()
    elif model_id == 3:
        model = RandomForestClassifier()
    elif model_id == 4:
        model = KNeighborsClassifier()
    else:
        raise ValueError('idが無効です')
    

    model.fit(X_train, y_train)

    # 予測と精度の計算
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, y_pred, y_test
