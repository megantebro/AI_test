from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_and_predict_classification(df, target_column,model_id):
    # 特徴量と目的変数に分ける
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 数値データだけを使う（object型などを除外）
    X = X.select_dtypes(include=['float64', 'int64'])

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

def train_and_predict_regression(df, target_column, model_id=1):
    # 特徴量と目的変数を分離
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 数値データのみに限定（回帰は数値でないとNG）
    X = X.select_dtypes(include=["float64", "int64"])

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデル選択
    if model_id == 1:
        model = LinearRegression()
    elif model_id == 2:
        model = SVR()
    elif model_id == 3:
        model = RandomForestRegressor()
    elif model_id == 4:
        model = KNeighborsRegressor()
    else:
        raise ValueError("無効なmodel_idです（1〜4）")

    # 学習＆予測
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 評価指標
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, mse, r2, y_pred, y_test, X_test
