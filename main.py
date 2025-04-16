import streamlit as st
import pandas as pd
import os
from io import StringIO
import seaborn as sns 
from llm_chat import ask_gpt_for_model_choice
from predictor import train_and_predict_classification,train_and_predict_regression
st.set_page_config(page_title="InsightBot", page_icon="🤖", layout="wide")

# タイトルと説明
st.title("🤖 InsightBot")
st.markdown("CSVをアップロードして、AIと会話しながらデータを理解しよう！")

# サイドバーに説明を表示
st.sidebar.header("📂 CSVアップロード")
st.sidebar.markdown("データをアップロードして、InsightBotと対話を始めましょう。")

st.sidebar.header("📊 使用するデータの選択")

dataset_option = st.sidebar.selectbox(
    "データを選んでください",
    ("サンプル：Iris", "サンプル：Tips", "CSVをアップロード")
)


# セッション状態の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

df = None  # 最初に定義しておく



# ファイルをDataFrameに変換
try:
    df = None  # 最初に定義しておく

    if dataset_option == "サンプル：Iris":
        df = sns.load_dataset("iris")
        st.sidebar.success("サンプルデータ 'iris' が読み込まれました ✅")

    elif dataset_option == "サンプル：Tips":
        df = sns.load_dataset("tips")
        st.sidebar.success("サンプルデータ 'tips' が読み込まれました ✅")

    elif dataset_option == "CSVをアップロード":
        uploaded_file = st.sidebar.file_uploader("CSVファイルを選んでください", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("CSVファイルが正常に読み込まれました ✅")
            except Exception as e:
                st.sidebar.error(f"読み込み中にエラーが発生しました: {e}")

    # データのプレビュー表示
    st.subheader("📊 アップロードされたデータ")
    st.dataframe(df,height=400)

    # 簡単なデータ要約
    st.subheader("📈 データの基本情報")
    st.text(f"行数: {df.shape[0]} 行\n列数: {df.shape[1]} 列")
    st.write("カラム一覧:", df.columns.tolist())

    target_col = st.selectbox("予測したいカラムを選んでください",df.columns)
    
    is_regression = st.toggle("🔁 回帰モードにする", value=True)

    # チャットボックス
    st.subheader("💬 InsightBotと話してみよう")
    user_input = st.text_input("あなた: ", "")

    if st.button("AIに予測してもらう"):
        reply, selected_model = ask_gpt_for_model_choice(df, is_regression,user_input)

        if is_regression:
            model, mae, mse, r2, y_pred, y_test, X_test = train_and_predict_regression(df, target_col, selected_model)

            model_names = {
                1: "Linear Regression",
                2: "SVR",
                3: "Random Forest Regressor",
                4: "KNN Regressor"
            }
            st.success(f"✅ GPTが選んだモデル: **{model_names[selected_model]}**（モデル番号: {selected_model}）")
            st.write(f"📊 MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

            st.subheader("🤖 InsightBotのモデル選定理由")
            st.write(reply)

            st.subheader("📌 予測結果の例")
            results_df = X_test.copy()
            results_df["正解（実際の値）"] = y_test
            results_df["予測値"] = y_pred
            results_df["誤差"] = y_pred - y_test
            st.dataframe(results_df)
        
        else:
            model, acc, y_pred, y_test = train_and_predict_classification(df, target_col, selected_model)

            model_names = {
                1: "Perceptron",
                2: "SVM",
                3: "Random Forest",
                4: "KNN"
            }
            st.success(f"✅ GPTが選んだモデル: **{model_names[selected_model]}**（モデル番号: {selected_model}）")
            st.write(f"🎯 精度（accuracy）：**{acc:.2f}**")

            st.subheader("🤖 InsightBotのモデル選定理由")
            st.write(reply)

            X = df.drop(columns=[target_col])
            X_test_display = X.loc[y_test.index]

            st.subheader("📌 予測結果の例")
            results_df = X_test_display.copy()
            results_df["正解（実際の値）"] = y_test
            results_df["予測"] = y_pred
            st.dataframe(results_df)



    st.subheader("手動予測")

    if is_regression:
        model_names = {
            1: "Linear Regression",
            2: "SVR",
            3: "Random Forest Regressor",
            4: "KNN Regressor"
        }
    else:
        model_names = {
            1: "Perceptron",
            2: "SVM",
            3: "Random Forest",
            4: "KNN"
        }

    model_label_to_id = {v: k for k, v in model_names.items()}
    selected_model_label = st.selectbox("使用するモデルを選んでください", list(model_names.values()))
    selected_model_id = model_label_to_id[selected_model_label]

    if st.button("モデルを学習して予測"):
        if is_regression:
            model, mae, mse, r2, y_pred, y_test, X_test = train_and_predict_regression(df, target_col, selected_model_id)
            st.success(f"📊 MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

            st.subheader("📌 予測結果の例")
            results_df = X_test.copy()
            results_df["正解（実際の値）"] = y_test
            results_df["予測値"] = y_pred
            results_df["誤差"] = y_pred - y_test
            st.dataframe(results_df)
        else:
            model, acc, y_pred, y_test = train_and_predict_classification(df, target_col, selected_model_id)
            st.success(f"✅ モデルの精度（accuracy）：{acc:.2f}")

            X = df.drop(columns=[target_col])
            X_test_display = X.loc[y_test.index]
            st.subheader("📌 予測結果の例")
            results_df = X_test_display.copy()
            results_df["正解（実際の値）"] = y_test
            results_df["予測"] = y_pred
            st.dataframe(results_df)

    if user_input:
        # ユーザー入力を履歴に追加
        st.session_state.chat_history.append(("ユーザー", user_input))
        model_id, explanation = ask_gpt_for_model_choice(df, target_col,user_input)
        # 仮の応答（今後AIモデルで拡張予定）
        bot_reply = "これは仮の返答です。今後、データに基づいたAI応答をここに表示します。"
        st.session_state.chat_history.append(("InsightBot", bot_reply))


    # チャット履歴の表示
    if st.session_state.chat_history:
        for sender, msg in st.session_state.chat_history:
            if sender == "ユーザー":
                st.markdown(f"🧑‍💻 **{sender}**: {msg}")
            else:
                st.markdown(f"🤖 **{sender}**: {msg}")

except Exception as e:
    st.error(f"読み込み中にエラーが発生しました: {e}")

