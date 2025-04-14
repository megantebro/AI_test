import streamlit as st
import pandas as pd
import os
from io import StringIO
import predictor
from llm_chat import ask_gpt_for_model_chice
from predictor import train_and_predict
st.set_page_config(page_title="InsightBot", page_icon="🤖", layout="wide")

# タイトルと説明
st.title("🤖 InsightBot")
st.markdown("CSVをアップロードして、AIと会話しながらデータを理解しよう！")

# サイドバーに説明を表示
st.sidebar.header("📂 CSVアップロード")
st.sidebar.markdown("データをアップロードして、InsightBotと対話を始めましょう。")

# ファイルアップロード
uploaded_file = st.sidebar.file_uploader("CSVファイルを選んでください", type=["csv"])

# セッション状態の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    # ファイルをDataFrameに変換
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSVファイルが正常に読み込まれました ✅")

        # データのプレビュー表示
        st.subheader("📊 アップロードされたデータ")
        st.dataframe(df,height=400)

        # 簡単なデータ要約
        st.subheader("📈 データの基本情報")
        st.text(f"行数: {df.shape[0]} 行\n列数: {df.shape[1]} 列")
        st.write("カラム一覧:", df.columns.tolist())

        # チャットボックス
        st.subheader("💬 InsightBotと話してみよう")
        user_input = st.text_input("あなた: ", "")

        target_col = st.selectbox("予測したいカラムを選んでください",df.columns)

        if st.button("モデルを学習して予測"):
            model,acc,y_pred,y_test = train_and_predict(df,target_col)
            st.success(f"✅ モデルの精度（accuracy）：{acc:.2f}")
            

            st.subheader("📌 予測結果の例")
            results_df = pd.DataFrame({
                "正解（実際の値）": y_test[:10].tolist(),
                "予測": y_pred[:10].tolist()
            })
            st.dataframe(results_df)
        if user_input:
            # ユーザー入力を履歴に追加
            st.session_state.chat_history.append(("ユーザー", user_input))
            model_id, explanation = ask_gpt_for_model_chice(df, target_col)
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
else:
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")
