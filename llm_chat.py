from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt_for_model_chice(df):
    columns_description = "\n".join(df.columns)

    target_variable = "species"

    prompt = f"""
    このデータは以下のようなカラムを持っています：

    {columns_description}

    目的変数は {target_variable} です。

    あなたが分類器を選ぶとしたら、次のうちどれが最適だと思いますか？

    1. Perceptron  
    2. SVM  
    3. Random Forest  
    4. KNN  

    最も適していると思う番号を、文章の最初に数字だけで返答してください。その後に理由を説明してください。
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "あなたはデータ分析に長けた機械学習アドバイザーです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    # 出力結果
    reply = response["choices"][0]["message"]["content"]
    print("GPTの返答:", reply)

    # 数字だけ取り出す（1, 2, 3, 4）
    selected_model = int(reply.strip()[0])
    print("選ばれたモデル番号:", selected_model)