from openai import OpenAI
import openai
from dotenv import load_dotenv
import os
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_gpt_for_model_choice(df, is_regression, user_input=None):
    print(user_input)
    columns_description = "\n".join(df.columns)
    target_variable = "species"  # ※必要なら引数に変更も可

    # 基本のプロンプト部分
    prompt = f"""
このデータは以下のようなカラムを持っています：

{columns_description}

目的変数は {target_variable} です。
この問題は{"回帰" if is_regression else "分類"}問題です。

あなたが{'回帰モデル' if is_regression else '分類器'}を選ぶとしたら、次のうちどれが最適だと思いますか？

1. Perceptron  
2. SVM  
3. Random Forest  
4. KNN
"""

    # ユーザーの追加メッセージがある場合
    if user_input:
        prompt += f"""

ユーザーからの追加のリクエストがあります：
「{user_input}」

⚠️ 上記のリクエストを最優先で考慮してください。
「〇〇を使わないで」「〇〇以外で」などのリクエストがある場合は、そのモデルを**絶対に選ばないでください**。
"""

    # 回答フォーマットの強制
    prompt += """

以下のフォーマットで回答してください：

選んだモデル番号: <1〜4のどれか>
理由: <なぜそのモデルを選んだか>
他に検討してもよいモデル: <任意>
"""

    # GPT呼び出し
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたはデータ分析に長けた機械学習アドバイザーです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    # GPTの応答取得
    reply = response.choices[0].message.content.strip()

    # モデル番号を安全に抽出
    match = re.search(r"選んだモデル番号[:：]\s*([1-4])", reply)
    if match:
        selected_model = int(match.group(1))
    else:
        selected_model = 3  # fallback: 3（Random Forest）または任意で変更可

    return reply, selected_model
