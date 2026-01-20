import streamlit as st
import pandas as pd
import google.generativeai as genai
import io

# --- Functions ---

def dataframe_to_markdown(df, question_id, attribute, question_text, answer_type):
    """Converts a filtered dataframe into a markdown formatted string for the prompt."""
    if df.empty:
        return f"### {question_id}, {attribute}: データなし\n"

    header = f"### {question_id}, {attribute}: {question_text} ({answer_type}回答)"
    
    # For overall results (Attribute == '全体'), table is simpler
    if attribute == '全体':
        table_df = df[['Choice', 'Value']]
        markdown_table = table_df.to_markdown(index=False)
    # For crosstabs, include the Category
    else:
        table_df = df[['Category', 'Choice', 'Value']]
        markdown_table = table_df.to_markdown(index=False)
        
    return f"{header}\n{markdown_table}\n\n"

def generate_analysis_prompt(data_markdown, user_request, user_example):
    """Generates the full prompt for the Gemini API."""
    
    prompt = f"""
あなたはプロのデータアナリストです。
以下の##集計データと##分析対象リストに基づいて、アンケート調査報告書を作成してください。

報告書は、ユーザーが提示した##回答例の構成とトーンを参考に、以下の指示に従って作成してください。

### 指示
1.  **順番の厳守**: 「分析対象リスト」に記載された組み合わせごとに、順番に分析コメントを記述してください。
2.  **全体傾向の分析**: まず、各設問（QuestionID）の全体傾向（Attributeが「全体」）を要約してください。主要な選択肢とその数値を具体的に挙げてください。
3.  **比較分析**: 次に、同じ設問のクロス集計結果（例：「性別」「年代」）を分析し、全体傾向との比較や、カテゴリ間の差異（例：男女差、年代差）が明確にわかるように記述してください。ポイント差など具体的な数値を用いて比較してください。
4.  **設問ごとの考察**: 各設問（Q1, Q2...）の分析の最後には、データから読み取れる総合的な示唆や考察を、ユーザーの回答例のように簡潔にまとめてください。
5.  **自然な文章**: 全体を通して、レポートとして自然で、読みやすい文章で出力してください。

---

## 回答例
{user_example}

---

## 分析対象リスト
{user_request}

---

## 集計データ
{data_markdown}

---

## 生成する報告書
"""
    return prompt

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("📊 Gemini アンケート報告書作成支援アプリ")

# --- Sidebar for API Key and Instructions ---
with st.sidebar:
    st.header("設定")
    # Use st.secrets for deployment, with a fallback for local development
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("APIキーを読み込みました。")
    except (FileNotFoundError, KeyError):
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        st.info("Streamlit Secretsに `GEMINI_API_KEY` を設定することを推奨します。")

    st.header("データ形式")
    st.caption("以下の8列を持つCSVファイルをアップロードしてください。")
    st.markdown("""
    - `QuestionID`
    - `QuestionText`
    - `AnswerType` (Single/Multiple)
    - `Attribute`
    - `Category`
    - `Choice`
    - `ValueType` (回答数/割合)
    - `Value`
    """)
    
    # Example data download
    example_df = pd.DataFrame({
        'QuestionID': ['Q1', 'Q1', 'Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2', 'Q2'],
        'QuestionText': ['サービスへの総合満足度', 'サービスへの総合満足度', 'サービスへの総合満足度', 'サービスへの総合満足度', 'サービスへの総合満足度', 'サービスへの総合満足度', 'よく利用する機能', 'よく利用する機能', 'よく利用する機能', 'よく利用する機能', 'よく利用する機能'],
        'AnswerType': ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Multiple', 'Multiple', 'Multiple', 'Multiple', 'Multiple'],
        'Attribute': ['全体', '全体', '性別', '性別', '性別', '性別', '全体', '全体', '全体', '性別', '性別'],
        'Category': ['全体', '全体', '男性', '女性', '男性', '女性', '全体', '全体', '全体', '男性', '女性'],
        'Choice': ['満足', '不満', '満足', '満足', '不満', '不満', '機能A', '機能B', '機能C', '機能A', '機能C'],
        'ValueType': ['回答数', '回答数', '回答数', '回答数', '回答数', '回答数', '回答数', '回答数', '回答数', '回答数', '回答数'],
        'Value': [400, 100, 250, 150, 40, 60, 300, 200, 450, 200, 200]
    })
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8-sig')

    csv = convert_df_to_csv(example_df)
    
    st.download_button(
        label="サンプルCSVをダウンロード",
        data=csv,
        file_name='sample_survey_data.csv',
        mime='text/csv',
    )


# --- Main Content ---
uploaded_file = st.file_uploader("集計結果のCSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Check for required columns
        required_cols = {'QuestionID', 'QuestionText', 'AnswerType', 'Attribute', 'Category', 'Choice', 'ValueType', 'Value'}
        if not required_cols.issubset(df.columns):
            st.error(f"エラー: CSVファイルに必須列が含まれていません。必須列: {', '.join(required_cols)}")
        else:
            st.success("CSVファイルを正常に読み込みました。")
            st.dataframe(df.head())

            st.subheader("分析対象の指定")
            request_text = st.text_area(
                "分析したい `QuestionID` と `Attribute` の組み合わせを、1行に1つずつ入力してください。",
                height=150,
                value="Q1,全体\nQ1,性別\nQ2,全体\nQ2,性別",
                help="例:\nQ1,全体\nQ1,性別\nQ2,年代"
            )

            # User example as provided in the prompt
            user_example_text = """
Q1サービスへの総合満足度は、「とても満足」と答えた人がxx%で最も多く、やや不満がxx%で続いている。「とても不満」と答えた人がxx%で最も少なかった。
Q1を性別でみると、「とても満足」と答えた人は男性の方がxxポイント高く、「とても不満」と答えた人は女性の方がxxポイント高かった。
全体に満足度と答えた人は半数を超えているが、男性の方が満足度が高く、今後女性の視点を考慮する必要がある。
Q2は機能Cと答えた人がxx%で最も多く、機能Aと答えた人がxx%で続いている。機能Eと答えた人はxx%で最も低かった。
性別でみると機能Cと答えた人は女性の方がxxポイント高く、一方機能Aは男性がxxポイント高くなっている。
年代別に見ると、機能Cは年代が若いほど利用する人の比率は高くなっており、機能Aは年齢が高い人の方が利用の比率が高くなっている。
全体で見ると利用率の高い機能Cが若い男性の利用が高いため、その要因についてさらに調査する必要がある。
            """

            if st.button("分析コメントを生成する", type="primary"):
                if not gemini_api_key:
                    st.error("サイドバーでGemini APIキーを設定してください。")
                elif not request_text.strip():
                    st.warning("分析対象を1つ以上入力してください。")
                else:
                    try:
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')

                        with st.spinner("分析コメントを生成中です..."):
                            # Parse user request
                            request_list = [line.strip().split(',') for line in request_text.strip().split('\n') if line.strip()]
                            
                            filtered_data_markdown = ""
                            valid_requests = []
                            for req in request_list:
                                if len(req) == 2:
                                    qid, attr = req[0].strip(), req[1].strip()
                                    
                                    # Filter dataframe for the request
                                    subset_df = df[(df['QuestionID'] == qid) & (df['Attribute'] == attr)]
                                    
                                    if not subset_df.empty:
                                        # Get QuestionText and AnswerType from the first row of the subset
                                        q_text = subset_df['QuestionText'].iloc[0]
                                        a_type = subset_df['AnswerType'].iloc[0]
                                        
                                        # Append markdown table to the string
                                        filtered_data_markdown += dataframe_to_markdown(subset_df, qid, attr, q_text, a_type)
                                        valid_requests.append(f"{qid},{attr}")
                                    else:
                                        st.warning(f"警告: `{qid}, {attr}` に該当するデータが見つかりませんでした。スキップします。")
                                else:
                                    st.warning(f"警告: `{','.join(req)}` は不正な形式です。スキップします。")
                            
                            if filtered_data_markdown:
                                # Generate the final prompt
                                final_prompt = generate_analysis_prompt(filtered_data_markdown, "\n".join(valid_requests), user_example_text)
                                
                                # Call Gemini API
                                response = model.generate_content(final_prompt)
                                
                                st.subheader("生成された分析レポート")
                                st.markdown(response.text)
                            else:
                                st.error("分析対象のデータが見つかりませんでした。入力内容を確認してください。")

                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

    except Exception as e:
        st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
