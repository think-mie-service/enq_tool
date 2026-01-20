import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import time #
# --- Functions ---

def dataframe_to_markdown(df, question_id, attribute, question_text, answer_type):
    """
    Converts a filtered dataframe into a markdown formatted string for the prompt.
    (This function remains the same as before)
    """
    if df.empty:
        return f"### {question_id}, {attribute}: データなし\n"

    header = f"### 分析対象: {question_text} ({question_id}, {attribute}, {answer_type}回答)"
    
    if attribute == '全体':
        table_df = df[['Choice', 'Value']]
        markdown_table = table_df.to_markdown(index=False)
    else:
        table_df = df[['Category', 'Choice', 'Value']]
        markdown_table = table_df.to_markdown(index=False)
        
    return f"{header}\n{markdown_table}\n\n"

# --- NEW: Modified Prompt Generation Function ---
def generate_single_analysis_prompt(data_markdown, user_example):
    """
    Generates a prompt for a SINGLE analysis combination.
    """
    prompt = f"""
あなたはプロのデータアナリストです。
以下の##集計データのみを分析し、調査報告書に記載する分析コメントを作成してください。

報告書は、ユーザーが提示した##回答例の構成とトーンを参考に、以下の指示を厳守してください。

### 指示
1.  **分析対象の明確化**: まず、分析対象の設問が何であるかを簡潔に述べます。
2.  **客観的事実の記述**: 集計データから読み取れる客観的な事実（例：「『満足』と回答した割合は、男性が50%であるのに対し、女性は70%と20ポイント高い」）を具体的な数値を用いて記述してください。特に、最も多い選択肢、少ない選択肢、カテゴリ間の差が大きい点などに着目してください。
3.  **考察の記述**: その事実から考えられる示唆や考察を、読み手が納得しやすいように論理的に記述してください。
4.  **形式**: 全体をMarkdown形式で、報告書にそのまま引用できるような、見出しを含んだ丁寧な文章で出力してください。

---

## 回答例
{user_example}

---

## 集計データ
{data_markdown}

---

## 生成する分析コメント
"""
    return prompt

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("📊 Gemini アンケート報告書作成支援アプリ V2")

# --- Sidebar (remains mostly the same) ---
with st.sidebar:
    st.header("設定")
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
        required_cols = {'QuestionID', 'QuestionText', 'AnswerType', 'Attribute', 'Category', 'Choice', 'ValueType', 'Value'}
        if not required_cols.issubset(df.columns):
            st.error(f"エラー: CSVファイルに必須列が含まれていません。必須列: {', '.join(required_cols)}")
        else:
            st.success("CSVファイルを正常に読み込みました。")
            
            # Allow user to select which questions to display
            unique_questions = df['QuestionID'].unique()
            selected_questions = st.multiselect("プレビューする設問を選択", options=unique_questions, default=unique_questions[:3])
            if selected_questions:
                st.dataframe(df[df['QuestionID'].isin(selected_questions)])

            st.subheader("分析対象の指定")
            request_text = st.text_area(
                "分析したい `QuestionID` と `Attribute` の組み合わせを、1行に1つずつ入力してください。",
                height=150,
                value="問1,全体\n問2,全体\n問3,全体",
                help="例:\n問1,全体\n問1,性別\n問10,全体"
            )

            # --- NEW: More concrete user example ---
            user_example_text = """
### Q1. サービス満足度（性別クロス）
**事実**:
「満足」と回答した割合は、男性が86.2%（250/290）であるのに対し、女性は71.4%（150/210）と、男性が14.8ポイント高い結果となった。特に「不満」と回答した割合は女性（28.6%）が男性（13.8%）の2倍以上となっている。

**考察**:
全体として満足度は高いものの、男女間で満足度に差が見られる。特に女性の不満度が高い背景には、サービスの特定機能やデザインが男性向けになっている可能性が考えられる。女性ユーザーの具体的な不満点を深掘り調査する必要がある。
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
                        
                        st.subheader("生成された分析レポート")
                        
                        request_list = [line.strip().split(',') for line in request_text.strip().split('\n') if line.strip()]
                        
                        # --- NEW: Process requests one by one ---
                        progress_bar = st.progress(0)
                        for i, req in enumerate(request_list):
                            if len(req) == 2:
                                qid, attr = req[0].strip(), req[1].strip()
                                
                                with st.spinner(f"分析中: {qid}, {attr}"):
                                    subset_df = df[(df['QuestionID'] == qid) & (df['Attribute'] == attr)]
                                    
                                    if not subset_df.empty:
                                        q_text = subset_df['QuestionText'].iloc[0]
                                        a_type = subset_df['AnswerType'].iloc[0]
                                        
                                        data_md = dataframe_to_markdown(subset_df, qid, attr, q_text, a_type)
                                        
                                        final_prompt = generate_single_analysis_prompt(data_md, user_example_text)
                                        
                                        # Call Gemini API
                                        response = model.generate_content(final_prompt)
                                        
                                        st.markdown(f"--- \n\n {response.text}")
                                        
                                        # To avoid hitting API rate limits, wait 4 seconds
                                        time.sleep(4) 
                                    else:
                                        st.warning(f"**警告**: `{qid}, {attr}` に該当するデータが見つかりませんでした。スキップします。")
                            else:
                                st.warning(f"**警告**: `{','.join(req)}` は不正な形式です。スキップします。")
                            
                            # Update progress bar
                            progress_bar.progress((i + 1) / len(request_list))

                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

    except Exception as e:
        st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")

