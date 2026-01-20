import streamlit as st
import pandas as pd
import re
import io

st.set_page_config(page_title="アンケート・クロス集計ツール", layout="wide")

# --- 1. 設問定義ファイルの解析関数 ---
def parse_markdown_yaml(content):
    questions = {}
    # ヘッダー（## QID 設問文）の抽出
    headers = re.findall(r'##\s+([\w\-]+)\s+(.*?)\n', content)
    header_map = {qid: title for qid, title in headers}

    # YAMLブロックの抽出
    blocks = re.findall(r'```yaml\s*\{(.*?)\}\n(.*?)```', content, re.DOTALL)
    for meta, body in blocks:
        qid_match = re.search(r'qid:\s*([\w\-]+)', body)
        if not qid_match: continue
        qid = qid_match.group(1).strip()
        
        # 選択肢(choices)の抽出
        choices = {}
        choice_block = re.search(r'choices:\n(.*?)(?=\n\w+:|\Z)', body, re.DOTALL)
        if choice_block:
            choice_lines = re.findall(r'^\s+"?([\w\-]+)"?:\s+"?(.*?)"?$', choice_block.group(1), re.MULTILINE)
            choices = {k: v for k, v in choice_lines}
            
        questions[qid] = {
            'title': header_map.get(qid, qid),
            'choices': choices
        }
    return questions

# --- 2. メインUI ---
st.title("📊 アンケート・クロス集計ツール")
st.markdown("Markdown（定義）とCSV（ローデータ）をアップロードして集計します。")

# サイドバーでファイルをアップロード
st.sidebar.header("📁 ファイルアップロード")
md_file = st.sidebar.file_uploader("1. 設問定義(md)をアップロード", type=["md", "txt"])
data_file = st.sidebar.file_uploader("2. ローデータ(csv)をアップロード", type=["csv"])

if md_file and data_file:
    try:
        # ファイルの読み込み
        df_raw = pd.read_csv(data_file)
        md_content = md_file.getvalue().decode("utf-8")
        q_defs = parse_markdown_yaml(md_content)
        
        st.sidebar.success("✅ 読み込み完了")

        # 設問の選択肢リスト作成（データにある列のみ）
        qid_options = [qid for qid in q_defs.keys() if qid in df_raw.columns]
        qid_labels = {qid: f"{qid}: {q_defs[qid]['title'][:30]}..." for qid in qid_options}

        st.sidebar.divider()
        st.sidebar.header("⚙️ 集計設定")
        
        row_var = st.sidebar.selectbox(
            "行の見出し（例：地区、年齢）", 
            qid_options, 
            format_func=lambda x: qid_labels[x]
        )
        col_var = st.sidebar.selectbox(
            "列の見出し（集計したい設問）", 
            qid_options, 
            format_func=lambda x: qid_labels[x]
        )

        if st.button("集計を実行"):
            # データクリーニング
            def clean_val(v):
                if pd.isna(v): return "無回答"
                return str(v).split('.')[0]

            df_plot = df_raw[[row_var, col_var]].copy()
            df_plot[row_var] = df_plot[row_var].apply(clean_val)
            df_plot[col_var] = df_plot[col_var].apply(clean_val)

            # ラベルへのマッピング
            row_choices = q_defs[row_var]['choices']
            col_choices = q_defs[col_var]['choices']
            
            df_plot[row_var] = df_plot[row_var].map(lambda x: row_choices.get(x, x))
            df_plot[col_var] = df_plot[col_var].map(lambda x: col_choices.get(x, x))

            # クロス集計
            ct_count = pd.crosstab(df_plot[row_var], df_plot[col_var], margins=True, margins_name="合計")
            ct_percent = pd.crosstab(df_plot[row_var], df_plot[col_var], normalize='index').applymap(lambda x: f"{x:.1%}")

            # 結果表示
            st.subheader(f"分析結果: {q_defs[col_var]['title']}")
            
            tab1, tab2 = st.tabs(["🔢 度数表（人数）", "📈 構成比（％）"])
            with tab1:
                st.dataframe(ct_count, use_container_width=True)
            with tab2:
                st.dataframe(ct_percent, use_container_width=True)
            # --- ここから追加 ---
            if st.button("📋 クリップボード用にデータを整形"):
                # Excelに貼り付けやすい「タブ区切りテキスト」に変換
                # ct_percent（割合）または ct_count（度数）お好きな方を指定してください
                tsv_text = ct_percent.to_csv(sep='\t')
                
                st.info("下のテキストを全選択（Ctrl+A）してコピーし、Excelに貼り付けてください。")
                st.text_area("コピー用エリア", value=tsv_text, height=150)
            # --- ここまで ---
            # エクセル出力
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                ct_count.to_excel(writer, sheet_name='度数表')
                ct_percent.to_excel(writer, sheet_name='構成比')
            
            st.download_button(
                label="📥 集計結果をExcelで保存",
                data=output.getvalue(),
                file_name=f"crosstab_{row_var}_{col_var}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    # ファイル未アップロード時のガイド
    st.info("サイドバーから『設問定義（Markdown）』と『ローデータ（CSV）』をアップロードしてください。")
    
    # 仕組みの図解（任意）
    #