import streamlit as st
import pandas as pd
from scipy import stats
import io

st.set_page_config(page_title="アンケート統計解析ツール", layout="wide")

st.title("📊 アンケートクロス集計・カイ二乗検定ツール")

uploaded_file = st.file_uploader("クロス集計後のCSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        df = pd.read_csv(uploaded_file, encoding='cp932')

    base_cols = ['QID', 'questions', 'choices', 'total']
    attr_cols = [c for c in df.columns if c not in base_cols]
    
    summary_list = []
    detailed_list = []
    
    unique_qids = df['QID'].unique()

    for qid in unique_qids:
        q_data = df[df['QID'] == qid].copy()
        test_data = q_data[q_data['choices'] != '全体'][attr_cols].values
        
        p_value = None
        sig_mark = ""

        if test_data.size > 0 and test_data.sum() > 0:
            try:
                chi2, p, dof, expected = stats.chi2_contingency(test_data)
                p_value = p
                if p <= 0.01: sig_mark = "***"
                elif p <= 0.05: sig_mark = "**"
                elif p <= 0.10: sig_mark = "*"
            except:
                sig_mark = "検定不可"

        # 1. 要約用データの作成（設問につき1行）
        summary_list.append({
            "QID": qid,
            "設問内容": q_data['questions'].iloc[0],
            "p値": p_value,
            "有意水準": sig_mark
        })

        # 2. 詳細表用データの加工（先頭行にだけ結果を表示し、他は空文字にする）
        q_data['p値'] = ""
        q_data['有意水準'] = ""
        # '全体'という選択肢の行、またはその設問の最初の行にセット
        if '全体' in q_data['choices'].values:
            idx = q_data[q_data['choices'] == '全体'].index[0]
        else:
            idx = q_data.index[0]
            
        q_data.at[idx, 'p値'] = f"{p_value:.4f}" if p_value is not None else ""
        q_data.at[idx, '有意水準'] = sig_mark
        detailed_list.append(q_data)

    # 表示用データ作成
    summary_df = pd.DataFrame(summary_list)
    detailed_df = pd.concat(detailed_list)

    # UI表示
    st.subheader("📋 設問別・検定結果サマリー")
    st.write("どの設問に有意な差があるかの一覧です。")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("🔍 詳細集計表（検定結果付き）")
    st.write("各設問の「全体」行にのみ検定結果を表示しています。")
    st.dataframe(detailed_df, use_container_width=True)

    # ダウンロード機能（詳細表）
    csv_buffer = io.StringIO()
    detailed_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    
    st.download_button(
        label="✅ 解析済みCSVをダウンロード",
        data=csv_buffer.getvalue(),
        file_name="survey_analysis_results.csv",
        mime="text/csv",
    )

else:
    st.info("CSVファイルをアップロードしてください。")