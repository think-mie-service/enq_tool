import streamlit as st
import pandas as pd
import re
import io

def parse_metadata(content):
    """
    YAML形式のマークダウンから設問テキスト、タイプ、選択肢を抽出する
    """
    q_defs = {}
    # ヘッダーから設問文を抽出 (## QID Text) [cite: 2]
    headers = re.findall(r'^##\s+(Q[\w-]+)\s+(.*)$', content, re.MULTILINE)
    header_map = {qid: text.strip() for qid, text in headers}

    # YAMLブロックを抽出 [cite: 2]
    blocks = re.findall(r'```yaml(.*?)```', content, re.DOTALL)

    for block in blocks:
        qid_match = re.search(r'qid:\s*(Q[\w-]+)', block)
        type_match = re.search(r'type:\s*([\w]+)', block) # カッコは1つなのでgroup(1)
        
        # qidとtypeの両方が見つかった場合のみ処理
        if qid_match and type_match:
            qid = qid_match.group(1)
            qtype = type_match.group(1) # group(2)からgroup(1)に修正
            
            # 選択肢(choices)セクションの解析 [cite: 2]
            choices_map = {}
            if 'choices:' in block:
                try:
                    choices_part = block.split('choices:')[1]
                    # "1": "ラベル" または 1: ラベル の形式を抽出
                    choice_pairs = re.findall(r'^\s+["\']?([\w.-]+)["\']?:\s*["\']?(.*?)["\']?$', choices_part, re.MULTILINE)
                    for k, v in choice_pairs:
                        choices_map[k] = v.strip()
                except Exception:
                    pass
            
            q_defs[qid] = {
                'Question': header_map.get(qid, ""),
                'type': qtype,
                'choices_map': choices_map
            }
    return q_defs

def transform_to_tableau_format(data_df, q_defs):
    """
    データをTableau用の縦持ち形式に変換する
    """
    rows = []
    
    # 最初の列（サンプル番号）の列名を取得 
    # odai_data.csvの構造に基づき、1列目をNoとして扱う
    no_col = data_df.columns[0]
    
    for _, row in data_df.iterrows():
        sample_no = row[no_col]
        
        for qid in data_df.columns:
            if qid == no_col:
                continue
                
            # 設問定義がない場合はIDのみで出力
            if qid not in q_defs:
                continue
                
            q_info = q_defs[qid]
            val = str(row[qid])
            
            # 空値(NaN)やnull文字列の処理 
            if val.lower() in ['nan', '', 'null', 'none']:
                continue
            
            # 複数回答(multi)の処理：カンマ区切りを分割して別行にする
            if q_info['type'] == 'multi':
                choices_list = [c.strip() for c in val.split(',')]
                for c_val in choices_list:
                    # 1.0 などの浮動小数点形式を整数文字列に変換
                    clean_c = c_val.split('.')[0]
                    label = q_info['choices_map'].get(clean_c, c_val)
                    rows.append({
                        'No': sample_no,
                        'qid': qid,
                        'Question': q_info['Question'],
                        'type': q_info['type'],
                        'choices': label
                    })
            else:
                # 単一回答等の処理
                clean_val = val.split('.')[0]
                label = q_info['choices_map'].get(clean_val, val)
                rows.append({
                    'No': sample_no,
                    'qid': qid,
                    'Question': q_info['Question'],
                    'type': q_info['type'],
                    'choices': label
                })
                
    return pd.DataFrame(rows)

# Streamlit UI
st.set_page_config(page_title="Tableau Data Converter", layout="wide")
st.title("Tableau用アンケートデータ変換ツール")

col1, col2 = st.columns(2)

with col1:
    uploaded_yaml = st.file_uploader("1. 設問定義ファイル (odai_yaml.md) をアップロード", type=['md', 'txt'])
with col2:
    uploaded_data = st.file_uploader("2. 回答データファイル (odai_data.csv) をアップロード", type=['csv'])

if uploaded_yaml and uploaded_data:
    if st.button("変換を開始"):
        try:
            # ファイル読み込み（UTF-8指定） [cite: 2]
            yaml_content = uploaded_yaml.read().decode("utf-8")
            data_df = pd.read_csv(uploaded_data)
            
            with st.spinner('データを変換中...'):
                q_defs = parse_metadata(yaml_content)
                result_df = transform_to_tableau_format(data_df, q_defs)
            
            if result_df.empty:
                st.warning("変換後のデータが空です。QIDが一致しているか確認してください。")
            else:
                st.success(f"変換完了！ {len(result_df)} 行のデータを作成しました。")
                
                # プレビュー
                st.write("### 変換結果プレビュー")
                st.dataframe(result_df.head(20))
                
                # ダウンロード用
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                st.download_button(
                    label="変換済みCSVをダウンロード",
                    data=csv_buffer.getvalue(),
                    file_name="tableau_tate_data.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")