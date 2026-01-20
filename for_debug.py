import csv
import io
from collections import defaultdict

# 引数は必ずこの4つ（左側の変数リストと一致させる）
def main(q_csv: str, d_csv: str, c_csv: str, target_tag: str):
    # 1. データの受け取りチェック
    if not q_csv or len(str(q_csv)) < 5:
        return {"output": "【重大エラー】設問定義データがPythonに届いていません。"}
    
    target_tag = target_tag.strip()
    
    # 5つの固定された分析軸
    axes = [
        {'qid': 'Q0-3', 'name': '地区'},
        {'qid': 'Q0-1', 'name': '年齢'},
        {'qid': 'Q1-1', 'name': '家族構成'},
        {'qid': 'Q1-3', 'name': '経済状況'},
        {'qid': 'Q2-8', 'name': '外出頻度'}
    ]
    
    # 2. 選択肢マスタの読み込み
    choice_map = defaultdict(dict)
    c_file = io.StringIO(c_csv.strip())
    c_reader = csv.DictReader(c_file)
    for row in c_reader:
        qid = row.get('qid', '').strip()
        val = row.get('choice_value', '').strip().split('.')[0]
        label = row.get('choice_label', '').strip()
        if qid and val:
            choice_map[qid][val] = label

    # ラベル変換用ヘルパー
    def get_labels(qid, val_str):
        if not val_str or str(val_str).lower() in ['nan', '', 'none', 'null']:
            return "無回答"
        parts = [p.strip() for p in str(val_str).split(',')]
        labels = []
        for p in parts:
            clean_p = p.split('.')[0]
            area_age_maps = {
                'Q0-3': {"1": "日進", "2": "川添", "3": "三瀬谷", "4": "荻原", "5": "領内", "6": "大杉谷"},
                'Q0-1': {"1": "65歳未満", "2": "65-69歳", "3": "70-74歳", "4": "75-79歳", "5": "80-84歳", "6": "85-89歳", "7": "90歳以上"}
            }
            if qid in area_age_maps:
                l = area_age_maps[qid].get(clean_p, f"不明({clean_p})")
            else:
                l = choice_map.get(qid, {}).get(clean_p, f"選択肢{clean_p}")
            labels.append(l)
        return ",".join(labels)

    # 3. 設問定義の解析
    q_file = io.StringIO(q_csv.strip())
    q_reader = csv.DictReader(q_file)
    target_q_info = {}
    for row in q_reader:
        tags_raw = row.get('tags', '')
        if not tags_raw: continue
        tag_list = [t.strip() for t in str(tags_raw).split(',')]
        if target_tag in tag_list:
            qid = row.get('qid')
            if qid:
                target_q_info[qid] = row.get('question', qid)
            
    if not target_q_info:
        return {"output": f"タグ '{target_tag}' に該当する設問が見つかりませんでした。"}

    # 4. ローデータの読み込み
    d_file = io.StringIO(d_csv.strip())
    d_reader = list(csv.DictReader(d_file))

    report = f"# 【総合分析レポート】課題：{target_tag}\n\n"
    
    for axis in axes:
        axis_col = axis['qid']
        axis_name = axis['name']
        report += f"## 分析軸：{axis_name} ({axis_col})\n"
        
        for qid, q_text in target_q_info.items():
            counts = defaultdict(lambda: defaultdict(int))
            all_choices_set = set()
            
            for row in d_reader:
                raw_axis_val = row.get(axis_col, "").strip()
                axis_label = get_labels(axis_col, raw_axis_val)
                val = row.get(qid, "").strip()
                label = get_labels(qid, val)
                counts[axis_label][label] += 1
                all_choices_set.add(label)
            
            all_choices = sorted(list(all_choices_set))
            if "無回答" in all_choices:
                all_choices.remove("無回答")
                all_choices.append("無回答")
            
            report += f"### 設問 {qid}: {q_text}\n"
            report += "| 比較項目 | " + " | ".join(all_choices) + " |\n"
            report += "| --- | " + " | ".join(["---"] * len(all_choices)) + " |\n"
            
            for axis_label in sorted(counts.keys()):
                row_vals = counts[axis_label]
                total = sum(row_vals.values())
                if total > 0:
                    report += f"| {axis_label} | " + " | ".join([f"{(row_vals[c]/total*100):.1f}%" for c in all_choices]) + " |\n"
            report += "\n"
        report += "---\n\n"
            
    return {"output": report}