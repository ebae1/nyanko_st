import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
from decimal import Decimal, InvalidOperation

#requirements.txtに必要ライブラリを記入しておく

# --- ページ設定 ---
st.set_page_config(layout='wide')

# --- データ読み込み ---
@st.cache_data
def load_data():
    return pd.read_excel('./catsdb.xlsx', index_col=0)

@st.cache_data
def load_data_e():
    return pd.read_excel('./nyanko_enemy_db.xlsx', index_col=0)

# オリジナルのデータフレームをロードして保持
df_orig = load_data()
df_e_orig = load_data_e()

# グラフと表の順序の基準となるリスト
target_cols_cats = ['own','No.','ランク','キャラクター名','コスト','再生産F','速度','範囲','射程','発生F','攻撃力','頻度F','DPS','体力','KB','特性']
target_cols_enemy = ['速度','射程','DPS','体力','KB','攻撃力','頻度F','攻発F','お金']

# --- データクレンジング ---
numeric_cols_cats = [col for col in target_cols_cats if col != '範囲' or col !='特性' or col !='ランク' or col !='キャラクター名']
for col in numeric_cols_cats:
    if col in df_orig.columns:
        df_orig[col] = pd.to_numeric(df_orig[col], errors='coerce')
for col in target_cols_enemy:
    if col in df_e_orig.columns:
        df_e_orig[col] = pd.to_numeric(df_e_orig[col], errors='coerce')






# --- 抽出による列の作成 ---

import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation

# --- 🎯 サンプルデータを作成 ---

df = df_orig.copy()
src_col = '特性'

# --- ユーティリティ関数 ---
def to_decimal(value):
    if pd.isna(value): return np.nan
    try: return Decimal(str(value))
    except InvalidOperation: return np.nan

# --- 📜 ここからがメインの処理 ---

# 0. 行内改行で分割・展開
df_exploded = df.assign(line=df[src_col].str.split('\n')).explode('line')
s = df_exploded['line'].astype(str).str.strip()
results_df = pd.DataFrame(index=s.index)

# 0b. 処理対象外の行を特定
process_s = s.copy()
ignore_mask = s.str.startswith('無効', na=False)
process_s[ignore_mask] = '' 

# --- ▼ 1. 行全体から抽出可能な全ての属性を先に計算しておく ▼ ---
probability = process_s.str.extract(r'(\d+\.?\d*)\s*[%％]の確率で').astype(float)
condition_num_str = process_s.str.extract(r'(\d+\.?\d*)(?=\s*[%％].*?で)')[0]
results_df['発動条件'] = condition_num_str.apply(to_decimal)
damage_multiplier = process_s.str.extract(r'与ダメ\s*x(-?[\d.]+)').astype(float)
effect_detail_s = process_s.str.extract(r'で(.*$)')[0].fillna('')

duration_range = effect_detail_s.str.extract(r'(\d+)～(\d+)F')
duration_single_start = effect_detail_s.str.extract(r'^\s*(\d+)F')
duration_anywhere = process_s.str.extract(r'\((\d+)\s*F')[0].astype(float)
duration_max = pd.to_numeric(duration_range[1], errors='coerce') \
                 .fillna(pd.to_numeric(duration_single_start[0], errors='coerce')) \
                 .fillna(duration_anywhere)

# 射程の抽出ロジックを、負の値や小数を含む数値に対応させる
# '射程 X~Y' 形式の抽出
extracted_range_xy = process_s.str.extract(r'射程\s*(-?[\d.]+)～(-?[\d.]+)')
range_min_from_xy = pd.to_numeric(extracted_range_xy[0], errors='coerce')
range_max_from_xy = pd.to_numeric(extracted_range_xy[1], errors='coerce')

# '射程 X' 単一形式の抽出 ( '~' が続かないことを確認)
extracted_single_range = process_s.str.extract(r'射程\s*(-?[\d.]+)(?!～)')[0]
single_range_value = pd.to_numeric(extracted_single_range, errors='coerce')

debuff_rate = process_s.str.extract(r'攻撃力(\d+\.?\d*)[%％]に低下').astype(float) / 100
money_multiplier = process_s.str.extract(r'撃破時お金x(\d+\.?\d*)').astype(float)

# --- ▼ 2. 詳細な効果（effect & ability）ごとにループし、関連情報を専用列に格納 ▼ ---
effect_patterns = {'攻撃力低下': r'攻撃力\d+\.?\d*[%％]に低下', '動きを止める': r'動きを止める', '動きを遅くする': r'動きを遅くする', 'ふっとばす': r'ふっとばす', '呪い': r'呪い', '攻撃無効': r'攻撃無効'}
ability_patterns = {'渾身の一撃': r'渾身の一撃', '攻撃力上昇': r'攻撃力上昇', '生き残る': r'生き残る', '撃破時お金': r'撃破時お金', '爆波': r'爆波', '波動': r'(?<!小)波動', '小波動': r'小波動', '烈波': r'(?<!小)烈波', '小烈波': r'小烈波', 'クリティカル': r'クリティカル', '悪魔シールド貫通': r'悪魔シールド貫通', 'バリアブレイク': r'バリアブレイク', 'ゾンビキラー': r'ゾンビキラー', '敵城': r'敵城'}

# 2c. 妨害系(effect)のループ処理
for effect, pattern in effect_patterns.items():
    mask = process_s.str.contains(pattern, regex=True, na=False)
    if not mask.any(): continue
    results_df.loc[mask, f'{effect}:確率'] = probability[mask]
    results_df.loc[mask, f'{effect}:効果時間_Max'] = duration_max[mask]
    if effect == '攻撃力低下':
        results_df.loc[mask, f'{effect}:倍率'] = debuff_rate[mask]

# 2d. 能力系(ability)のループ処理
for ability, pattern in ability_patterns.items():
    mask = process_s.str.contains(pattern, regex=True, na=False)
    if not mask.any(): continue
    
    results_df.loc[mask, f'{ability}:確率'] = probability[mask]
    
    if ability in ['渾身の一撃', '攻撃力上昇']:
        results_df.loc[mask, f'{ability}:与ダメ係数'] = damage_multiplier[mask]
        
    # ★修正点: 波動・爆波の範囲処理を削除★
    if ability in ['爆波', '波動', '小波動', '烈波', '小烈波']:
        # 射程が範囲形式 ('X~Y') か単一形式 ('X') かを事前に判定
        mask_has_range_xy = mask & process_s.str.contains(r'射程\s*-?[\d.]+～-?[\d.]+', na=False)
        mask_has_single_range = mask & process_s.str.contains(r'射程\s*(-?[\d.]+)(?!～)', na=False)

        if ability in ['波動', '小波動']:
            # 波動・小波動の特殊ルール:
            # 射程_Min は常に -67.5 に固定
            results_df.loc[mask, f'{ability}:射程_Min'] = -67.5

            # 射程_Max はテキストから抽出する
            # 波動系のルール: 単一の射程値のみ処理
            if mask_has_single_range.any():
                relevant_mask_single = mask & mask_has_single_range
                results_df.loc[relevant_mask_single, f'{ability}:射程_Max'] = single_range_value[relevant_mask_single]

        elif ability == '爆波':
            # 爆波のルール: 単一の射程値のみ処理
            if mask_has_single_range.any():
                relevant_mask_single = mask & mask_has_single_range
                results_df.loc[relevant_mask_single, f'{ability}:射程'] = single_range_value[relevant_mask_single]

        elif ability in ['烈波', '小烈波']: # 烈波と小烈波の場合のみ
            # 烈波・小烈波のルール: 射程は通常抽出、効果時間も抽出
            if mask_has_range_xy.any():
                relevant_mask_xy = mask & mask_has_range_xy
                results_df.loc[relevant_mask_xy, f'{ability}:射程_Min'] = range_min_from_xy[relevant_mask_xy]
                results_df.loc[relevant_mask_xy, f'{ability}:射程_Max'] = range_max_from_xy[relevant_mask_xy]
            
            # 烈波・小烈波の場合のみ「効果時間_Max」を設定
            results_df.loc[mask, f'{ability}:効果時間_Max'] = duration_max[mask]
        
    if ability == '撃破時お金':
        results_df.loc[mask, f'{ability}:倍率'] = money_multiplier[mask]
    if ability == '敵城':
        results_df.loc[mask, f'{ability}:与ダメ係数'] = damage_multiplier[mask]

# --- ▼ 3 & 4. ブール値のチェックと最終処理 ▼ ---
booleans = {'めっぽう強い': 'めっぽう強い', '打たれ強い': '打たれ強い', '超打たれ強い': '超打たれ強い', '超ダメージ': '超ダメージ', '極ダメージ': '極ダメージ', 'ターゲット限定': 'のみに攻撃', '魂攻撃': '魂攻撃', 'メタルキラー': 'メタルキラー', '被ダメージ1': r'被ダメージ1', '波動ストッパー': '波動ストッパー', '烈波カウンター': '烈波カウンター', '1回攻撃': '1回攻撃'}
for col_name, search_pattern in booleans.items():
    results_df[col_name] = s.str.contains(search_pattern, na=False)
agg_dict = {col: 'any' if results_df[col].dtype == 'bool' else 'first' for col in results_df.columns}
grouped_results = results_df.groupby(results_df.index).agg(agg_dict)
df = df.join(grouped_results)
base_cols = ['特性', '発動条件'] + list(booleans.keys())
effect_cols = sorted([col for col in df.columns if ':' in col])
final_cols = base_cols + effect_cols
existing_cols = [col for col in final_cols if col in df.columns]
df = df[existing_cols]








# --- スライダー処理関数 ---
def add_slider(df, col):
    min_val = df[col].dropna().min()
    max_val = df[col].dropna().max()
    if pd.isna(min_val) or pd.isna(max_val):
        st.sidebar.write(f'{col}: データなし')
        return df
    min_val, max_val = int(min_val), int(max_val)
    if min_val == max_val:
        st.sidebar.write(f'{col}: {min_val}')
        return df
    step = max(int((max_val - min_val) / 100), 1)
    range_val = st.sidebar.slider(f'{col}', min_val, max_val, (min_val, max_val), step=step)
    filtered_df = df.dropna(subset=[col])
    filtered_df = filtered_df[filtered_df[col].between(*range_val)]
    return df[df.index.isin(filtered_df.index)]

# --- グラフ表示関数 ---
def show_comparison_bar_chart(selected_data, current_max_data, current_min_data, item_order):
    chart_data = []
    for item in item_order:
        if item in selected_data:
            value = selected_data[item]
            if pd.notna(value) and item != '範囲':
                max_val = current_max_data.get(item)
                normalized_value = (value / max_val * 100) if max_val and max_val > 0 else 0
                chart_data.append({
                    '項目': item,
                    '値': value,
                    '正規化された値': normalized_value,
                    '表示中の最大値': max_val,
                    '表示中の最小値': current_min_data.get(item),
                })
    if not chart_data:
        st.write("表示するグラフデータがありません。")
        return

    df_chart = pd.DataFrame(chart_data)
    sort_order = df_chart['項目'].tolist()

    background = alt.Chart(df_chart).mark_bar(color='#e0e0e0', cornerRadius=3).encode(
        x=alt.X('max(正規化された値):Q', scale=alt.Scale(domain=[0, 100]), title="表示中の最大値に対する割合 (%)", axis=alt.Axis(format='%')),
        y=alt.Y('項目:N', sort=sort_order, title=None),
        tooltip=[alt.Tooltip('項目:N'), alt.Tooltip('値:Q', title='実際の値', format=','), alt.Tooltip('表示中の最大値:Q', format=','), alt.Tooltip('表示中の最小値:Q', format=',')]
    ).transform_calculate(正規化された値="100")
    
    foreground = alt.Chart(df_chart).mark_bar(cornerRadius=3).encode(
        x='正規化された値:Q',
        y=alt.Y('項目:N', sort=sort_order, title=None),
        color=alt.condition(
            "datum.項目 == '攻撃力' || datum.項目 == 'DPS'",
            alt.value('#d62728'),
            alt.value('#1f77b4')
        ),
        tooltip=[
            alt.Tooltip('項目:N'), 
            alt.Tooltip('値:Q', title='実際の値', format=','), 
            alt.Tooltip('表示中の最大値:Q', format=','), 
            alt.Tooltip('表示中の最小値:Q', format=',')
        ]
    )
    
    chart = (background + foreground).properties(
        height=alt.Step(30)
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    ).configure_legend(
        disable=True
    )

    st.altair_chart(chart, use_container_width=True)

# --- タブ切り替え ---
page = st.radio("tab", ["Cats", "Enemy"], horizontal=True, label_visibility="collapsed")

# --- Catsタブ ---
if page == "Cats":
    df
    st.sidebar.title("Cats フィルター")
    
    # フィルタリング処理...
    own = st.sidebar.checkbox('own'); search = st.sidebar.text_input("キャラクター名")
    if own: df = df[df['own'] > 0]
    if search: df = df[df['キャラクター名'].str.contains(search, na=False)]
    types_options = ['赤', '浮', '黒', 'メタル', '天使', 'エイリアン', 'ゾンビ', '古代種', '悪魔', '無属性']
    types = st.segmented_control('対象属性', types_options, selection_mode='multi')
    if types:
        mask_types = pd.Series(True, index=df.index)
        for t in types: mask_types &= (df[t].fillna(0) > 0)
        df = df[mask_types]
    rank = st.sidebar.multiselect('ランク', ['基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア'])
    if rank: df = df[df['ランク'].isin(rank)]
    attack_range = st.sidebar.multiselect('単体or範囲', ['単体', '範囲'], default=['単体', '範囲'])
    if attack_range: df = df[df['範囲'].isin(attack_range)]
    effect = st.sidebar.multiselect('特殊効果', ['めっぽう強い', '打たれ強い', '超打たれ強い', '超ダメージ', '極ダメージ', 'ターゲット限定', '攻撃力低下', '動きを止める', '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効'])
    if effect:
        mask_effect = pd.Series(True, index=df.index)
        for e in effect:
            df[e] = pd.to_numeric(df[e], errors='coerce')
            mask_effect &= (df[e].fillna(0) > 0)
        df = df[mask_effect]
    ability = st.sidebar.multiselect('特殊能力', ['超生命体特効', '超獣特効', '超賢者特効','爆波', '波動', '小波動', '烈波', '小烈波', 'クリティカル', '渾身の一撃', '攻撃力アップ', '悪魔シールドブレイク', 'バリアブレイク', 'ゾンビキラー', '魂攻撃', 'メタルキラー', '敵城', '撃破時お金', '生き残る', '被ダメージ1', '波動ストッパー', '烈波カウンター', '波動無効', '烈波無効', '爆波無効', '毒撃無効', '攻撃力低下無効', '止める無効', '遅くする無効', 'ふっとばす無効', 'ワープ無効', '古代の呪い無効', '1回攻撃', '遠方範囲', '全方位', '遠方攻撃', '連続攻撃'])
    if ability:
        mask_ability = pd.Series(True, index=df.index)
        for a in ability:
            df[a] = pd.to_numeric(df[a], errors='coerce')
            mask_ability &= (df[a].fillna(0) > 0)
        df = df[mask_ability]
    slider_settings = ['コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB']
    for col_slider in slider_settings:
        if col_slider in df.columns: df = add_slider(df, col_slider)

    # 表示処理...
    st.header("Cats DB")
    if not df.empty:
        df_current_max = df[numeric_cols_cats].max()
        df_current_min = df[numeric_cols_cats].min()
        # 表示する列を、あらかじめ定義した列に限定する
        base_column_order = target_cols_cats
        other_columns = [col for col in df.columns.tolist() if col not in base_column_order ]
        final_column_order = base_column_order + other_columns
        display_columns = [col for col in final_column_order if col in df.columns]
        df_display = df[display_columns]
        
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_default_column(suppressMenu=True, filter=False)
        gb.configure_selection(selection_mode="single")
        grid_options = gb.build()

        # ★★★ 修正箇所1 ★★★
        grid_response = AgGrid(
            df_display,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True  # カラム幅を自動調整するオプションを追加
        )
        
        selected_rows = grid_response['selected_rows']
        if selected_rows is not None and len(selected_rows) > 0:
            selected = pd.DataFrame(selected_rows).iloc[0]
            st.subheader(f"📊 {selected['キャラクター名']} のステータス")
            selected_numeric = {k: v for k, v in selected.items() if k in target_cols_cats}
            show_comparison_bar_chart(selected_numeric, df_current_max, df_current_min, target_cols_cats)
        else:
            st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")
    else:
        st.warning("この条件に一致するキャラクターはいません。")

# --- Enemyタブ ---
elif page == "Enemy":
    df_e = df_e_orig.copy()
    st.sidebar.title("Enemy フィルター")
    search = st.sidebar.text_input("敵キャラクター名")
    if search: df_e = df_e[df_e['キャラクター名'].str.contains(search, na=False)]
    
    st.header("Enemy DB")
    if not df_e.empty:
        df_e_current_max = df_e[target_cols_enemy].max()
        df_e_current_min = df_e[target_cols_enemy].min()
        
        gb = GridOptionsBuilder.from_dataframe(df_e)
        gb.configure_default_column(suppressMenu=True, filter=False)
        gb.configure_selection(selection_mode="single")
        grid_options = gb.build()
        
        # ★★★ 修正箇所2 ★★★
        grid_response = AgGrid(
            df_e,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True  # カラム幅を自動調整するオプションを追加
        )
        
        selected_rows = grid_response['selected_rows']
        if selected_rows is not None and len(selected_rows) > 0:
            selected = pd.DataFrame(selected_rows).iloc[0]
            st.subheader(f"📊 {selected['キャラクター名']} のステータス")
            selected_numeric = {k: v for k, v in selected.items() if k in target_cols_enemy}
            show_comparison_bar_chart(selected_numeric, df_e_current_max, df_e_current_min, target_cols_enemy)
    else:
        st.warning("この条件に一致する敵キャラクターはいません。")