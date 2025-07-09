import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
from decimal import Decimal, InvalidOperation

# --- ページ設定 ---
st.set_page_config(layout='wide')

# --- 1. 特性解析とデータ前処理を行う関数 ---
# この関数全体の結果がキャッシュされることで、アプリのパフォーマンスが大幅に向上します。
@st.cache_data
def load_processed_cats_data():
    """
    Excelからネコデータを読み込み、特性解析を行い、
    数値変換まで済ませたDataFrameを返す関数。
    """
    df = pd.read_excel('./catsdb.xlsx', index_col=0)
    
    # --- データクレンジング（数値変換）---
    # ★修正点1: 数値に変換する列を明示的に指定し、エラーを防ぐ
    numeric_cols_to_convert = [
        'own', 'No.', 'コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB'
    ]
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 特性解析ロジック ---
    src_col = '特性'
    # 特性列がない、またはすべて空の場合は、解析せずにスキップ
    if src_col not in df.columns or df[src_col].isnull().all():
        return df

    # 元のdfの列を保持するため、解析結果は別の一時的なDataFrameに格納
    df_exploded = df.assign(line=df[src_col].str.split('\n')).explode('line')
    s = df_exploded['line'].astype(str).str.strip()
    results_df = pd.DataFrame(index=s.index)

    # --- 特性解析 ---
    # Boolean(True/False)で判定する特性
    boolean_effects = {
        'めっぽう強い': 'めっぽう強い', '打たれ強い': '打たれ強い', '超打たれ強い': '超打たれ強い', 
        '超ダメージ': '超ダメージ', '極ダメージ': '極ダメージ', 'ターゲット限定': 'のみに攻撃', 
        '魂攻撃': '魂攻撃', 'メタルキラー': 'メタルキラー', '被ダメージ1': r'被ダメージ\s*1', 
        '波動ストッパー': '波動ストッパー', '烈波カウンター': '烈波カウンター', '1回攻撃': '1回攻撃',
        'ゾンビキラー': 'ゾンビキラー', 'バリアブレイク': 'バリアブレイク', '悪魔シールド貫通': '悪魔シールド貫通'
        # ... 他のBoolean特性もここに追加 ...
    }
    for col_name, search_pattern in boolean_effects.items():
        results_df[col_name] = s.str.contains(search_pattern, na=False, regex=True)

    # 存在フラグを立てる特性（フィルタリング用）
    # ★修正点2: フィルタリングしやすいように、特性の有無をTrue/Falseで持つ列を作成
    flag_effects = [
        '攻撃力低下', '動きを止める', '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効',
        '渾身の一撃', '攻撃力上昇', '生き残る', 'クリティカル', '波動', '小波動', '烈波', '小烈波', '爆波'
        # ... 他のフラグを立てたい特性もここに追加 ...
    ]
    for effect in flag_effects:
        results_df[effect] = s.str.contains(effect, na=False)

    # --- 集計と結合 ---
    # ★修正点3: 元の列を失わないように、生成した列だけを集計して結合
    agg_dict = {col: 'any' for col in results_df.columns}
    grouped_results = results_df.groupby(results_df.index).agg(agg_dict)
    
    final_df = df.join(grouped_results)
    
    # 結合後に生成されなかった列をFalseで埋める
    all_generated_cols = list(boolean_effects.keys()) + flag_effects
    for col in all_generated_cols:
        if col not in final_df.columns:
            final_df[col] = False
            
    return final_df

@st.cache_data
def load_enemy_data():
    """Excelから敵データを読み込み、数値変換まで済ませたDataFrameを返す関数。"""
    df = pd.read_excel('./nyanko_enemy_db.xlsx', index_col=0)
    target_cols_enemy = ['体力','KB','速度','攻撃力','DPS','頻度F','攻発F','射程','お金']
    for col in target_cols_enemy:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- 2. データのロード ---
# アプリ起動時に一度だけ（またはデータが変わった時だけ）実行される
df_processed_orig = load_processed_cats_data()
df_e_orig = load_enemy_data()


# --- 3. 共通関数とUI要素の定義 ---
target_cols_cats_display = ['own','No.','ランク','キャラクター名','コスト','再生産F','速度','範囲','射程','発生F','攻撃力','頻度F','DPS','体力','KB','特性']
numeric_cols_cats = [col for col in target_cols_cats_display if col not in ['範囲', '特性', 'ランク', 'キャラクター名']]
target_cols_enemy = ['体力','KB','速度','攻撃力','DPS','頻度F','攻発F','射程','お金']

# (スライダー関数とグラフ表示関数は変更なしのため省略)
def add_slider(df, col):
    min_val = df[col].dropna().min()
    max_val = df[col].dropna().max()
    if pd.isna(min_val) or pd.isna(max_val): return df
    min_val, max_val = int(min_val), int(max_val)
    if min_val == max_val: return df
    step = max(int((max_val - min_val) / 100), 1)
    range_val = st.sidebar.slider(f'{col}', min_val, max_val, (min_val, max_val), step=step)
    filtered_df = df.dropna(subset=[col])
    filtered_df = filtered_df[filtered_df[col].between(*range_val)]
    return df[df.index.isin(filtered_df.index)]

def show_comparison_bar_chart(selected_data, current_max_data, current_min_data, item_order):
    chart_data = []
    # グラフ描画対象を数値のみに絞る
    item_order_numeric = [item for item in item_order if item in numeric_cols_cats or item in target_cols_enemy]
    for item in item_order_numeric:
        if item in selected_data:
            value = selected_data[item]
            if pd.notna(value):
                max_val = current_max_data.get(item)
                normalized_value = (value / max_val * 100) if max_val and max_val > 0 else 0
                chart_data.append({'項目': item, '値': value, '正規化された値': normalized_value, '表示中の最大値': max_val, '表示中の最小値': current_min_data.get(item)})
    if not chart_data:
        st.write("表示するグラフデータがありません。")
        return
    df_chart = pd.DataFrame(chart_data)
    sort_order = df_chart['項目'].tolist()
    background = alt.Chart(df_chart).mark_bar(color='#e0e0e0', cornerRadius=3).encode(x=alt.X('max(正規化された値):Q', scale=alt.Scale(domain=[0, 100]), title="表示中の最大値に対する割合 (%)", axis=alt.Axis(format='%')), y=alt.Y('項目:N', sort=sort_order, title=None), tooltip=[alt.Tooltip('項目:N'), alt.Tooltip('値:Q', title='実際の値', format=','), alt.Tooltip('表示中の最大値:Q', format=','), alt.Tooltip('表示中の最小値:Q', format=',')] ).transform_calculate(正規化された値="100")
    foreground = alt.Chart(df_chart).mark_bar(cornerRadius=3).encode(x='正規化された値:Q', y=alt.Y('項目:N', sort=sort_order, title=None), color=alt.condition("datum.項目 == '攻撃力' || datum.項目 == 'DPS'", alt.value('#d62728'), alt.value('#1f77b4')), tooltip=[alt.Tooltip('項目:N'), alt.Tooltip('値:Q', title='実際の値', format=','), alt.Tooltip('表示中の最大値:Q', format=','), alt.Tooltip('表示中の最小値:Q', format=',')] )
    chart = (background + foreground).properties(height=alt.Step(30)).configure_axis(grid=False).configure_view(strokeWidth=0).configure_legend(disable=True)
    st.altair_chart(chart, use_container_width=True)


# --- 4. メインのページ表示ロジック ---
page = st.radio("tab", ["Cats", "Enemy"], horizontal=True, label_visibility="collapsed")

if page == "Cats":
    # ★最重要修正: 毎回、加工済みのオリジナルデータからコピーしてフィルタリングを開始する
    df = df_processed_orig.copy()
    
    st.sidebar.title("Cats フィルター")
    
    # --- フィルタリング処理 ---
    own = st.sidebar.checkbox('own')
    search = st.sidebar.text_input("キャラクター名")
    
    if own: df = df[df['own'] > 0]
    if search: df = df[df['キャラクター名'].str.contains(search, na=False)]

    types_options = ['赤', '浮', '黒', 'メタル', '天使', 'エイリアン', 'ゾンビ', '古代種', '悪魔', '無属性']
    # st.segmented_control はそのまま使用
    types = st.sidebar.multiselect('対象属性', types_options)
    if types:
        mask = pd.Series(True, index=df.index)
        for t in types:
            if t in df.columns:
                mask &= (df[t].fillna(0) > 0)
        df = df[mask]
        
    rank = st.sidebar.multiselect('ランク', ['基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア'])
    if rank: df = df[df['ランク'].isin(rank)]
    
    attack_range = st.sidebar.multiselect('単体or範囲', ['単体', '範囲'], default=['単体', '範囲'])
    if attack_range: df = df[df['範囲'].isin(attack_range)]
    
    # ★修正点4: フィルタリングのロジックを、特性解析で生成した列を使うように修正
    effect_options = ['めっぽう強い', '打たれ強い', '超ダメージ', '攻撃力低下', '動きを止める', '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効']
    effects = st.sidebar.multiselect('特殊効果', effect_options)
    if effects:
        mask = pd.Series(True, index=df.index)
        for e in effects:
            if e in df.columns:
                mask &= (df[e] == True) # 生成したTrue/Falseの列でフィルタ
        df = df[mask]

    ability_options = ['波動', '小波動', '烈波', '小烈波', '爆波', 'クリティカル', '渾身の一撃', 'ゾンビキラー', '生き残る', '波動ストッパー']
    abilities = st.sidebar.multiselect('特殊能力', ability_options)
    if abilities:
        mask = pd.Series(True, index=df.index)
        for a in abilities:
            if a in df.columns:
                mask &= (df[a] == True) # 生成したTrue/Falseの列でフィルタ
        df = df[mask]

    slider_settings = ['コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB']
    for col_slider in slider_settings:
        if col_slider in df.columns: df = add_slider(df, col_slider)

    # --- 表示処理 ---
    st.header("Cats DB")
    if not df.empty:
        df_current_max = df[numeric_cols_cats].max()
        df_current_min = df[numeric_cols_cats].min()
        
        # 表示する列を主要なものに絞る
        display_columns = [col for col in target_cols_cats_display if col in df.columns]
        df_display = df[display_columns]
        
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_default_column(suppressMenu=True)
        gb.configure_selection(selection_mode="single")
        # 最小幅を設定して見やすくする
        gb.configure_column('キャラクター名', minWidth=150)
        gb.configure_column('特性', minWidth=300, wrapText=True, autoHeight=True)
        
        cols_to_set_width1 = ['ランク','範囲','KB','No.','own','速度']
        # ループを使って、リスト内の各列に個別に設定を適用する
        for col_name in cols_to_set_width1:
            if col_name in df_display.columns:
                gb.configure_column(
                    col_name,
                    initialWidth=100
                )
       
        
        grid_options = gb.build()

        grid_response = AgGrid(
            df_display,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            
        )
        
        selected_rows = grid_response['selected_rows']
        if selected_rows is not None and len(selected_rows) > 0:
            selected = pd.DataFrame(selected_rows).iloc[0]
            st.subheader(f"📊 {selected['キャラクター名']} のステータス")
            show_comparison_bar_chart(selected, df_current_max, df_current_min, target_cols_cats_display)
        else:
            st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")
    else:
        st.warning("この条件に一致するキャラクターはいません。")

elif page == "Enemy":
    # (Enemyタブのロジックは変更なしのため、元のコードをそのまま使用)
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
        grid_response = AgGrid(df_e, gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED, allow_unsafe_jscode=True, fit_columns_on_grid_load=True)
        selected_rows = grid_response['selected_rows']
        if selected_rows is not None and len(selected_rows) > 0:
            selected = pd.DataFrame(selected_rows).iloc[0]
            st.subheader(f"📊 {selected['キャラクター名']} のステータス")
            show_comparison_bar_chart(selected, df_e_current_max, df_e_current_min, target_cols_enemy)
    else:
        st.warning("この条件に一致する敵キャラクターはいません。")
