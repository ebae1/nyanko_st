import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from typing import List, Dict, Tuple

# === 定数定義 ===
CATS_DATA_PATH = './0.datafiles/org_catsdb.xlsx'
ENEMY_DATA_PATH = './0.datafiles/nyanko_enemy_db.xlsx'

NUMERIC_COLUMNS_CATS: List[str] = [
    'Own', 'No.', 'コスト', '再生産F', '速度', '射程', '発生F',
    '攻撃力', '頻度F', 'DPS', '体力', 'KB'
]

NUMERIC_COLUMNS_ENEMY: List[str] = [
    '体力', 'KB', '速度', '攻撃力', 'DPS', '頻度F', '攻発F', '射程', 'お金'
]

DISPLAY_COLUMNS_CATS: List[str] = [
    'Own', 'No.', 'ランク', 'キャラクター名', 'コスト', '再生産F',
    '速度', '範囲', '射程', '発生F', '攻撃力', '頻度F', 'DPS',
    '体力', 'KB', '特性'
]

COLOR_TRAITS: List[str] = [
    '赤', '浮', '黒', 'メタル', '天使', 'エイリアン',
    'ゾンビ', '古代種', '悪魔', '白'
]

BOOLEAN_TRAITS: Dict[str, str] = {
    'めっぽう強い': 'めっぽう強い',
    '打たれ強い': '打たれ強い',
    '超打たれ強い': '超打たれ強い',
    '超ダメージ': '超ダメージ',
    '極ダメージ': '極ダメージ',
    'ターゲット限定': 'のみに攻撃',
    '魂攻撃': '魂攻撃',
    'メタルキラー': 'メタルキラー',
    '被ダメージ1': r'被ダメージ\s*1',
    '波動ストッパー': '波動ストッパー',
    '烈波カウンター': '烈波カウンター',
    '1回攻撃': '1回攻撃',
    'ゾンビキラー': 'ゾンビキラー',
    'バリアブレイク': 'バリアブレイク',
    '悪魔シールド貫通': '悪魔シールド貫通',
}

FLAG_TRAITS: List[str] = [
    '攻撃力低下', '動きを止める', '動きを遅くする', 'ふっとばす',
    '呪い', '攻撃無効', '渾身の一撃', '攻撃力上昇', '生き残る',
    'クリティカル', '波動', '小波動', '烈波', '小烈波', '爆波',
]

ENEMY_COL_ORDER: List[str] = [
    '属性','射程','キャラクター名','速度','範囲','DPS','攻撃力','頻度F','攻発F','体力','KB','お金','特性','No.',
]

# === データ読み込み・処理関数 ===

@st.cache_data
def load_cats_data() -> pd.DataFrame:
    """
    Catsデータを読み込み、数値カラム変換と特性フラグ列を追加
    """
    df = pd.read_excel(CATS_DATA_PATH, index_col=0).dropna(axis=0, how='all').dropna(axis=1, how='all')
    for col in NUMERIC_COLUMNS_CATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if '特性' not in df.columns or df['特性'].isnull().all():
        return df

    trait_lines = df['特性'].str.split('\n').explode().str.strip()
    traits_flags = pd.DataFrame(index=trait_lines.index)

    for color in COLOR_TRAITS:
        pattern = rf'対(?!.*全敵.*{color}.*除く).*{color}.*'
        traits_flags[color] = trait_lines.str.contains(pattern, na=False)

    for trait_name, regex in BOOLEAN_TRAITS.items():
        traits_flags[trait_name] = trait_lines.str.contains(regex, na=False, regex=True)

    for flag_trait in FLAG_TRAITS:
        traits_flags[flag_trait] = trait_lines.str.contains(flag_trait, na=False)

    aggregated_flags = traits_flags.groupby(traits_flags.index).any()
    df = df.join(aggregated_flags)

    all_traits = list(BOOLEAN_TRAITS.keys()) + FLAG_TRAITS + COLOR_TRAITS
    for trait in all_traits:
        if trait not in df.columns:
            df[trait] = False

    return df

@st.cache_data
def load_enemy_data() -> pd.DataFrame:
    """
    Enemyデータを読み込み、数値カラムを適切な型に変換
    """
    df = pd.read_excel(ENEMY_DATA_PATH, index_col=0).dropna(axis=0, how='all').dropna(axis=1, how='all')
    for col in NUMERIC_COLUMNS_ENEMY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# === フィルタリング関数群 ===

def filter_numeric_range(
    df: pd.DataFrame,
    column: str,
    sidebar_label_prefix: str = ""
) -> pd.DataFrame:
    """
    サイドバーのスライダーで指定範囲の行だけを抽出
    """
    if column not in df.columns:
        return df

    series = df[column].dropna()
    if series.empty:
        return df

    min_val, max_val = int(series.min()), int(series.max())
    if min_val == max_val:
        return df

    step = max((max_val - min_val) // 100, 1)
    label = f"{sidebar_label_prefix}{column}" if sidebar_label_prefix else column

    selected_min, selected_max = st.sidebar.slider(
        label=label,
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=step
    )

    return df[df[column].between(selected_min, selected_max)]

def filter_checkbox_column(
    df: pd.DataFrame,
    column: str,
    is_checked: bool
) -> pd.DataFrame:
    """
    チェックボックスオン時、カラムが0より大きい行のみに絞る
    """
    if is_checked and column in df.columns:
        return df[df[column] > 0]
    return df

def filter_text_search(
    df: pd.DataFrame,
    column: str,
    search_text: str
) -> pd.DataFrame:
    """
    指定カラムのテキスト検索で絞り込み
    """
    if search_text and column in df.columns:
        return df[df[column].str.contains(search_text, na=False)]
    return df

def filter_multi_flag_selection(
    df: pd.DataFrame,
    selected_flags: List[str]
) -> pd.DataFrame:
    """
    フラグのAND条件で行を絞り込み。選択なしの場合は元のDFを返す
    """
    if not selected_flags:
        return df

    mask = pd.Series(True, index=df.index)
    for flag in selected_flags:
        if flag in df.columns:
            mask &= df[flag]

    return df[mask]

# === 可視化関数 ===

def draw_status_comparison(
    row: pd.Series,
    max_vals: pd.Series,
    min_vals: pd.Series,
    items: List[str],
) -> None:
    """
    選択行の数値項目を最大値に対する割合で棒グラフ表示
    """
    chart_data = []
    numeric_items = [i for i in items 
                     if (i in NUMERIC_COLUMNS_CATS or i in NUMERIC_COLUMNS_ENEMY) and i != 'Own']

    for item in numeric_items:
        value = row.get(item)
        if pd.notna(value):
            max_val = max_vals.get(item, 0)
            min_val = min_vals.get(item, None)
            normalized = (value / max_val * 100) if max_val > 0 else 0
            chart_data.append({
                '項目': item,
                '値': value,
                '正規化値': normalized,
                '最大値': max_val,
                '最小値': min_val,
            })

    if not chart_data:
        st.write("表示できるデータがありません。")
        return

    df_chart = pd.DataFrame(chart_data)
    sort_order = df_chart['項目'].tolist()

    # 項目ごとの色をあらかじめ定義
    color_mapping = {
        '攻撃力': '#d62728',   # 赤
        'DPS': '#d62728',      # 赤
        '再生産F': '#6fb66b',  # 緑
        '頻度F': '#bea557',    # 黄土色
        '発生F': '#e9e8ae',    # 薄黄
        # その他の項目は以下のdefault_colorに
    }
    default_color = '#1f77b4'  # 青

    foreground = alt.Chart(df_chart).mark_bar(cornerRadius=3).encode(
        x='正規化値:Q',
        y=alt.Y('項目:N', sort=sort_order, title=None),
        color=alt.Color(
            '項目:N',
            scale=alt.Scale(
                domain=list(color_mapping.keys()),
                range=list(color_mapping.values()),
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip('項目:N'),
            alt.Tooltip('値:Q', format=','),
            alt.Tooltip('最大値:Q', format=','),
            alt.Tooltip('最小値:Q', format=','),
        ],
    )

    background = alt.Chart(df_chart).mark_bar(
        color='#e0e0e0', cornerRadius=3
    ).encode(
        x=alt.X('max(正規化値):Q', scale=alt.Scale(domain=[0, 100]), title='最大値に対する割合(%)'),
        y=alt.Y('項目:N', sort=sort_order, title=None),
        tooltip=[
            alt.Tooltip('項目:N'),
            alt.Tooltip('値:Q', format=','),
            alt.Tooltip('最大値:Q', format=','),
            alt.Tooltip('最小値:Q', format=','),
        ],
    ).transform_calculate(正規化値='100')

    chart = (background + foreground).properties(
        height=alt.Step(30)
    ).configure_axis(grid=False).configure_view(strokeWidth=0).configure_legend(disable=True)

    st.altair_chart(chart, use_container_width=True)


def get_numeric_columns_max_min(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """
    指定数値カラムの最大・最小値を取得
    """
    valid_cols = [c for c in numeric_cols if c in df.columns]
    if not valid_cols:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return df[valid_cols].max(), df[valid_cols].min()

# === メイン関数 ===

def main() -> None:
    st.set_page_config(layout="wide")

    cats_df = load_cats_data()
    enemy_df = load_enemy_data()

    selected_tab = st.radio("tab", options=["Cats", "Enemy"], horizontal=True, label_visibility="collapsed")

    if selected_tab == "Cats":
        with st.sidebar:
            st.title("Cats フィルター")
            filter_own_only = st.checkbox("Own")
            search_name = st.text_input("キャラクター名")
            
            selected_ranks = st.multiselect('ランク', ['基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア'])
            selected_ranges = st.multiselect('単体or範囲', ['単体', '範囲'], default=['単体', '範囲'])
            selected_effects = st.multiselect('特殊効果', [
                'めっぽう強い', '打たれ強い', '超ダメージ', '攻撃力低下', '動きを止める',
                '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効'])
            selected_abilities = st.multiselect('特殊能力', [
                '波動', '小波動', '烈波', '小烈波', '爆波',
                'クリティカル', '渾身の一撃', 'ゾンビキラー', '悪魔シールド貫通', 'バリアブレイク',
                '生き残る', '波動ストッパー'])
            
        selected_colors = st.segmented_control('対象属性', COLOR_TRAITS,selection_mode='multi')
        filtered_df = cats_df.copy()
        filtered_df = filter_checkbox_column(filtered_df, 'Own', filter_own_only)
        filtered_df = filter_text_search(filtered_df, 'キャラクター名', search_name)
        filtered_df = filter_multi_flag_selection(filtered_df, selected_colors)
        if selected_ranks:
            filtered_df = filtered_df[filtered_df['ランク'].isin(selected_ranks)]
        if selected_ranges:
            filtered_df = filtered_df[filtered_df['範囲'].isin(selected_ranges)]
        filtered_df = filter_multi_flag_selection(filtered_df, selected_effects)
        filtered_df = filter_multi_flag_selection(filtered_df, selected_abilities)

        slider_columns = ['コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB']
        for col in slider_columns:
            filtered_df = filter_numeric_range(filtered_df, col)

        st.header("Cats DB")

        if filtered_df.empty:
            st.warning("この条件に一致するキャラクターはいません。")
            return

        max_vals, min_vals = get_numeric_columns_max_min(filtered_df, NUMERIC_COLUMNS_CATS)
        visible_columns = [col for col in DISPLAY_COLUMNS_CATS if col in filtered_df.columns]
        display_df = filtered_df[visible_columns]

        grid_builder = GridOptionsBuilder.from_dataframe(display_df)
        grid_builder.configure_default_column(suppressMenu=True)
        grid_builder.configure_selection(selection_mode='single')
        if 'キャラクター名' in display_df.columns:
            grid_builder.configure_column('キャラクター名', minWidth=150)
        if '特性' in display_df.columns:
            grid_builder.configure_column('特性', minWidth=300, wrapText=True, autoHeight=True)
        for col_name in ['ランク', '範囲', 'KB', 'No.', 'Own', '速度']:
            if col_name in display_df.columns:
                grid_builder.configure_column(col_name, initialWidth=100)
        grid_options = grid_builder.build()

        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
        )

        selected_rows = grid_response.get('selected_rows', [])
        if selected_rows is not None and len(selected_rows) > 0:
            selected_series = pd.DataFrame(selected_rows).iloc[0]
            character_name = selected_series.get('キャラクター名', '')
            st.subheader(f"📊 {character_name} のステータス")
            draw_status_comparison(selected_series, max_vals, min_vals, visible_columns)
        else:
            st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")

    elif selected_tab == "Enemy":
        with st.sidebar:
            st.title("Enemy フィルター")
            search_enemy_name = st.text_input("敵キャラクター名")

        filtered_enemy_df = enemy_df.copy()
        filtered_enemy_df = filter_text_search(filtered_enemy_df, 'キャラクター名', search_enemy_name)

        st.header("Enemy DB")
        if filtered_enemy_df.empty:
            st.warning("この条件に一致する敵キャラクターはいません。")
            return

        max_vals, min_vals = get_numeric_columns_max_min(filtered_enemy_df, NUMERIC_COLUMNS_ENEMY)
        columns_order = ENEMY_COL_ORDER
        columns_to_show = [col for col in columns_order if col in filtered_enemy_df.columns]
        filtered_enemy_df = filtered_enemy_df[columns_to_show]
        
        grid_builder = GridOptionsBuilder.from_dataframe(filtered_enemy_df)
        grid_builder.configure_default_column(suppressMenu=True, filter=False)
        grid_builder.configure_selection(selection_mode='single')
        grid_options = grid_builder.build()

        grid_response = AgGrid(
            filtered_enemy_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
        )

        selected_rows = grid_response.get('selected_rows', [])
        if selected_rows is not None and len(selected_rows) > 0:
            selected_series = pd.DataFrame(selected_rows).iloc[0]
            enemy_name = selected_series.get('キャラクター名', '')
            st.subheader(f"📊 {enemy_name} のステータス")
            draw_status_comparison(selected_series, max_vals, min_vals, NUMERIC_COLUMNS_ENEMY)

if __name__ == "__main__":
    main()
