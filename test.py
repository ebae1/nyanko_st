import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from typing import List, Dict, Tuple, Optional

from modules.preprocessing import preprocess_cats_df, preprocess_enemy_df, COLOR_TRAITS

# === 定数定義 ===

CATS_DATA_FILE_PATH = './0.datafiles/org_catsdb.xlsx'
ENEMY_DATA_FILE_PATH = './0.datafiles/nyanko_enemy_db.xlsx'

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

ENEMY_COLUMNS_DISPLAY_ORDER: List[str] = [
    '属性', '射程', 'キャラクター名', '速度', '範囲', 'DPS', '攻撃力',
    '頻度F', '攻発F', '体力', 'KB', 'お金', '特性', 'No.',
]

RATIO_COLUMN_PAIRS: List[Tuple[str, str, Optional[str]]] = [
    ('DPS', 'コスト', None),
    ('体力', 'コスト', None),
    # 追加可
]

# === データ読み込み関数 ===

@st.cache_data
def load_cats_data() -> pd.DataFrame:
    df = pd.read_excel(
        CATS_DATA_FILE_PATH,
        index_col=0
    ).dropna(axis=0, how='all').dropna(axis=1, how='all')
    df = preprocess_cats_df(df, NUMERIC_COLUMNS_CATS, RATIO_COLUMN_PAIRS)
    return df

@st.cache_data
def load_enemy_data() -> pd.DataFrame:
    df = pd.read_excel(
        ENEMY_DATA_FILE_PATH,
        index_col=0
    ).dropna(axis=0, how='all').dropna(axis=1, how='all')
    df = preprocess_enemy_df(df, NUMERIC_COLUMNS_ENEMY)
    return df

# === フィルタリング用関数群 ===

def filter_rows_by_numeric_range(
    df: pd.DataFrame,
    column_name: str,
    sidebar_label_prefix: str = ""
) -> pd.DataFrame:
    if column_name not in df.columns:
        return df
    series = df[column_name].dropna()
    if series.empty:
        return df
    min_val, max_val = int(series.min()), int(series.max())
    if min_val == max_val:
        return df
    step = max((max_val - min_val) // 100, 1)
    slider_label = f"{sidebar_label_prefix}{column_name}" if sidebar_label_prefix else column_name
    selected_min, selected_max = st.sidebar.slider(
        label=slider_label,
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=step
    )
    return df[df[column_name].between(selected_min, selected_max)]

def filter_rows_by_checkbox_flag(
    df: pd.DataFrame,
    column_name: str,
    is_checked: bool
) -> pd.DataFrame:
    if is_checked and column_name in df.columns:
        return df[df[column_name] > 0]
    return df

def filter_rows_by_text_search(
    df: pd.DataFrame,
    column_name: str,
    search_text: str
) -> pd.DataFrame:
    if search_text and column_name in df.columns:
        return df[df[column_name].str.contains(search_text, na=False)]
    return df

def filter_rows_by_multiple_flags(
    df: pd.DataFrame,
    selected_flags: List[str]
) -> pd.DataFrame:
    if not selected_flags:
        return df
    mask = pd.Series(True, index=df.index)
    for flag_column in selected_flags:
        if flag_column in df.columns:
            mask &= df[flag_column]
    return df[mask]

# === グラフ化関数 ===

def draw_comparison_bar_chart(
    selected_row: pd.Series,
    max_values: pd.Series,
    min_values: pd.Series,
    display_items: List[str],
) -> None:
    chart_data = []
    numeric_items = [
        item for item in display_items
        if (item in NUMERIC_COLUMNS_CATS or item in NUMERIC_COLUMNS_ENEMY) and item != 'Own'
    ]
    for item in numeric_items:
        value = selected_row.get(item)
        if pd.notna(value):
            max_val = max_values.get(item, 0)
            min_val = min_values.get(item, None)
            normalized_value = (value / max_val * 100) if max_val > 0 else 0
            chart_data.append({
                '項目': item,
                '値': value,
                '正規化値': normalized_value,
                '最大値': max_val,
                '最小値': min_val,
            })
    if not chart_data:
        st.write("表示できるデータがありません。")
        return
    df_chart = pd.DataFrame(chart_data)
    sort_order = df_chart['項目'].tolist()
    color_mapping = {
        '攻撃力': '#d62728',     # 赤
        'DPS': '#d62728',        # 赤
        '再生産F': '#6fb66b',    # 緑
        '頻度F': "#0ee6d7",      # 黄土色
        '発生F': "#08e0dcb5",    # 薄黄
    }
    bar_foreground = alt.Chart(df_chart).mark_bar(
        cornerRadius=3).encode(
        x='正規化値:Q',
        y=alt.Y('項目:N', sort=sort_order, title=None),
        color=alt.Color(
            '項目:N',
            scale=alt.Scale(domain=list(color_mapping.keys()), range=list(color_mapping.values())),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('項目:N'),
            alt.Tooltip('値:Q', format=','),
            alt.Tooltip('最大値:Q', format=','),
            alt.Tooltip('最小値:Q', format=','),
        ],
    )
    bar_background = alt.Chart(df_chart).mark_bar(
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
    
    #値ラベルを項目名の左側へ追加
    value_labels = alt.Chart(df_chart).mark_text(
        align='left', 
        baseline='middle', 
        dx=5,
        color='white',
    ).encode(
        x=alt.value(0), #x軸の左端に固定
        y=alt.Y('項目:N', sort=sort_order, title=None),
        text=alt.Text('値:Q')
    )
    
    chart = (bar_background + bar_foreground + value_labels
        ).properties(height=alt.Step(30)
        ).configure_axisY(offset=50
        ).configure_axis(grid=False
        ).configure_view(strokeWidth=0
        ).configure_legend(disable=True)
    st.altair_chart(chart, use_container_width=True)

def get_max_min_of_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: List[str]
) -> Tuple[pd.Series, pd.Series]:
    existing_numeric_cols = [col for col in numeric_columns if col in df.columns]
    if not existing_numeric_cols:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    max_values = df[existing_numeric_cols].max()
    min_values = df[existing_numeric_cols].min()
    return max_values, min_values

# === メイン処理関数 ===

def main() -> None:
    st.set_page_config(layout="wide")
    cats_data = load_cats_data()
    enemy_data = load_enemy_data()

    default_display_columns = [ 'キャラクター名', 'コスト', '射程', 'DPS', '体力', 'DPS/コスト', '体力/コスト' ]
    
    numeric_columns_cats_extended = NUMERIC_COLUMNS_CATS.copy()
    display_columns_cats_extended = DISPLAY_COLUMNS_CATS.copy()

    # 比率列名の追加
    ratio_columns = []
    for numerator, denominator, new_col in RATIO_COLUMN_PAIRS:
        col_name = new_col if new_col is not None else f"{numerator}/{denominator}"
        ratio_columns.append(col_name)
    for col in ratio_columns:
        if col not in numeric_columns_cats_extended:
            numeric_columns_cats_extended.append(col)
    for col in ratio_columns:
        if col in cats_data.columns and col not in display_columns_cats_extended:
            display_columns_cats_extended.append(col)

    selected_tab = st.radio(
        label="tab",
        options=["Cats", "Enemy"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if selected_tab == "Cats":
        with st.sidebar:
            st.title("Cats フィルター")
            filter_own_only = st.checkbox("Own")
            
            all_possible_columns = [col for col in cats_data.columns if col in DISPLAY_COLUMNS_CATS or col in ratio_columns]
            selected_display_columns = st.multiselect(
                label='表示項目を選択',
                options=all_possible_columns,
                default=default_display_columns
            )
            
            search_character_name = st.text_input("キャラクター名")
            selected_ranks = st.multiselect(
                'ランク',
                ['基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア']
            )
            selected_ranges = st.multiselect(
                '単体or範囲',
                ['単体', '範囲'],
                default=['単体', '範囲']
            )
            selected_effects = st.multiselect(
                '特殊効果',
                [
                    'めっぽう強い', '打たれ強い', '超ダメージ', '攻撃力低下', '動きを止める',
                    '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効'
                ]
            )
            selected_abilities = st.multiselect(
                '特殊能力',
                [
                    '波動', '小波動', '烈波', '小烈波', '爆波',
                    'クリティカル', '渾身の一撃', 'ゾンビキラー',
                    '悪魔シールド貫通', 'バリアブレイク', '生き残る', '波動ストッパー'
                ]
            )

        selected_colors = st.segmented_control(
            label='対象属性',
            options=COLOR_TRAITS,
            selection_mode='multi'
        )

        filtered_cats_df = cats_data.copy()
        filtered_cats_df = filter_rows_by_checkbox_flag(filtered_cats_df, 'Own', filter_own_only)
        filtered_cats_df = filter_rows_by_text_search(filtered_cats_df, 'キャラクター名', search_character_name)
        filtered_cats_df = filter_rows_by_multiple_flags(filtered_cats_df, selected_colors)
        if selected_ranks:
            filtered_cats_df = filtered_cats_df[filtered_cats_df['ランク'].isin(selected_ranks)]
        if selected_ranges:
            filtered_cats_df = filtered_cats_df[filtered_cats_df['範囲'].isin(selected_ranges)]
        filtered_cats_df = filter_rows_by_multiple_flags(filtered_cats_df, selected_effects)
        filtered_cats_df = filter_rows_by_multiple_flags(filtered_cats_df, selected_abilities)
        numeric_slider_columns = [
            'コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB',
        ]
        for col in ratio_columns:
            if col in filtered_cats_df.columns and col not in numeric_slider_columns:
                numeric_slider_columns.append(col)
        for numeric_col in numeric_slider_columns:
            filtered_cats_df = filter_rows_by_numeric_range(filtered_cats_df, numeric_col)
        st.header("Cats DB")
        if filtered_cats_df.empty:
            st.warning("この条件に一致するキャラクターはいません。")
            return
        max_vals, min_vals = get_max_min_of_numeric_columns(filtered_cats_df, numeric_columns_cats_extended)
        visible_columns = [col for col in selected_display_columns if col in filtered_cats_df.columns]
        display_df = filtered_cats_df[visible_columns]
        grid_builder = GridOptionsBuilder.from_dataframe(display_df)
        grid_builder.configure_default_column(suppressMenu=True)
        grid_builder.configure_selection(selection_mode='single')
        grid_builder.configure_column('キャラクター名', minWidth=150)
        grid_builder.configure_column('特性', minWidth=300, wrapText=True, autoHeight=True)
        for col_name in ['ランク', '範囲', 'KB', 'No.', 'Own', '速度']:
            if col_name in display_df.columns:
                grid_builder.configure_column(col_name, initialWidth=100)
        for col in ratio_columns:
            if col in display_df.columns:
                grid_builder.configure_column(col, valueFormatter="x.toFixed(2)")
        grid_options = grid_builder.build()
        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            key='cats_grid'
        )
        selected_rows = grid_response.get('selected_rows', [])
        if isinstance(selected_rows, pd.DataFrame):
            if not selected_rows.empty:
                selected_series = selected_rows.iloc[0]
            else:
                selected_series = None
        elif isinstance(selected_rows, list):
            if len(selected_rows) > 0:
                selected_series = pd.Series(selected_rows[0])
            else:
                selected_series = None
        else:
            selected_series = None
        if selected_series is not None:
            character_name = selected_series.get('キャラクター名', '')
            st.subheader(f"📊 {character_name} のステータス")
            draw_comparison_bar_chart(selected_series, max_vals, min_vals, visible_columns)
        else:
            st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")

    elif selected_tab == "Enemy":
        with st.sidebar:
            st.title("Enemy フィルター")
            search_enemy_name = st.text_input("敵キャラクター名")
        filtered_enemy_df = enemy_data.copy()
        filtered_enemy_df = filter_rows_by_text_search(filtered_enemy_df, 'キャラクター名', search_enemy_name)
        st.header("Enemy DB")
        if filtered_enemy_df.empty:
            st.warning("この条件に一致する敵キャラクターはいません。")
            return
        max_vals, min_vals = get_max_min_of_numeric_columns(filtered_enemy_df, NUMERIC_COLUMNS_ENEMY)
        visible_enemy_columns = [col for col in ENEMY_COLUMNS_DISPLAY_ORDER if col in filtered_enemy_df.columns]
        filtered_enemy_df = filtered_enemy_df[visible_enemy_columns]
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
            key='enemy_grid'
        )
        selected_rows = grid_response.get('selected_rows', [])
        if isinstance(selected_rows, list) and len(selected_rows) > 0:
            selected_series = pd.DataFrame(selected_rows).iloc[0]
            enemy_name = selected_series.get('キャラクター名', '')
            st.subheader(f"📊 {enemy_name} のステータス")
            draw_comparison_bar_chart(selected_series, max_vals, min_vals, NUMERIC_COLUMNS_ENEMY)

if __name__ == "__main__":
    main()
