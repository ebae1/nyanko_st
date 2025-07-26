import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from typing import List, Dict, Tuple


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

ENEMY_COLUMNS_DISPLAY_ORDER: List[str] = [
    '属性', '射程', 'キャラクター名', '速度', '範囲', 'DPS', '攻撃力',
    '頻度F', '攻発F', '体力', 'KB', 'お金', '特性', 'No.',
]


# === データ読み込み関数 ===

@st.cache_data
def load_cats_data() -> pd.DataFrame:
    """
    Catsデータを読み込み処理
    - 数値カラムの型変換
    - 特性列からフラグ列を追加
    """

    df = pd.read_excel(
        CATS_DATA_FILE_PATH,
        index_col=0
    ).dropna(axis=0, how='all').dropna(axis=1, how='all')

    # 数値カラムを適切な型に
    for numeric_col in NUMERIC_COLUMNS_CATS:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')

    if '特性' not in df.columns or df['特性'].isnull().all():
        return df

    # 特性を1行ずつ分解
    traits_lines = df['特性'].str.split('\n').explode().str.strip()
    traits_flags_df = pd.DataFrame(index=traits_lines.index)

    # 属性カラーによる特性検出（正規表現パターンを用いる）
    for color_trait in COLOR_TRAITS:
        pattern = rf'対(?!.*全敵.*{color_trait}.*除く).*{color_trait}.*'
        traits_flags_df[color_trait] = traits_lines.str.contains(pattern, na=False)

    # 真偽値系特性の検出
    for trait_name, regex_pattern in BOOLEAN_TRAITS.items():
        traits_flags_df[trait_name] = traits_lines.str.contains(regex_pattern, na=False, regex=True)

    # フラグ系特性の検出（単純包含検索）
    for flag_trait in FLAG_TRAITS:
        traits_flags_df[flag_trait] = traits_lines.str.contains(flag_trait, na=False)

    # 行ごとに複数の分解行があるため集約（OR条件）
    aggregated_traits_flags = traits_flags_df.groupby(traits_flags_df.index).any()

    # 元のdfにフラグ列を結合
    df = df.join(aggregated_traits_flags)

    # 全特性列が存在しない場合はデフォルトFalse列を追加
    all_traits = list(BOOLEAN_TRAITS.keys()) + FLAG_TRAITS + COLOR_TRAITS
    for trait in all_traits:
        if trait not in df.columns:
            df[trait] = False

    return df


@st.cache_data
def load_enemy_data() -> pd.DataFrame:
    """
    Enemyデータを読み込み、数値カラムの型を変換する
    """

    df = pd.read_excel(
        ENEMY_DATA_FILE_PATH,
        index_col=0
    ).dropna(axis=0, how='all').dropna(axis=1, how='all')

    for col in NUMERIC_COLUMNS_ENEMY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# === フィルタリング用関数群 ===

def filter_rows_by_numeric_range(
    df: pd.DataFrame,
    column_name: str,
    sidebar_label_prefix: str = ""
) -> pd.DataFrame:
    """
    サイドバーのスライダーで指定された数値範囲に含まれる行だけ抽出する

    Args:
        df: 対象DataFrame
        column_name: 数値カラム名
        sidebar_label_prefix: ラベル接頭辞
    
    Returns:
        フィルタリング後のDataFrame
    """
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
    """
    チェックボックス選択時に、該当カラムの値が0より大きい行のみを残す

    Args:
        df: 対象DataFrame
        column_name: フラグカラム名
        is_checked: チェック状況

    Returns:
        フィルタリング済みDataFrame
    """
    if is_checked and column_name in df.columns:
        return df[df[column_name] > 0]
    return df


def filter_rows_by_text_search(
    df: pd.DataFrame,
    column_name: str,
    search_text: str
) -> pd.DataFrame:
    """
    テキスト検索で指定列に部分一致する行をフィルタリング
    
    Args:
        df: 対象DataFrame
        column_name: 検索対象列名
        search_text: 検索文字列

    Returns:
        フィルタリング済みDataFrame
    """
    if search_text and column_name in df.columns:
        return df[df[column_name].str.contains(search_text, na=False)]
    return df


def filter_rows_by_multiple_flags(
    df: pd.DataFrame,
    selected_flags: List[str]
) -> pd.DataFrame:
    """
    複数のフラグ列に対してAND条件でフィルター

    Args:
        df: 対象DataFrame
        selected_flags: 選択されたフラグ名リスト

    Returns:
        フィルタリング済みDataFrame
    """
    if not selected_flags:
        return df

    mask = pd.Series(True, index=df.index)
    for flag_column in selected_flags:
        if flag_column in df.columns:
            mask &= df[flag_column]

    return df[mask]


# === 可視化関数 ===

def draw_comparison_bar_chart(
    selected_row: pd.Series,
    max_values: pd.Series,
    min_values: pd.Series,
    display_items: List[str],
) -> None:
    """
    選択された行の数値項目を、最大値との比率で棒グラフ表示する

    Args:
        selected_row: 選択された行のpd.Series
        max_values: 各項目の最大値Series
        min_values: 各項目の最小値Series
        display_items: 表示対象の項目名リスト
    """

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

    # 項目ごとの色マッピング
    color_mapping = {
        '攻撃力': '#d62728',     # 赤
        'DPS': '#d62728',        # 赤
        '再生産F': '#6fb66b',    # 緑
        '頻度F': '#bea557',      # 黄土色
        '発生F': '#e9e8ae',      # 薄黄
        # 他は青（default）
    }
    default_color = '#1f77b4'

    bar_foreground = alt.Chart(df_chart).mark_bar(cornerRadius=3).encode(
        x='正規化値:Q',
        y=alt.Y('項目:N', sort=sort_order, title=None),
        color=alt.Color(
            '項目:N',
            scale=alt.Scale(domain=list(color_mapping.keys()), range=list(color_mapping.values())),
            legend=None,
            condition=alt.condition(
                alt.datum.項目,  # dummy condition to preserve colors
                alt.value(default_color),
                alt.value(default_color)
            )
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

    chart = (bar_background + bar_foreground).properties(
        height=alt.Step(30)
    ).configure_axis(grid=False).configure_view(strokeWidth=0).configure_legend(disable=True)

    st.altair_chart(chart, use_container_width=True)


def get_max_min_of_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """
    指定された数値カラムの最大値・最小値Seriesを返す
    
    Args:
        df: 対象DataFrame
        numeric_columns: 数値カラム名リスト

    Returns:
        (最大値Series, 最小値Series)
    """
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
            'コスト', '再生産F', '速度', '射程', '発生F',
            '攻撃力', '頻度F', 'DPS', '体力', 'KB'
        ]
        for numeric_col in numeric_slider_columns:
            filtered_cats_df = filter_rows_by_numeric_range(filtered_cats_df, numeric_col)

        st.header("Cats DB")

        if filtered_cats_df.empty:
            st.warning("この条件に一致するキャラクターはいません。")
            return

        max_vals, min_vals = get_max_min_of_numeric_columns(filtered_cats_df, NUMERIC_COLUMNS_CATS)

        visible_columns = [col for col in DISPLAY_COLUMNS_CATS if col in filtered_cats_df.columns]
        display_df = filtered_cats_df[visible_columns]

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
        if selected_rows and len(selected_rows) > 0:
            selected_series = pd.DataFrame(selected_rows).iloc[0]
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
        )

        selected_rows = grid_response.get('selected_rows', [])
        if selected_rows and len(selected_rows) > 0:
            selected_series = pd.DataFrame(selected_rows).iloc[0]
            enemy_name = selected_series.get('キャラクター名', '')
            st.subheader(f"📊 {enemy_name} のステータス")
            draw_comparison_bar_chart(selected_series, max_vals, min_vals, NUMERIC_COLUMNS_ENEMY)


if __name__ == "__main__":
    main()
