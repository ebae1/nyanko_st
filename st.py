import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from typing import List, Optional, Union, Dict

# --- 定数 ---
CATS_FILE = '../0.datafiles/org_catsdb.csv'
ENEMY_FILE = '../0.datafiles/nyanko_enemy_db.csv'

NUMERIC_COLS_CATS: List[str] = [
    'own', 'No.', 'コスト', '再生産F', '速度', '射程', '発生F',
    '攻撃力', '頻度F', 'DPS', '体力', 'KB'
]

NUMERIC_COLS_ENEMY: List[str] = [
    '体力', 'KB', '速度', '攻撃力', 'DPS', '頻度F', '攻発F', '射程', 'お金'
]

DISPLAY_COLS_CATS: List[str] = [
    'own', 'No.', 'ランク', 'キャラクター名', 'コスト', '再生産F', '速度',
    '範囲', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB', '特性'
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

st.set_page_config(layout='wide')


@st.cache_data
def load_and_process_cats_data() -> pd.DataFrame:
    """ExcelファイルからCatsデータを読み込み、特性を解析・付加して返す。

    Returns:
        pd.DataFrame: 特性解析後かつ数値変換済みのCatsデータフレーム
    """
    df = pd.read_csv(CATS_FILE, index_col=0)
    df.dropna(axis=1,how='all', inplace=True)   #空の列を削除
    df.dropna(axis=0,how='all', inplace=True)   #空の行を削除
    
    # 数値列を明示的に数値変換
    for col in NUMERIC_COLS_CATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if '特性' not in df.columns or df['特性'].isnull().all():
        return df

    exploded_df = df.assign(line=df['特性'].str.split('\n')).explode('line')
    traits_lines = exploded_df['line'].astype(str).str.strip()
    traits_df = pd.DataFrame(index=traits_lines.index)

    # 色特性判定
    for color in COLOR_TRAITS:
        pattern = rf'対(?!.*全敵.*{color}.*除く).*{color}.*'
        traits_df[color] = traits_lines.str.contains(pattern, na=False)

    # Boolean特性判定
    for trait_name, regex_pattern in BOOLEAN_TRAITS.items():
        traits_df[trait_name] = traits_lines.str.contains(regex_pattern, na=False, regex=True)

    # フラグ特性判定
    for flag_trait in FLAG_TRAITS:
        traits_df[flag_trait] = traits_lines.str.contains(flag_trait, na=False)

    agg_funcs = {col: 'any' for col in traits_df.columns}
    traits_aggregated = traits_df.groupby(traits_df.index).agg(agg_funcs)

    df = df.join(traits_aggregated)

    # 未生成の列にFalse埋め
    all_trait_cols = list(BOOLEAN_TRAITS.keys()) + FLAG_TRAITS + COLOR_TRAITS
    for col in all_trait_cols:
        if col not in df.columns:
            df[col] = False

    return df


@st.cache_data
def load_and_process_enemy_data() -> pd.DataFrame:
    """ExcelファイルからEnemyデータを読み込み、数値変換して返す。

    Returns:
        pd.DataFrame: 数値変換済みのEnemyデータフレーム
    """
    df = pd.read_csv(ENEMY_FILE, index_col=0)
    df.dropna(axis=1,how='all', inplace=True)   #空の列を削除
    df.dropna(axis=0,how='all', inplace=True)   #空の行を削除
    for col in NUMERIC_COLS_ENEMY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def filter_by_range_slider(
    dataframe: pd.DataFrame, column: str
) -> pd.DataFrame:
    """指定した列の値の範囲をスライダーで選択し、フィルタリングする。

    Args:
        dataframe (pd.DataFrame): 対象データフレーム
        column (str): フィルタリング対象の列名

    Returns:
        pd.DataFrame: フィルタリング後のデータフレーム
    """
    if column not in dataframe.columns:
        return dataframe

    col_series = dataframe[column].dropna()
    if col_series.empty:
        return dataframe

    min_value = int(col_series.min())
    max_value = int(col_series.max())
    if min_value == max_value:
        return dataframe

    step_size = max((max_value - min_value) // 100, 1)
    selected_range = st.sidebar.slider(
        label=column, min_value=min_value, max_value=max_value,
        value=(min_value, max_value), step=step_size
    )
    return dataframe[dataframe[column].between(*selected_range)]


def draw_comparison_bar_chart(
    selected_row: pd.Series,
    max_values: pd.Series,
    min_values: pd.Series,
    items: List[str],
) -> None:
    """選択された行の各数値項目の値を、現在の最大値と比較して棒グラフを表示する。

    Args:
        selected_row (pd.Series): 選択された1行のデータ
        max_values (pd.Series): 表示中の最大値集合
        min_values (pd.Series): 表示中の最小値集合
        items (List[str]): グラフに表示する項目リスト
    """
    bar_chart_data = []
    numeric_items = [
        item for item in items
        if item in NUMERIC_COLS_CATS or item in NUMERIC_COLS_ENEMY
    ]

    for item in numeric_items:
        value = selected_row.get(item)
        if pd.notna(value):
            max_val = max_values.get(item, 0)
            normalized_value = (value / max_val * 100) if max_val > 0 else 0
            bar_chart_data.append({
                '項目': item,
                '値': value,
                '正規化値': normalized_value,
                '最大値': max_val,
                '最小値': min_values.get(item),
            })

    if not bar_chart_data:
        st.write("表示できるデータがありません。")
        return

    chart_df = pd.DataFrame(bar_chart_data)
    sort_order = chart_df['項目'].tolist()

    background_bars = alt.Chart(chart_df).mark_bar(
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

    foreground_bars = alt.Chart(chart_df).mark_bar(cornerRadius=3).encode(
        x='正規化値:Q',
        y=alt.Y('項目:N', sort=sort_order, title=None),
        color=alt.condition(
            (alt.datum.項目 == '攻撃力') | (alt.datum.項目 == 'DPS'),
            alt.value('#d62728'),
            alt.value('#1f77b4'),
        ),
        tooltip=[
            alt.Tooltip('項目:N'),
            alt.Tooltip('値:Q', format=','),
            alt.Tooltip('最大値:Q', format=','),
            alt.Tooltip('最小値:Q', format=','),
        ],
    )

    chart = (
        background_bars + foreground_bars
    ).properties(height=alt.Step(30)).configure_axis(grid=False).configure_view(
        strokeWidth=0
    ).configure_legend(disable=True)

    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    """Streamlitアプリのメイン処理。CatsとEnemyタブを切り替えて表示する。"""
    df_cats = load_and_process_cats_data()
    df_enemy = load_and_process_enemy_data()

    selected_page = st.radio(
        label="tab",
        options=["Cats", "Enemy"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_page == "Cats":
        df_filtered = df_cats.copy()
        st.sidebar.title("Cats フィルター")

        if st.sidebar.checkbox('own'):
            df_filtered = df_filtered[df_filtered['own'] > 0]

        search_text = st.sidebar.text_input("キャラクター名")
        if search_text:
            df_filtered = df_filtered[df_filtered['キャラクター名'].str.contains(search_text, na=False)]

        selected_colors = st.segmented_control(
            '対象属性', COLOR_TRAITS, selection_mode='multi'
        )
        if selected_colors:
            mask = pd.Series(True, index=df_filtered.index)
            for color in selected_colors:
                if color in df_filtered.columns:
                    mask &= df_filtered[color]
            df_filtered = df_filtered[mask]

        selected_ranks = st.sidebar.multiselect(
            'ランク', ['基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア']
        )
        if selected_ranks:
            df_filtered = df_filtered[df_filtered['ランク'].isin(selected_ranks)]

        selected_ranges = st.sidebar.multiselect('単体or範囲', ['単体', '範囲'], default=['単体', '範囲'])
        if selected_ranges:
            df_filtered = df_filtered[df_filtered['範囲'].isin(selected_ranges)]

        selected_effects = st.sidebar.multiselect(
            '特殊効果',
            ['めっぽう強い', '打たれ強い', '超ダメージ', '攻撃力低下', '動きを止める',
             '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効']
        )
        if selected_effects:
            mask = pd.Series(True, index=df_filtered.index)
            for effect in selected_effects:
                if effect in df_filtered.columns:
                    mask &= df_filtered[effect]
            df_filtered = df_filtered[mask]

        selected_abilities = st.sidebar.multiselect(
            '特殊能力',
            ['波動', '小波動', '烈波', '小烈波', '爆波', 'クリティカル',
             '渾身の一撃', 'ゾンビキラー', '悪魔シールド貫通', 'バリアブレイク',
             '生き残る', '波動ストッパー']
        )
        if selected_abilities:
            mask = pd.Series(True, index=df_filtered.index)
            for ability in selected_abilities:
                if ability in df_filtered.columns:
                    mask &= df_filtered[ability]
            df_filtered = df_filtered[mask]

        slider_columns = ['コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB']
        for col in slider_columns:
            df_filtered = filter_by_range_slider(df_filtered, col)

        st.header("Cats DB")

        if not df_filtered.empty:
            max_vals = df_filtered[NUMERIC_COLS_CATS].max()
            min_vals = df_filtered[NUMERIC_COLS_CATS].min()

            columns_to_display = [col for col in DISPLAY_COLS_CATS if col in df_filtered.columns]
            display_df = df_filtered[columns_to_display]

            grid_builder = GridOptionsBuilder.from_dataframe(display_df)
            grid_builder.configure_default_column(suppressMenu=True)
            grid_builder.configure_selection(selection_mode="single")
            grid_builder.configure_column('キャラクター名', minWidth=150)
            grid_builder.configure_column('特性', minWidth=300, wrapText=True, autoHeight=True)

            short_width_cols = ['ランク', '範囲', 'KB', 'No.', 'own', '速度']
            for col_name in short_width_cols:
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
            if selected_rows:
                selected_series = pd.DataFrame(selected_rows).iloc[0]
                st.subheader(f"📊 {selected_series['キャラクター名']} のステータス")
                draw_comparison_bar_chart(selected_series, max_vals, min_vals, DISPLAY_COLS_CATS)
            else:
                st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")
        else:
            st.warning("この条件に一致するキャラクターはいません。")

    elif selected_page == "Enemy":
        df_filtered = df_enemy.copy()
        st.sidebar.title("Enemy フィルター")

        search_text = st.sidebar.text_input("敵キャラクター名")
        if search_text:
            df_filtered = df_filtered[df_filtered['キャラクター名'].str.contains(search_text, na=False)]

        st.header("Enemy DB")

        if not df_filtered.empty:
            max_vals = df_filtered[NUMERIC_COLS_ENEMY].max()
            min_vals = df_filtered[NUMERIC_COLS_ENEMY].min()

            grid_builder = GridOptionsBuilder.from_dataframe(df_filtered)
            grid_builder.configure_default_column(suppressMenu=True, filter=False)
            grid_builder.configure_selection(selection_mode="single")
            grid_options = grid_builder.build()

            grid_response = AgGrid(
                df_filtered,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
            )

            selected_rows = grid_response.get('selected_rows', [])
            if selected_rows:
                selected_series = pd.DataFrame(selected_rows).iloc[0]
                st.subheader(f"📊 {selected_series['キャラクター名']} のステータス")
                draw_comparison_bar_chart(selected_series, max_vals, min_vals, NUMERIC_COLS_ENEMY)
        else:
            st.warning("この条件に一致する敵キャラクターはいません。")


if __name__ == "__main__":
    main()
