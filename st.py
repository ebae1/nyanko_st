import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from typing import List, Dict, Tuple

# === 定数定義 ===
CATS_DATA_FILE = './0.datafiles/org_catsdb.xlsx'
ENEMY_DATA_FILE = './0.datafiles/nyanko_enemy_db.xlsx'

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

# === データロード・処理関数 ===

@st.cache_data
def load_cats_data() -> pd.DataFrame:
    """
    Catsデータを読み込み、数値カラム変換および特性解析で各特性のフラグを追加する。
    """
    df = pd.read_excel(CATS_DATA_FILE, index_col=0)
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # 数値列を強制的に数値化（解析・比較のため）
    for col in NUMERIC_COLUMNS_CATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 特性列が無ければ以降の特性解析はスキップ
    if '特性' not in df.columns or df['特性'].isnull().all():
        return df

    # 特性列は改行で区切られ複数存在する可能性があるため、行として展開
    traits_expanded = df.assign(trait_lines=df['特性'].str.split('\n')).explode('trait_lines')
    trait_lines = traits_expanded['trait_lines'].astype(str).str.strip()

    traits_flags = pd.DataFrame(index=trait_lines.index)

    # 色特性フラグ。特定の条件付きregexを用いて含む行をTrueに
    for color in COLOR_TRAITS:
        pattern = rf'対(?!.*全敵.*{color}.*除く).*{color}.*'
        traits_flags[color] = trait_lines.str.contains(pattern, na=False)

    # ブール特性フラグ（正規表現含む）
    for trait_name, regex_pattern in BOOLEAN_TRAITS.items():
        traits_flags[trait_name] = trait_lines.str.contains(regex_pattern, na=False, regex=True)

    # フラグ特性（単純含有チェック）
    for flag_trait in FLAG_TRAITS:
        traits_flags[flag_trait] = trait_lines.str.contains(flag_trait, na=False)

    # 行単位で論理和(any)集約、元データに結合
    aggregated_traits = traits_flags.groupby(traits_flags.index).agg('any')
    df = df.join(aggregated_traits)

    # 欠損している可能性のある特性列はFalse補完しておく
    all_traits = list(BOOLEAN_TRAITS.keys()) + FLAG_TRAITS + COLOR_TRAITS
    for trait in all_traits:
        if trait not in df.columns:
            df[trait] = False

    return df


@st.cache_data
def load_enemy_data() -> pd.DataFrame:
    """
    Enemyデータを読み込み、数値列に変換する。
    """
    df = pd.read_csv(ENEMY_DATA_FILE, index_col=0)
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    for col in NUMERIC_COLUMNS_ENEMY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# === UI・フィルタ共通関数 ===

def filter_dataframe_by_numeric_range(
    dataframe: pd.DataFrame,
    column: str,
    sidebar_label_prefix: str = ""
) -> pd.DataFrame:
    """
    指定カラムの値をスライダーで範囲選択してフィルターするUIを生成し、
    選択範囲に該当する行のみを返す。
    """
    if column not in dataframe.columns:
        return dataframe

    series = dataframe[column].dropna()
    if series.empty:
        return dataframe

    min_val = int(series.min())
    max_val = int(series.max())
    if min_val == max_val:
        return dataframe

    step = max((max_val - min_val) // 100, 1)
    label = f"{sidebar_label_prefix}{column}" if sidebar_label_prefix else column

    selected_range = st.sidebar.slider(
        label=label,
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=step
    )

    return dataframe[dataframe[column].between(*selected_range)]


def filter_by_checkbox_column(
    dataframe: pd.DataFrame, 
    column: str,
    checked_label: str,
) -> pd.DataFrame:
    """
    単純に特定カラムの値が0より大きい行のみ抽出(checkboxオン時)
    """
    if checked_label and column in dataframe.columns:
        return dataframe[dataframe[column] > 0]
    return dataframe


def filter_by_text_search(
    dataframe: pd.DataFrame,
    column: str,
    search_text: str,
) -> pd.DataFrame:
    """
    指定カラムに対し文字列含有検索し、合致する行のみ抽出
    """
    if search_text and column in dataframe.columns:
        return dataframe[dataframe[column].str.contains(search_text, na=False)]
    return dataframe


def filter_by_multi_select_flags(
    dataframe: pd.DataFrame,
    flags: List[str],
) -> pd.DataFrame:
    """
    True/Falseフラグの複数選択フィルタリング。全選択フラグに合致する行のみ抽出
    """
    if not flags:
        return dataframe

    mask = pd.Series(True, index=dataframe.index)
    for flag in flags:
        if flag in dataframe.columns:
            mask &= dataframe[flag]

    return dataframe[mask]


# === 可視化関数 ===

def draw_status_comparison_chart(
    selected_row: pd.Series,
    max_values: pd.Series,
    min_values: pd.Series,
    items: List[str],
) -> None:
    """
    選択行の数値項目を、その項目の全体最大値と比較した割合の棒グラフとして描画
    """
    chart_data = []
    numeric_items = [item for item in items if item in NUMERIC_COLUMNS_CATS or item in NUMERIC_COLUMNS_ENEMY]

    for item in numeric_items:
        value = selected_row.get(item)
        if pd.notna(value):
            max_val = max_values.get(item, 0)
            min_val = min_values.get(item, None)
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

    chart_df = pd.DataFrame(chart_data)
    sort_order = chart_df['項目'].tolist()

    background = alt.Chart(chart_df).mark_bar(
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

    foreground = alt.Chart(chart_df).mark_bar(cornerRadius=3).encode(
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

    combined = (background + foreground).properties(height=alt.Step(30)).configure_axis(grid=False).configure_view(
        strokeWidth=0
    ).configure_legend(disable=True)

    st.altair_chart(combined, use_container_width=True)


def get_numeric_columns_max_min(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """
    指定した数値カラムに存在するカラムの最大値・最小値を取得する。
    """
    valid_cols = [col for col in numeric_cols if col in df.columns]
    if not valid_cols:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return df[valid_cols].max(), df[valid_cols].min()


# === メイン処理 ===

def main() -> None:
    st.set_page_config(layout="wide")

    cats_df = load_cats_data()
    enemy_df = load_enemy_data()

    selected_tab = st.radio("tab", options=["Cats", "Enemy"], horizontal=True, label_visibility="collapsed")

    if selected_tab == "Cats":
        with st.sidebar:
            st.title("Cats フィルター")
            own_only = st.checkbox("Own")
            search_character_name = st.text_input("キャラクター名")
            selected_colors = st.multiselect('対象属性', COLOR_TRAITS)
            selected_ranks = st.multiselect('ランク', ['基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア'])
            selected_ranges = st.multiselect('単体or範囲', ['単体', '範囲'], default=['単体', '範囲'])
            selected_effects = st.multiselect('特殊効果',
                                             ['めっぽう強い', '打たれ強い', '超ダメージ', '攻撃力低下', '動きを止める',
                                              '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効'])
            selected_abilities = st.multiselect('特殊能力',
                                               ['波動', '小波動', '烈波', '小烈波', '爆波',
                                                'クリティカル', '渾身の一撃', 'ゾンビキラー', '悪魔シールド貫通', 'バリアブレイク',
                                                '生き残る', '波動ストッパー'])

        filtered_df = cats_df.copy()
        if own_only:
            filtered_df = filter_by_checkbox_column(filtered_df, 'Own', own_only)

        filtered_df = filter_by_text_search(filtered_df, 'キャラクター名', search_character_name)

        if selected_colors:
            filtered_df = filter_by_multi_select_flags(filtered_df, selected_colors)

        if selected_ranks:
            filtered_df = filtered_df[filtered_df['ランク'].isin(selected_ranks)]

        if selected_ranges:
            filtered_df = filtered_df[filtered_df['範囲'].isin(selected_ranges)]

        if selected_effects:
            filtered_df = filter_by_multi_select_flags(filtered_df, selected_effects)

        if selected_abilities:
            filtered_df = filter_by_multi_select_flags(filtered_df, selected_abilities)

        slider_cols = ['コスト', '再生産F', '速度', '射程', '発生F', '攻撃力', '頻度F', 'DPS', '体力', 'KB']
        for col in slider_cols:
            filtered_df = filter_dataframe_by_numeric_range(filtered_df, col, sidebar_label_prefix="")

        st.header("Cats DB")
        if filtered_df.empty:
            st.warning("この条件に一致するキャラクターはいません。")
            return

        max_vals, min_vals = get_numeric_columns_max_min(filtered_df, NUMERIC_COLUMNS_CATS)
        display_columns = [col for col in DISPLAY_COLUMNS_CATS if col in filtered_df.columns]
        display_df = filtered_df[display_columns]

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
        if selected_rows:
            selected_series = pd.DataFrame(selected_rows).iloc[0]
            character_name = selected_series.get('キャラクター名', '')
            st.subheader(f"📊 {character_name} のステータス")
            draw_status_comparison_chart(selected_series, max_vals, min_vals, display_columns)
        else:
            st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")


    elif selected_tab == "Enemy":
        with st.sidebar:
            st.title("Enemy フィルター")
            search_enemy_name = st.text_input("敵キャラクター名")

        filtered_enemy_df = enemy_df.copy()
        filtered_enemy_df = filter_by_text_search(filtered_enemy_df, 'キャラクター名', search_enemy_name)

        st.header("Enemy DB")
        if filtered_enemy_df.empty:
            st.warning("この条件に一致する敵キャラクターはいません。")
            return

        max_vals, min_vals = get_numeric_columns_max_min(filtered_enemy_df, NUMERIC_COLUMNS_ENEMY)
        columns_to_show = filtered_enemy_df.columns.tolist()

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
        if selected_rows:
            selected_series = pd.DataFrame(selected_rows).iloc[0]
            enemy_name = selected_series.get('キャラクター名', '')
            st.subheader(f"📊 {enemy_name} のステータス")
            draw_status_comparison_chart(selected_series, max_vals, min_vals, NUMERIC_COLUMNS_ENEMY)


if __name__ == "__main__":
    main()
