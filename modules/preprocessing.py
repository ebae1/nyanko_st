# modules/preprocessing.py

import pandas as pd
from typing import List, Tuple, Optional, Dict

# === 定数定義（必要に応じてimportしても良い） ===
COLOR_TRAITS = ['赤', '浮', '黒', 'メタル', '天使', 'エイリアン', 'ゾンビ', '古代種', '悪魔', '白']
BOOLEAN_TRAITS = {'めっぽう強い': 'めっぽう強い', '打たれ強い': '打たれ強い', '超打たれ強い': '超打たれ強い', '超ダメージ': '超ダメージ', '極ダメージ': '極ダメージ', 'ターゲット限定': 'のみに攻撃', '魂攻撃': '魂攻撃', 'メタルキラー': 'メタルキラー', '被ダメージ1': r'被ダメージ\s*1', '波動ストッパー': '波動ストッパー', '烈波カウンター': '烈波カウンター', '1回攻撃': '1回攻撃', 'ゾンビキラー': 'ゾンビキラー', 'バリアブレイク': 'バリアブレイク', '悪魔シールド貫通': '悪魔シールド貫通'}
FLAG_TRAITS = ['攻撃力低下', '動きを止める', '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効', '渾身の一撃', '攻撃力上昇', '生き残る', 'クリティカル', '波動', '小波動', '烈波', '小烈波', '爆波']

def add_ratio_column(df: pd.DataFrame, numerator_col: str, denominator_col: str, new_col_name: Optional[str] = None, fillna_value: Optional[float] = None) -> pd.DataFrame:
    """分子/分母の新しい比率列を追加"""
    if new_col_name is None:
        new_col_name = f"{numerator_col}/{denominator_col}"
    df[new_col_name] = df.apply(
        lambda row: (row[numerator_col] / row[denominator_col])
        if pd.notna(row[numerator_col]) and pd.notna(row[denominator_col]) and row[denominator_col] != 0
        else None, axis=1
    )
    if fillna_value is not None:
        df[new_col_name] = df[new_col_name].fillna(fillna_value)
    df[new_col_name] = df[new_col_name].round(2)
    return df

def add_multiple_ratio_columns(df: pd.DataFrame, ratio_pairs: List[Tuple[str, str, Optional[str]]]) -> pd.DataFrame:
    """複数の比率列を一括追加"""
    for numerator, denominator, new_col in ratio_pairs:
        df = add_ratio_column(df, numerator, denominator, new_col_name=new_col)
    return df

def preprocess_cats_df(
    df: pd.DataFrame,
    numeric_columns: List[str],
    ratio_pairs: List[Tuple[str, str, Optional[str]]]
) -> pd.DataFrame:
    """Cats用: データ型変換/特性フラグ/比率列追加"""
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if '特性' not in df.columns or df['特性'].isnull().all():
        df = add_multiple_ratio_columns(df, ratio_pairs)
        return df
    traits_lines = df['特性'].str.split('\n').explode().str.strip()
    traits_flags_df = pd.DataFrame(index=traits_lines.index)
    for color_trait in COLOR_TRAITS:
        pattern = rf'対(?!.*全敵.*{color_trait}.*除く).*{color_trait}.*'
        traits_flags_df[color_trait] = traits_lines.str.contains(pattern, na=False)
    for trait_name, regex_pattern in BOOLEAN_TRAITS.items():
        traits_flags_df[trait_name] = traits_lines.str.contains(regex_pattern, na=False, regex=True)
    for flag_trait in FLAG_TRAITS:
        traits_flags_df[flag_trait] = traits_lines.str.contains(flag_trait, na=False)
    aggregated_traits_flags = traits_flags_df.groupby(traits_flags_df.index).any()
    df = df.join(aggregated_traits_flags)
    all_traits = list(BOOLEAN_TRAITS.keys()) + FLAG_TRAITS + COLOR_TRAITS
    for trait in all_traits:
        if trait not in df.columns:
            df[trait] = False
    df = add_multiple_ratio_columns(df, ratio_pairs)
    return df

def preprocess_enemy_df(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Enemy用: データ型変換のみ"""
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
