# utils/filters.py
import pandas as pd
import streamlit as st
from typing import Optional, List, Tuple

class DataFilter:
    @staticmethod
    def filter_by_range_slider(
        df: pd.DataFrame,
        column: str,
        label: Optional[str] = None
    ) -> pd.DataFrame:
        """数値範囲によるフィルタリング"""
        if column not in df.columns:
            return df
            
        col_series = df[column].dropna()
        if col_series.empty:
            return df
            
        min_value = int(col_series.min())
        max_value = int(col_series.max())
        
        if min_value == max_value:
            return df
            
        step_size = max((max_value - min_value) // 100, 1)
        display_label = label if label else column
        
        selected_range = st.sidebar.slider(
            label=display_label,
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
            step=step_size
        )
        
        return df[df[column].between(*selected_range)]

    @staticmethod
    def filter_by_text_search(
        df: pd.DataFrame,
        column: str,
        search_text: str
    ) -> pd.DataFrame:
        """テキスト検索によるフィルタリング"""
        if not search_text or column not in df.columns:
            return df
        return df[df[column].str.contains(search_text, na=False)]

    @staticmethod
    def filter_by_multiselect(
        df: pd.DataFrame,
        column: str,
        options: List[str],
        default: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """複数選択によるフィルタリング"""
        if column not in df.columns:
            return df
            
        selected_values = st.sidebar.multiselect(
            column,
            options=options,
            default=default
        )
        
        if not selected_values:
            return df
            
        return df[df[column].isin(selected_values)]

    @staticmethod
    def filter_by_traits(
        df: pd.DataFrame,
        traits: List[str]
    ) -> pd.DataFrame:
        """特性によるフィルタリング"""
        if not traits:
            return df
            
        mask = pd.Series(True, index=df.index)
        for trait in traits:
            if trait in df.columns:
                mask &= df[trait]
        return df[mask]

    @staticmethod
    def safe_get_max_min(
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> Tuple[pd.Series, pd.Series]:
        """数値カラムの最大値と最小値を安全に取得"""
        cols = [col for col in numeric_cols if col in df.columns]
        if not cols:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        return df[cols].max(), df[cols].min()

    @staticmethod
    def apply_Own_filter(
        df: pd.DataFrame,
        Own_column: str = 'Own'
    ) -> pd.DataFrame:
        """所有フィルターの適用"""
        if st.sidebar.checkbox('Own') and Own_column in df.columns:
            return df[df[Own_column] > 0]
        return df