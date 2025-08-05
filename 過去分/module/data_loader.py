# data_loader.py
import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Tuple, Any
from settings import FileSettings, ColumnSettings

class DataLoader:
    file_settings = FileSettings()
    column_settings = ColumnSettings()

    @staticmethod
    @st.cache_data
    def load_cats_data() -> pd.DataFrame:  # selfパラメータを削除
        """Catsデータの読み込み"""
        try:
            return DataLoader._load_csv_file(  # selfをDataLoaderに変更
                DataLoader.file_settings.CATS_FILE,  # selfをDataLoaderに変更
                DataLoader.column_settings.NUMERIC_COLS_CATS
            )
        except Exception as e:
            st.error(f"Catsデータの読み込みに失敗しました: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data
    def load_enemy_data() -> pd.DataFrame:  # selfパラメータを削除
        """敵データの読み込み"""
        try:
            return DataLoader._load_csv_file(  # selfをDataLoaderに変更
                DataLoader.file_settings.ENEMY_FILE,  # selfをDataLoaderに変更
                DataLoader.column_settings.NUMERIC_COLS_ENEMY
            )
        except Exception as e:
            st.error(f"敵データの読み込みに失敗しました: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _load_csv_file(
        file_path: str,  # selfパラメータを削除
        numeric_cols: List[str]
    ) -> pd.DataFrame:
        """csvファイルの読み込みと基本的な前処理"""
        df = pd.read_csv(file_path, encoding='shift_jis', index_col=0)
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df