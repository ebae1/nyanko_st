# services/data_loader.py
import pandas as pd
import streamlit as st
from typing import Optional

from ..config.settings import FileSettings, ColumnSettings

class DataLoader:
    def __init__(self):
        self.file_settings = FileSettings()
        self.column_settings = ColumnSettings()

    @st.cache_data
    def load_cats_data(self) -> pd.DataFrame:
        """Catsデータの読み込み"""
        try:
            return self._load_csv_file(
                self.file_settings.CATS_FILE,
                self.column_settings.NUMERIC_COLS_CATS
            )
        except Exception as e:
            st.error(f"Catsデータの読み込みに失敗しました: {str(e)}")
            return pd.DataFrame()

    @st.cache_data
    def load_enemy_data(self) -> pd.DataFrame:
        """敵データの読み込み"""
        try:
            return self._load_csv_file(
                self.file_settings.ENEMY_FILE,
                self.column_settings.NUMERIC_COLS_ENEMY
            )
        except Exception as e:
            st.error(f"敵データの読み込みに失敗しました: {str(e)}")
            return pd.DataFrame()

    def _load_csv_file(
        self, file_path: str, numeric_cols: List[str]
    ) -> pd.DataFrame:
        """csvファイルの読み込みと基本的な前処理"""
        df = pd.read_csv(file_path, encoding='shift_jis',index_col=0)
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df