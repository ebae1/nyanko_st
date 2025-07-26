# services/data_processor.py
import pandas as pd
from typing import List
from config.settings import TraitSettings

class DataProcessor:
    def __init__(self):
        self.trait_settings = TraitSettings()

    def process_traits(self, df: pd.DataFrame) -> pd.DataFrame:
        """特性の処理"""
        if '特性' not in df.columns or df['特性'].isnull().all():
            return df

        exploded_df = df.assign(line=df['特性'].str.split('\n')).explode('line')
        traits_lines = exploded_df['line'].astype(str).str.strip()
        traits_df = self._create_traits_df(traits_lines)
        
        return self._aggregate_and_fill_traits(df, traits_df)

    def _create_traits_df(self, traits_lines: pd.Series) -> pd.DataFrame:
        """特性データフレームの作成"""
        traits_df = pd.DataFrame(index=traits_lines.index)
        
        # 色特性の処理
        for color in self.trait_settings.COLOR_TRAITS:
            pattern = rf'対(?!.*全敵.*{color}.*除く).*{color}.*'
            traits_df[color] = traits_lines.str.contains(pattern, na=False)
        
        # Boolean特性の処理
        for trait_name, regex_pattern in self.trait_settings.BOOLEAN_TRAITS.items():
            traits_df[trait_name] = traits_lines.str.contains(
                regex_pattern, na=False, regex=True
            )
        
        # フラグ特性の処理
        for flag_trait in self.trait_settings.FLAG_TRAITS:
            traits_df[flag_trait] = traits_lines.str.contains(flag_trait, na=False)
        
        return traits_df

    def _aggregate_and_fill_traits(
        self, original_df: pd.DataFrame, traits_df: pd.DataFrame
    ) -> pd.DataFrame:
        """特性の集約と欠損値の補完"""
        agg_funcs = {col: 'any' for col in traits_df.columns}
        traits_aggregated = traits_df.groupby(traits_df.index).agg(agg_funcs)
        df = original_df.join(traits_aggregated)
        
        all_traits = (
            list(self.trait_settings.BOOLEAN_TRAITS.keys()) +
            self.trait_settings.FLAG_TRAITS +
            self.trait_settings.COLOR_TRAITS
        )
        
        for col in all_traits:
            if col not in df.columns:
                df[col] = False
        
        return df