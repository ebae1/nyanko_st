# services/chart_renderer.py
import streamlit as st
import altair as alt
import pandas as pd
from typing import List, Dict
from models.data_types import ChartData

class ChartRenderer:
    def draw_comparison_chart(
        self,
        selected_row: pd.Series,
        max_values: pd.Series,
        min_values: pd.Series,
        items: List[str],
        numeric_cols: List[str]
    ) -> None:
        """比較チャートの描画"""
        chart_data = self._prepare_chart_data(
            selected_row, max_values, min_values, items, numeric_cols
        )
        
        if not chart_data:
            st.write("表示できるデータがありません。")
            return
        
        chart_df = pd.DataFrame(chart_data)
        chart = self._create_chart(chart_df)
        st.altair_chart(chart, use_container_width=True)

    def _prepare_chart_data(
        self,
        selected_row: pd.Series,
        max_values: pd.Series,
        min_values: pd.Series,
        items: List[str],
        numeric_cols: List[str]
    ) -> List[Dict]:
        """チャートデータの準備"""
        chart_data = []
        numeric_items = [
            item for item in items if item in numeric_cols
        ]
        
        for item in numeric_items:
            value = selected_row.get(item)
            if pd.notna(value):
                max_val = max_values.get(item, 0)
                normalized_value = (value / max_val * 100) if max_val > 0 else 0
                chart_data.append({
                    '項目': item,
                    '値': value,
                    '正規化値': normalized_value,
                    '最大値': max_val,
                    '最小値': min_values.get(item),
                })
        
        return chart_data

    def _create_chart(self, chart_df: pd.DataFrame) -> alt.Chart:
        """チャートの作成"""
        sort_order = chart_df['項目'].tolist()
        
        background = self._create_background_bars(chart_df, sort_order)
        foreground = self._create_foreground_bars(chart_df, sort_order)
        
        return (
            background + foreground
        ).properties(
            height=alt.Step(30)
        ).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            disable=True
        )

    def _create_background_bars(
        self, chart_df: pd.DataFrame, sort_order: List[str]
    ) -> alt.Chart:
        """背景バーの作成"""
        return alt.Chart(chart_df).mark_bar(
            color='#e0e0e0',
            cornerRadius=3
        ).encode(
            x=alt.X(
                'max(正規化値):Q',
                scale=alt.Scale(domain=[0, 100]),
                title='最大値に対する割合(%)'
            ),
            y=alt.Y('項目:N', sort=sort_order, title=None),
            tooltip=[
                alt.Tooltip('項目:N'),
                alt.Tooltip('値:Q', format=','),
                alt.Tooltip('最大値:Q', format=','),
                alt.Tooltip('最小値:Q', format=','),
            ],
        ).transform_calculate(正規化値='100')

    def _create_foreground_bars(
        self, chart_df: pd.DataFrame, sort_order: List[str]
    ) -> alt.Chart:
        """前景バーの作成"""
        return alt.Chart(chart_df).mark_bar(
            cornerRadius=3
        ).encode(
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