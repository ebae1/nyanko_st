# main.py
import streamlit as st
from services.data_loader import DataLoader
from services.data_processor import DataProcessor
from services.chart_renderer import ChartRenderer
from services.grid_handler import GridHandler
from utils.filters import DataFilter
from models.data_types import GridOptions
from config.settings import (
    ColumnSettings,
    UISettings,
    TraitSettings
)

class NyankoApp:
    def __init__(self):
        """アプリケーションの初期化"""
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.chart_renderer = ChartRenderer()
        self.grid_handler = GridHandler()
        self.data_filter = DataFilter()
        
        self.column_settings = ColumnSettings()
        self.ui_settings = UISettings()
        self.trait_settings = TraitSettings()
        
        st.set_page_config(layout='wide')

    def run(self):
        """アプリケーションのメイン実行部分"""
        # データの読み込み
        df_cats = self.data_loader.load_cats_data()
        df_cats = self.data_processor.process_traits(df_cats)
        df_enemy = self.data_loader.load_enemy_data()

        # タブ選択
        selected_page = st.radio(
            label="tab",
            options=["Cats", "Enemy"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if selected_page == "Cats":
            self._show_cats_page(df_cats)
        else:
            self._show_enemy_page(df_enemy)

    def _show_cats_page(self, df: pd.DataFrame):
        """Cats ページの表示"""
        df_filtered = df.copy()
        st.sidebar.title("Cats フィルター")

        # フィルター適用
        df_filtered = self._apply_cats_filters(df_filtered)

        st.header("Cats DB")
        self._show_cats_grid_and_chart(df_filtered)

    def _apply_cats_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cats のフィルター適用"""
        # own フィルター
        df = self.data_filter.apply_own_filter(df)

        # キャラクター名検索
        search_text = st.sidebar.text_input("キャラクター名")
        df = self.data_filter.filter_by_text_search(
            df, 'キャラクター名', search_text
        )

        # 属性フィルター
        selected_colors = st.segmented_control(
            '対象属性',
            self.trait_settings.COLOR_TRAITS,
            selection_mode='multi'
        )
        df = self.data_filter.filter_by_traits(df, selected_colors)

        # ランクフィルター
        df = self.data_filter.filter_by_multiselect(
            df, 'ランク', self.ui_settings.RANK_OPTIONS
        )

        # 範囲フィルター
        df = self.data_filter.filter_by_multiselect(
            df, '範囲', self.ui_settings.RANGE_OPTIONS,
            default=self.ui_settings.RANGE_OPTIONS
        )

        # 特殊効果フィルター
        selected_effects = st.sidebar.multiselect(
            '特殊効果', self.ui_settings.SPECIAL_EFFECTS
        )
        df = self.data_filter.filter_by_traits(df, selected_effects)

        # 特殊能力フィルター
        selected_abilities = st.sidebar.multiselect(
            '特殊能力', self.ui_settings.SPECIAL_ABILITIES
        )
        df = self.data_filter.filter_by_traits(df, selected_abilities)

        # 数値フィルター
        slider_columns = [
            'コスト', '再生産F', '速度', '射程', '発生F',
            '攻撃力', '頻度F', 'DPS', '体力', 'KB'
        ]
        for col in slider_columns:
            df = self.data_filter.filter_by_range_slider(df, col)

        return df

    def _show_cats_grid_and_chart(self, df: pd.DataFrame):
        """Cats のグリッドとチャート表示"""
        if df.empty:
            st.warning("この条件に一致するキャラクターはいません。")
            return

        max_vals, min_vals = self.data_filter.safe_get_max_min(
            df, self.column_settings.NUMERIC_COLS_CATS
        )
        
        columns_to_display = [
            col for col in self.column_settings.DISPLAY_COLS_CATS
            if col in df.columns
        ]
        
        grid_options = GridOptions(
            columns=columns_to_display,
            min_widths={'キャラクター名': 150},
            special_columns=['ランク', '範囲', 'KB', 'No.', 'Own', '速度']
        )

        selected_row = self.grid_handler.configure_and_show_grid(
            df[columns_to_display], grid_options
        )

        if selected_row:
            name = selected_row.get('キャラクター名', '')
            st.subheader(f"📊 {name} のステータス")
            self.chart_renderer.draw_comparison_chart(
                pd.Series(selected_row),
                max_vals,
                min_vals,
                columns_to_display,
                self.column_settings.NUMERIC_COLS_CATS
            )
        else:
            st.info("上の表からキャラクターをクリックして選択すると、ステータスグラフが表示されます。")

    def _show_enemy_page(self, df: pd.DataFrame):
        """Enemy ページの表示"""
        df_filtered = df.copy()
        st.sidebar.title("Enemy フィルター")

        # キャラクター名検索
        search_text = st.sidebar.text_input("敵キャラクター名")
        df_filtered = self.data_filter.filter_by_text_search(
            df_filtered, 'キャラクター名', search_text
        )

        st.header("Enemy DB")
        self._show_enemy_grid_and_chart(df_filtered)

    def _show_enemy_grid_and_chart(self, df: pd.DataFrame):
        """Enemy のグリッドとチャート表示"""
        if df.empty:
            st.warning("この条件に一致する敵キャラクターはいません。")
            return

        max_vals, min_vals = self.data_filter.safe_get_max_min(
            df, self.column_settings.NUMERIC_COLS_ENEMY
        )

        grid_options = GridOptions(
            columns=df.columns.tolist(),
            min_widths={},
            special_columns=[]
        )

        selected_row = self.grid_handler.configure_and_show_grid(
            df, grid_options
        )

        if selected_row:
            name = selected_row.get('キャラクター名', '')
            st.subheader(f"📊 {name} のステータス")
            self.chart_renderer.draw_comparison_chart(
                pd.Series(selected_row),
                max_vals,
                min_vals,
                self.column_settings.NUMERIC_COLS_ENEMY,
                self.column_settings.NUMERIC_COLS_ENEMY
            )

if __name__ == "__main__":
    app = NyankoApp()
    app.run()