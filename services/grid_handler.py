# services/grid_handler.py
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
from typing import Optional, Dict, Any
from models.data_types import GridOptions

class GridHandler:
    def configure_and_show_grid(
        self,
        df: pd.DataFrame,
        grid_options: GridOptions
    ) -> Optional[Dict[str, Any]]:
        """グリッドの設定と表示"""
        grid_builder = self._create_grid_builder(df, grid_options)
        grid_options = grid_builder.build()
        
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True
        )
        
        selected_rows = grid_response.get('selected_rows', [])
        return selected_rows[0] if selected_rows else None

    def _create_grid_builder(
        self,
        df: pd.DataFrame,
        grid_options: GridOptions
    ) -> GridOptionsBuilder:
        """グリッドビルダーの作成"""
        grid_builder = GridOptionsBuilder.from_dataframe(df)
        grid_builder.configure_default_column(suppressMenu=True)
        grid_builder.configure_selection(selection_mode="single")
        
        # カラム幅の設定
        for col, width in grid_options.min_widths.items():
            if col in df.columns:
                grid_builder.configure_column(col, minWidth=width)
        
        # 特殊カラムの設定
        if '特性' in df.columns:
            grid_builder.configure_column(
                '特性', minWidth=300, wrapText=True, autoHeight=True
            )
        
        return grid_builder