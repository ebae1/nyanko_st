# models/data_types.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd

@dataclass
class ChartData:
    item: str
    value: float
    normalized_value: float
    max_value: float
    min_value: float

@dataclass
class GridOptions:
    columns: List[str]
    min_widths: Dict[str, int]
    special_columns: List[str]

@dataclass
class FilterState:
    Own_filter: bool
    search_text: str
    selected_colors: List[str]
    selected_ranks: List[str]
    selected_ranges: List[str]
    selected_effects: List[str]
    selected_abilities: List[str]