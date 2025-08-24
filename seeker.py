import numpy as np
from numba import njit, prange
import multiprocessing as mp
from typing import List, Optional

# テーブル定義を高速アクセス用に変換
ITEM_NAMES = [
    # Table 0 (ID: 0-19)
    'ちび巨神ネコ', 'ちびネコトカゲ', 'ちびネコフィッシュ', 'ちびネコノトリ', 'ちびウシネコ',
    'ちびキモネコ', 'ちびバトルネコNP', 'ちびタンクネコ', 'ちびネコNP', 'スピードアップ',
    'スピードアップ', 'スピードアップ', 'ニャンピューター', 'ニャンピューター', 'XP10000',
    'XP10000', 'XP10000', 'XP30000', 'XP30000', 'XP30000',
    # Table 1 (ID: 20-22)
    'ネコボン', 'おかめはちもく', 'スニャイパー',
    # Table 2 (ID: 23)
    'トレジャーレーダー'
]

# アイテム名→ID変換
ITEM_TO_ID = {name: i for i, name in enumerate(ITEM_NAMES)}

# テーブル情報 [開始ID, 個数, レアリティ上限]
TABLE_INFO = np.array([
    [0, 20, 7399],   # Table 0
    [20, 3, 9499],   # Table 1  
    [23, 1, 9999]    # Table 2
], dtype=np.int32)

DUP_TABLE_INDEX = 0
DUP_COUNT = 20

@njit(inline='always')
def xorshift32(x: np.uint32) -> np.uint32:
    """高速化されたxorshift32"""
    x = x ^ (x << np.uint32(13))
    x = x ^ (x >> np.uint32(17))
    x = x ^ (x << np.uint32(15))
    return x

@njit(inline='always')
def select_table_n(rarity: np.int32) -> tuple:
    """レアリティ値に応じたテーブル番号とテーブル内の項目数を返す"""
    if rarity <= 7399:
        return 0, 20
    elif rarity <= 9499:
        return 1, 3
    elif rarity <= 9999:
        return 2, 1
    return -1, -1

@njit(inline='always')
def get_item_id(table_idx: np.int32, slot: np.int32) -> np.int32:
    """テーブル番号とスロットからアイテムIDを取得"""
    if table_idx == 0:
        return slot
    elif table_idx == 1:
        return 20 + slot
    elif table_idx == 2:
        return 23
    return -1

@njit(inline='always')
def calc_alt_slot(seed: np.uint32, original_slot: np.int32) -> np.int32:
    """被り時の代替スロット計算"""
    alt_slot = xorshift32(seed) % 19  # 20-1
    if original_slot <= alt_slot:
        alt_slot += 1
    return alt_slot

@njit
def simulate_gacha_fast(initial_seed: np.uint32, target_items: np.ndarray) -> bool:
    """高速ガチャシミュレーション"""
    if initial_seed == 0:
        return False
    
    # 必要なロール数を計算（最大200回）
    max_rolls = min(len(target_items) * 3, 200)
    
    # トラックA/Bの事前割り当て
    track_a_items = np.zeros(max_rolls, dtype=np.int32)
    track_b_items = np.zeros(max_rolls, dtype=np.int32)
    track_a_tables = np.zeros(max_rolls, dtype=np.int32)
    track_b_tables = np.zeros(max_rolls, dtype=np.int32)
    track_a_slots = np.zeros(max_rolls, dtype=np.int32)
    track_b_slots = np.zeros(max_rolls, dtype=np.int32)
    track_a_seeds = np.zeros(max_rolls, dtype=np.uint32)
    track_b_seeds = np.zeros(max_rolls, dtype=np.uint32)
    
    a_idx = 0
    b_idx = 0
    
    seed = initial_seed
    
    # 各トラックにアイテムを配置
    for i in range(max_rolls * 2):
        # レアリティ決定
        rarity = np.int32(seed % 10000)
        next_seed = xorshift32(seed)
        
        # テーブルとスロット決定
        table_idx, count = select_table_n(rarity)
        if table_idx == -1:
            return False
        
        slot = np.int32(next_seed % count)
        item_id = get_item_id(table_idx, slot)
        
        # トラックに配置
        if i % 2 == 0:  # トラックA
            if a_idx >= max_rolls:
                break
            
            # 同一トラック内の被り処理
            if a_idx > 0 and table_idx == DUP_TABLE_INDEX and item_id == track_a_items[a_idx-1]:
                alt_slot = calc_alt_slot(next_seed, slot)
                item_id = alt_slot  # Table 0なのでそのままID
            
            track_a_items[a_idx] = item_id
            track_a_tables[a_idx] = table_idx
            track_a_slots[a_idx] = slot
            track_a_seeds[a_idx] = next_seed
            a_idx += 1
        else:  # トラックB
            if b_idx >= max_rolls:
                break
                
            # 同一トラック内の被り処理
            if b_idx > 0 and table_idx == DUP_TABLE_INDEX and item_id == track_b_items[b_idx-1]:
                alt_slot = calc_alt_slot(next_seed, slot)
                item_id = alt_slot
            
            track_b_items[b_idx] = item_id
            track_b_tables[b_idx] = table_idx
            track_b_slots[b_idx] = slot
            track_b_seeds[b_idx] = next_seed
            b_idx += 1
        
        seed = next_seed
    
    # トラック結合とターゲット照合
    result_idx = 0
    current_track = 0  # 0=A, 1=B
    a_ptr = 0
    b_ptr = 0
    tr_prev_item = -1
    prev_track = 0
    
    while result_idx < len(target_items):
        if current_track == 0:  # トラックA
            if a_ptr >= a_idx:
                return False
            
            # Bから移動してきた場合、1回スキップ
            if prev_track == 1:
                prev_track = 0
                a_ptr += 1
                continue
            
            prev_track = 0
            item = track_a_items[a_ptr]
            
            # トラック移動後の被り処理
            if result_idx > 0 and tr_prev_item >= 0 and track_a_tables[a_ptr] == DUP_TABLE_INDEX and item == tr_prev_item:
                alt_slot = calc_alt_slot(track_a_seeds[a_ptr], track_a_slots[a_ptr])
                item = alt_slot
                current_track = 1  # トラック移動
                tr_prev_item = item
            # 通常の被り（alt_item発生）でトラック移動
            elif a_ptr > 0 and track_a_tables[a_ptr] == DUP_TABLE_INDEX and item == track_a_items[a_ptr-1]:
                # alt_itemは既に適用済み
                current_track = 1  # トラック移動
                tr_prev_item = item
            else:
                tr_prev_item = item
            
            a_ptr += 1
            
        else:  # トラックB
            if b_ptr >= b_idx:
                return False
                
            # Aから移動してきた場合、1回スキップ
            if prev_track == 0:
                prev_track = 1
                b_ptr += 1
                continue
            
            prev_track = 1
            item = track_b_items[b_ptr]
            
            # トラック移動後の被り処理
            if result_idx > 0 and tr_prev_item >= 0 and track_b_tables[b_ptr] == DUP_TABLE_INDEX and item == tr_prev_item:
                alt_slot = calc_alt_slot(track_b_seeds[b_ptr], track_b_slots[b_ptr])
                item = alt_slot
                current_track = 0  # トラック移動
                tr_prev_item = item
            # 通常の被り（alt_item発生）でトラック移動
            elif b_ptr > 0 and track_b_tables[b_ptr] == DUP_TABLE_INDEX and item == track_b_items[b_ptr-1]:
                current_track = 0  # ト