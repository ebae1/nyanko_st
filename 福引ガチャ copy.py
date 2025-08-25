import time
from typing import List, Optional
import numpy as np
from numba import njit, prange

# --- 定数定義 ---
U32_MASK = 0xffffffff

TABLES = [
    {
        'count': 20,
        'rarity_max': 7399,
        0: "ちび巨神ネコ",
        1: "ちびネコトカゲ",
        2: "ちびネコフィッシュ",
        3: "ちびネコノトリ",
        4: "ちびウシネコ",
        5: "ちびキモネコ",
        6: "ちびバトルネコNP",
        7: "ちびタンクネコ",
        8: "ちびネコNP",
        9: "スピードアップ",
        10: "スピードアップ",
        11: "スピードアップ",
        12: "ニャンピューター",
        13: "ニャンピューター",
        14: "XP10000",
        15: "XP10000",
        16: "XP10000",
        17: "XP30000",
        18: "XP30000",
        19: "XP30000",
    },
    {
        'count': 3,
        'rarity_max': 9499,
        0: "ネコボン",
        1: "おかめはちもく",
        2: "スニャイパー",
    },
    {
        'count': 1,
        'rarity_max': 9999,
        0: 'トレジャーレーダー',
    },
]

DUP_TABLE_INDEX = 0
DUP_COUNT = TABLES[DUP_TABLE_INDEX]['count']

TARGET_ITEMS = [
    'XP30000', 'トレジャーレーダー', 'XP10000', 'ちびタンクネコ',
    'おかめはちもく', 'XP30000', 'ちびネコノトリ', 'ちびウシネコ',
    'ネコボン', 'ちびウシネコ', 'ちびキモネコ', 'ちびネコノトリ',
    'スピードアップ','XP30000','スピードアップ','XP30000',
    'XP10000','XP30000','ネコボン','ちび巨神ネコ',
    'XP30000','ちびタンクネコ','XP30000','ネコボン',
    'スピードアップ','XP10000'
]

# --- 乱数 ---
@njit
def xorshift32(seed: int) -> int:
    seed ^= (seed << 13) & U32_MASK
    seed ^= (seed >> 17)
    seed ^= (seed << 15) & U32_MASK
    return seed & U32_MASK

@njit
def select_table_idx(rarity: int) -> (int, int):
    for idx, tbl in enumerate(TABLES):
        if rarity <= tbl['rarity_max']:
            return idx, tbl['count']
    return -1, -1

@njit
def calc_alt_slot(seed2: int, count: int, original_slot: int) -> int:
    alt_slot = xorshift32(seed2) % (count - 1)
    if original_slot <= alt_slot:
        alt_slot += 1
    return alt_slot

# --- ガチャ模擬 ---
@njit
def simulate(seed: int, length: int, target: List[str]) -> bool:
    """
    与えられた seed からガチャを length 回シミュレーションし item列が target と一致するか
    """
    results = []
    prev_items = {"A": None, "B": None}
    current_track = "A"
    seed = xorshift32(seed)   # 初回スキップ挙動

    for i in range(length*2):
        rarity = seed % 10000
        seed2 = xorshift32(seed)
        table_idx, count = select_table_idx(rarity)
        slot = seed2 % count
        item = TABLES[table_idx][slot]

        # track選択
        track = "A" if (i % 2 == 0) else "B"

        # 重複判定
        if prev_items[track] == item and table_idx == DUP_TABLE_INDEX:
            alt_slot = calc_alt_slot(seed2, count, slot)
            item = TABLES[table_idx][alt_slot]
            current_track = "B" if track == "A" else "A"
            prev_items[track] = item
        else:
            prev_items[track] = item

        # 実際のストリーム選択
        if track == current_track:
            results.append(item)
            if len(results) == length:
                break

        seed = seed2

    # 判定
    if len(results) < length:
        return False
    for i in range(length):
        if results[i] != target[i]:
            return False
    return True

# --- 並列探索 ---
@njit(parallel=True)
def search_seed(start: int, end: int, target: List[str]) -> int:
    length = len(target)
    for s in prange(start, end):
        if simulate(s, length, target):
            return s
    return -1

if __name__ == "__main__":
    start_t = time.time()
    # 範囲を制御（例: 0～1000万まで探索）
    result = search_seed(0, 10_000_000, TARGET_ITEMS)
    end_t = time.time()

    if result != -1:
        print(f"一致するseedを発見: {result}")
    else:
        print("一致するseedなし (探索範囲内)")
    print(f"探索時間: {end_t - start_t:.2f} 秒")
