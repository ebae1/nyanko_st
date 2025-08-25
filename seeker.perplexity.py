import time
from typing import List, Optional, Tuple

import numba as nb
from numba import njit, prange

# --- 定数定義 ---
U32_MASK = 0xffffffff

# ガチャテーブル定義
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


# --- 疑似乱数 (Xorshift32) ---
@njit
def xorshift32(seed: int) -> int:
    seed ^= (seed << 13) & U32_MASK
    seed ^= (seed >> 17)
    seed ^= (seed << 15) & U32_MASK
    return seed & U32_MASK


# --- slot選択 ---
@njit
def select_table_idx(rarity: int):
    for idx in range(len(TABLES)):
        if rarity <= TABLES[idx]['rarity_max']:
            return idx, TABLES[idx]['count']
    return -1, -1


@njit
def calc_alt_slot(seed: int, count: int, original_slot: int) -> int:
    alt_slot = xorshift32(seed) % (count - 1)
    if original_slot <= alt_slot:
        alt_slot += 1
    return alt_slot


# --- 1シードからガチャ結果を生成 ---
@njit
def simulate(seed: int, rolls: int) -> List[str]:
    results = []
    next_seed = xorshift32(seed)
    track_a, track_b = [], []
    current = next_seed
    for i in range(rolls * 2):
        rarity = current % 10000
        next_seed2 = xorshift32(current)
        table_idx, count = select_table_idx(rarity)
        slot = next_seed2 % count
        item = TABLES[table_idx][slot]

        track = track_a if i % 2 == 0 else track_b
        prev_item = track[-1] if len(track) > 0 else None

        # 被り処理
        if prev_item is not None and table_idx == DUP_TABLE_INDEX and item == prev_item:
            alt_slot = calc_alt_slot(next_seed2, count, slot)
            item = TABLES[table_idx][alt_slot]

        track.append(item)
        current = next_seed2

    # トラック統合（A→B切替）
    result = []
    ca, cb = 0, 0
    flag = 'A'
    for _ in range(rolls):
        if flag == 'A':
            result.append(track_a[ca])
            ca += 1
            flag = 'B'
        else:
            result.append(track_b[cb])
            cb += 1
            flag = 'A'
    return result


# --- シード全探索（並列化で高速化可能） ---
@njit(parallel=True)
def find_seed(target: List[str], max_seed: int = 2**32 - 1) -> int:
    t_len = len(target)
    for s in prange(1, max_seed, 1):  # seed=0はxorshift32で無限ループするため除外
        result = simulate(s, t_len)
        match = True
        for i in range(t_len):
            if result[i] != target[i]:
                match = False
                break
        if match:
            return s
    return -1


# --- 実行部 ---
if __name__ == '__main__':
    start = time.perf_counter()
    # 入力するターゲット列
    target_items = [
        'XP30000',
        'トレジャーレーダー',
        'XP10000',
        'ちびタンクネコ',
        'おかめはちもく',
        'XP30000',
        'ちびネコノトリ',
        'ちびウシネコ',
        'ネコボン',
        'ちびウシネコ',
        'ちびキモネコ',
        'ちびネコノトリ',
        'スピードアップ',
        'XP30000',
        'スピードアップ',
        'XP30000',
        'XP10000',
        'XP30000',
        'ネコボン',
        'ちび巨神ネコ',
        'XP30000',
        'ちびタンクネコ',
        'XP30000',
        'ネコボン',
        'スピードアップ',
        'XP10000',
    ]
    seed = find_seed(target_items)
    end = time.perf_counter()
    if seed != -1:
        print(f"Found seed: {seed}")
    else:
        print("No matching seed found.")
    print(f"実行時間: {end - start:.4f} 秒")
