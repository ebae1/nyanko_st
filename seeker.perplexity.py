import time
import numpy as np
from numba import njit, prange
from multiprocessing import Pool, cpu_count
from typing import List

U32_MASK = 0xffffffff

# ガチャテーブル
TABLES = [
    {
        'count': 20,
        'rarity_max': 7399,
        '0': 'ちび巨神ネコ',
        '1': 'ちびネコトカゲ',
        '2': 'ちびネコフィッシュ',
        '3': 'ちびネコノトリ',
        '4': 'ちびウシネコ',
        '5': 'ちびキモネコ',
        '6': 'ちびバトルネコNP',
        '7': 'ちびタンクネコ',
        '8': 'ちびネコNP',
        '9': 'スピードアップ',
        '10': 'スピードアップ',
        '11': 'スピードアップ',
        '12': 'ニャンピューター',
        '13': 'ニャンピューター',
        '14': 'XP10000',
        '15': 'XP10000',
        '16': 'XP10000',
        '17': 'XP30000',
        '18': 'XP30000',
        '19': 'XP30000',
    },
    {
        'count': 3,
        'rarity_max': 9499,
        '0': 'ネコボン',
        '1': 'おかめはちもく',
        '2': 'スニャイパー',
    },
    {
        'count': 1,
        'rarity_max': 9999,
        '0': 'トレジャーレーダー',
    },
]

DUP_TABLE_INDEX = 0
DUP_COUNT = TABLES[DUP_TABLE_INDEX]['count']

# 各itemにユニークIDを振る辞書を作成
item_to_id = {}
id_counter = 0
for tidx, tbl in enumerate(TABLES):
    for sidx in range(tbl["count"]):
        name = tbl[str(sidx)]
        if name not in item_to_id:
            item_to_id[name] = id_counter
            id_counter += 1

# numba互換版で高速化したitemエンコード関数
@njit(inline='always')
def encode_table_item(table_idx: int, slot: int) -> int:
    # itemをユニーク整数IDにマッピング (ツール内IDを直接使う方式に変更不可なため仮実装)
    if table_idx == 0:
        return slot
    elif table_idx == 1:
        return 20 + slot
    else:
        return 23  # トレジャーレーダー


@njit(inline='always')
def xorshift32_u32(x: np.uint32) -> np.uint32:
    x ^= (x << 13) & U32_MASK
    x ^= (x >> 17)
    x ^= (x << 15) & U32_MASK
    return x & U32_MASK


@njit(inline='always')
def select_table_and_count(rarity: int) -> (int, int):
    if rarity <= 7399:
        return 0, 20
    elif rarity <= 9499:
        return 1, 3
    else:
        return 2, 1


@njit(inline='always')
def calc_alt_slot(seed: int, count: int, original_slot: int) -> int:
    alt_slot = xorshift32_u32(seed) % (count - 1)
    if original_slot <= alt_slot:
        alt_slot += 1
    return alt_slot


# 実際のガチャシミュレーション。早期打ち切り付き高速化用。
@njit
def simulate_gacha(seed: int, rolls: int, target_encoded: np.ndarray) -> bool:
    track_a_item = -1
    track_b_item = -1
    current_track = 0  # 0=A, 1=B
    prev_track = 0
    tr_prev_item = -1

    s = seed
    idx = 0

    for i in range(rolls):
        rarity = s % 10000
        s2 = xorshift32_u32(s)
        table_idx, count = select_table_and_count(rarity)
        slot = s2 % count
        item_id = encode_table_item(table_idx, slot)

        # 被り判定等のトラック遷移処理
        prev_item = track_a_item if current_track == 0 else track_b_item

        if table_idx == DUP_TABLE_INDEX and prev_item == item_id:
            alt_slot = calc_alt_slot(s2, count, slot)
            item_id = encode_table_item(table_idx, alt_slot)
            current_track = 1 - current_track
            tr_prev_item = item_id
        elif tr_prev_item == item_id and table_idx == DUP_TABLE_INDEX:
            alt_slot = calc_alt_slot(s2, count, slot)
            item_id = encode_table_item(table_idx, alt_slot)
            current_track = 1 - current_track
            tr_prev_item = item_id

        if item_id != target_encoded[idx]:
            return False

        # 前アイテム記録
        if current_track == 0:
            track_a_item = item_id
        else:
            track_b_item = item_id

        s = s2
        idx += 1

    return True


# チャンクごとにシード探索を並列実行するワーカー
def worker_search(args):
    start_seed, end_seed, target_encoded = args
    rolls = len(target_encoded)
    target_encoded_np = np.array(target_encoded, dtype=np.int32)

    for seed in range(start_seed, end_seed):
        # simulate_gachaはnumba jit高速化済み
        if simulate_gacha(seed, rolls, target_encoded_np):
            return seed
    return None


def find_seed_parallel(target_items: List[str], max_seed=2**32, chunk_size=10_000_000):
    # target_itemsを整数IDに変換
    target_encoded = [item_to_id[x] for x in target_items]

    # 並列処理用チャンク区切り作成
    num_cpu = cpu_count()
    print(f"CPU cores detected: {num_cpu}")
    start_time = time.time()

    for chunk_start in range(1, max_seed, chunk_size):
        chunk_end = min(chunk_start + chunk_size, max_seed)
        print(f"Searching seeds from {chunk_start} to {chunk_end - 1} ...")

        # マルチプロセスでチャンクを分割
        ranges = []
        step = max((chunk_end - chunk_start) // num_cpu, 1)
        for i in range(num_cpu):
            s = chunk_start + i * step
            e = min(s + step, chunk_end)
            if s >= e:
                break
            ranges.append((s, e, target_encoded))

        with Pool(processes=num_cpu) as pool:
            results = pool.map(worker_search, ranges)

        found_seeds = [r for r in results if r is not None]
        if found_seeds:
            elapsed = time.time() - start_time
            print(f"Found seed(s): {found_seeds} in {elapsed:.2f} seconds.")
            return found_seeds[0]

    elapsed = time.time() - start_time
    print(f"No seed found. Elapsed time: {elapsed:.2f} seconds.")
    return None


if __name__ == "__main__":
    observed = [
        'XP30000','トレジャーレーダー','XP10000','ちびタンクネコ','おかめはちもく',
        'XP30000','ちびネコノトリ','ちびウシネコ','ネコボン','ちびウシネコ',
        'ちびキモネコ','ちびネコノトリ','スピードアップ','XP30000','スピードアップ',
        'XP30000','XP10000','XP30000','ネコボン','ちび巨神ネコ','XP30000',
        'ちびタンクネコ','XP30000','ネコボン','スピードアップ','XP10000'
    ]

    start_overall = time.time()
    found_seed = find_seed_parallel(observed, max_seed=2**32, chunk_size=10_000_000)
    end_overall = time.time()

    if found_seed is not None:
        print(f"Initial seed found: {found_seed}")
    else:
        print("Initial seed not found in full search space.")
    print(f"Total elapsed time: {end_overall - start_overall:.2f} seconds")
