import numpy as np
from numba import cuda, njit, int32, uint32

# ====== テーブル定義・前処理 ======
TABLES = [
    {
        'count': 20,
        'rarity_max': 7399,
        '0': 'ちび巨神ネコ', '1': 'ちびネコトカゲ', '2': 'ちびネコフィッシュ',
        '3': 'ちびネコノトリ', '4': 'ちびウシネコ', '5': 'ちびキモネコ', '6': 'ちびバトルネコNP',
        '7': 'ちびタンクネコ', '8': 'ちびネコNP', '9': 'スピードアップ', '10': 'スピードアップ',
        '11': 'スピードアップ', '12': 'ニャンピューター', '13': 'ニャンピューター', '14': 'XP10000',
        '15': 'XP10000', '16': 'XP10000', '17': 'XP30000', '18': 'XP30000', '19': 'XP30000',
    },
    {
        'count': 3,
        'rarity_max': 9499,
        '0': 'ネコボン', '1': 'おかめはちもく', '2': 'スニャイパー',
    },
    {
        'count': 1,
        'rarity_max': 9999,
        '0': 'トレジャーレーダー',
    },
]
DUP_TABLE_INDEX = 0
DUP_COUNT = TABLES[0]['count']
U32_MASK = 0xffffffff

# item→数値ID変換
item_to_id = {}
id_counter = 0
for tidx, tbl in enumerate(TABLES):
    for sidx in range(tbl["count"]):
        name = tbl[str(sidx)]
        if name not in item_to_id:
            item_to_id[name] = id_counter
            id_counter += 1

def encode_table_item(table_idx, slot):
    if table_idx == 0:
        return slot           # 0-19
    elif table_idx == 1:
        return 20 + slot     # 20-22
    else:
        return 23            # トレジャーレーダー

def select_table_and_count(rarity):
    if rarity <= 7399:
        return 0, 20
    elif rarity <= 9499:
        return 1, 3
    else:
        return 2, 1

# ====== CUDA用関数類 ======
@cuda.jit(device=True)
def xorshift32_cuda(x):
    x ^= (x << 13) & U32_MASK
    x ^= (x >> 17)
    x ^= (x << 15) & U32_MASK
    return x & U32_MASK

@cuda.jit(device=True)
def encode_table_item_cuda(table_idx, slot):
    if table_idx == 0:
        return slot
    elif table_idx == 1:
        return 20 + slot
    else:
        return 23

@cuda.jit(device=True)
def select_table_and_count_cuda(rarity):
    if rarity <= 7399:
        return 0, 20
    elif rarity <= 9499:
        return 1, 3
    else:
        return 2, 1

@cuda.jit(device=True)
def calc_alt_slot_cuda(seed, count, original_slot):
    alt_slot = xorshift32_cuda(seed) % (count - 1)
    if original_slot <= alt_slot:
        alt_slot += 1
    return alt_slot

# ====== GPU探索カーネル ======
@cuda.jit
def search_seed_kernel(start_seed, target, n_prefix, result_seeds, max_results):
    idx = cuda.grid(1)
    seed = start_seed + idx
    s = seed
    track_a_item = -1
    track_b_item = -1
    current_track = 0
    prev_track = 0
    tr_prev_item = -1
    matched = True

    for i in range(n_prefix):
        rarity = s % 10000
        s2 = xorshift32_cuda(s)
        table_idx, count = select_table_and_count_cuda(rarity)
        slot = s2 % count
        item_id = encode_table_item_cuda(table_idx, slot)

        prev_item = track_a_item if current_track == 0 else track_b_item

        # 被り処理/トラック移動
        altitem = False
        if table_idx == DUP_TABLE_INDEX and prev_item == item_id:
            alt_slot = calc_alt_slot_cuda(s2, count, slot)
            item_id = encode_table_item_cuda(table_idx, alt_slot)
            current_track = 1 - current_track
            tr_prev_item = item_id
            altitem = True
        elif tr_prev_item == item_id and table_idx == DUP_TABLE_INDEX:
            alt_slot = calc_alt_slot_cuda(s2, count, slot)
            item_id = encode_table_item_cuda(table_idx, alt_slot)
            current_track = 1 - current_track
            tr_prev_item = item_id
            altitem = True

        if item_id != target[i]:
            matched = False
            break

        if current_track == 0:
            track_a_item = item_id
        else:
            track_b_item = item_id
        s = s2

    # prefix一致seedだけ結果に追加
    if matched:
        # スレッドセーフな方法で配列にpush
        pos = cuda.atomic.add(result_seeds, 0, 1)  # result_seeds[0]が件数管理
        if pos < max_results:
            result_seeds[pos+1] = seed  # はヒット数管理、[1:]に候補seed

# ====== CPU側:後半完全一致検証=====
@njit
def simulate_gacha(seed, rolls, target):
    track_a_item = -1
    track_b_item = -1
    current_track = 0
    prev_track = 0
    tr_prev_item = -1
    s = seed
    idx = 0
    for i in range(rolls):
        rarity = s % 10000
        s2 = xorshift32_cuda(s)
        table_idx, count = select_table_and_count_cuda(rarity)
        slot = s2 % count
        item_id = encode_table_item_cuda(table_idx, slot)
        prev_item = track_a_item if current_track == 0 else track_b_item
        if table_idx == DUP_TABLE_INDEX and prev_item == item_id:
            alt_slot = calc_alt_slot_cuda(s2, count, slot)
            item_id = encode_table_item_cuda(table_idx, alt_slot)
            current_track = 1 - current_track
            tr_prev_item = item_id
        elif tr_prev_item == item_id and table_idx == DUP_TABLE_INDEX:
            alt_slot = calc_alt_slot_cuda(s2, count, slot)
            item_id = encode_table_item_cuda(table_idx, alt_slot)
            current_track = 1 - current_track
            tr_prev_item = item_id
        if item_id != target[idx]:
            return False
        if current_track == 0:
            track_a_item = item_id
        else:
            track_b_item = item_id
        s = s2
        idx += 1
    return True

# ====== 実行ファンクション ======
def search_seed_gpu(observed_items, n_prefix=8, batch_size=100_000_000):
    # 文字列→ID
    observed_ids = [item_to_id[x] for x in observed_items]
    observed_prefix = np.array(observed_ids[:n_prefix], dtype=np.int32)
    rolls = len(observed_ids)

    # GPU設定
    threads = 256
    blocks = batch_size // threads

    total_checked = 0
    found_seeds = []

    import time
    time0 = time.time()
    for batch_idx in range(0, 2**32, batch_size):
        print(f"Batch {batch_idx}~{batch_idx + batch_size - 1}...")
        # 結果配列（ヒット件数・seed保存用）
        max_results = 100_000
        result_seeds = np.zeros(max_results+1, dtype=np.uint32)
        d_target = cuda.to_device(observed_prefix)
        d_result_seeds = cuda.to_device(result_seeds)
        search_seed_kernel[blocks, threads](batch_idx, d_target, n_prefix, d_result_seeds, max_results)
        cuda.synchronize()

        d_result_seeds.copy_to_host(result_seeds)
        n_found = result_seeds[0]
        seeds_found = result_seeds[1:n_found+1]
        print(f"  Prefix一致候補: {n_found}件")
        if n_found > 0:
            for cseed in seeds_found:
                if simulate_gacha(cseed, rolls, np.array(observed_ids, dtype=np.int32)):
                    print(f"\n!!! 完全一致seed: {cseed} !!!")
                    found_seeds.append(cseed)
        total_checked += batch_size
        print(f"Checked: {total_checked}")
        if found_seeds:
            break
        if time.time() - time0 > 3600*12:  # 12h超過なら自動break
            print("タイムリミット。")
            break
    elapsed = time.time() - time0
    print(f"=== 完了: 計測時間 {elapsed:.2f}秒 ({elapsed/60:.2f}分) ===")
    return found_seeds

# ====== MAIN ======
if __name__ == '__main__':
    # サンプル用: 25件
    observed = [
        'XP30000','トレジャーレーダー','XP10000','ちびタンクネコ','おかめはちもく',
        'XP30000','ちびネコノトリ','ちびウシネコ','ネコボン','ちびウシネコ',
        'ちびキモネコ','ちびネコノトリ','スピードアップ','XP30000','スピードアップ',
        'XP30000','XP10000','XP30000','ネコボン','ちび巨神ネコ','XP30000',
        'ちびタンクネコ','XP30000','ネコボン','スピードアップ','XP10000'
    ]
    res = search_seed_gpu(observed, n_prefix=8, batch_size=20_000_000)
    print("一致seed:", res)
