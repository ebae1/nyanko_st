from typing import List, Optional, Tuple

import z3

# --- 定数定義 ---
U32_MASK = 0xffffffff
SEQUENCE = [1380764099, 2907196537]

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
def xorshift32_z3(x: z3.BitVecRef) -> z3.BitVecRef:
    """Z3 用の xorshift32"""
    x = x ^ (x << 13)
    x = x ^ z3.LShR(x, 17)
    x = x ^ (x << 15)
    return x


def xorshift32_u32(x: int) -> int:
    """32bit整数用 xorshift32"""
    x &= U32_MASK
    x ^= (x << 13) & U32_MASK
    x ^= (x >> 17)
    x ^= (x << 15) & U32_MASK
    return x & U32_MASK


# --- シード探索 ---
def find_seed(sequence: List[int]) -> Optional[int]:
    """乱数列から内部シードを探索し、ユニーク性を検証"""
    seed_var = z3.BitVec('seed_var', 32)
    solver = z3.Solver()
    solver.add(seed_var != 0)

    current = seed_var
    for val in sequence:
        current = xorshift32_z3(current)
        solver.add(current == z3.BitVecVal(val, 32))

    if solver.check() == z3.sat:
        model = solver.model()
        seed_val = model[seed_var].as_long()
        print(f'seed: {seed_val}')

        #重解チェック
        solver.push()
        solver.add(seed_var != z3.BitVecVal(seed_val, 32))
        result2 = solver.check()
        if result2 == z3.unsat:
            print('unique: yes (proved)')
        elif result2 == z3.sat:
            print('unique: no (another seed exists)')
            alt_model = solver.model()
            print(f'another seed: {alt_model[seed_var].as_long()}')
        else:
            print(f'unique: unknown: {solver.reason_unknown()}')
        solver.pop()
        return seed_val
    else:
        print("unsat (一致するシードなし)")
        return None


def select_table_idx(rarity: int) -> tuple[int, int]:
    """
    レアリティ値に応じたテーブル番号とテーブル内の項目数を返す。
    """
    for idx, tbl in enumerate(TABLES):
        if rarity <= tbl['rarity_max']:
            return idx, tbl['count']
    raise ValueError('Rarity out of range')


def calc_alt_slot(seed, count, original_slot):
    """
    代替スロット計算。slotが重複する場合に使う。
    """
    alt_slot = xorshift32_u32(seed) % (count - 1)
    if original_slot <= alt_slot:
        alt_slot += 1
    return alt_slot


def rarity_and_slot(initial_seed: int, rolls: int) -> Tuple[List[dict], List[dict]]:
    """
    指定回数のロールに対して、レアリティ・テーブル選択・スロット選択を行い、
    トラックA、トラックBのそれぞれに結果を記録して返す。
    """
    track_a: List[dict] = []
    track_b: List[dict] = []
    seed = initial_seed

    for i in range(rolls * 2):
        rarity = seed % 10000
        next_seed = xorshift32_u32(seed)
        table_idx, count = select_table_idx(rarity)
        slot = next_seed % count
        item = TABLES[table_idx][slot]

        track = track_a if i % 2 == 0 else track_b
        track_label = 'A' if track is track_a else 'B'
        number = (i + 2) // 2

        entry = {
            'track': track_label,
            'No.': f'{number}{track_label}',
            'seed1': seed,
            'seed2': next_seed,
            'rarity': rarity,
            'table_idx': table_idx,
            'slot': slot,
            'item': item,
        }
        # 直前のアイテムを取得（必要に応じて）
        prev_item = track[-1]['item'] if track else None

        # 重複判定と代替アイテム設定
        if prev_item and table_idx == DUP_TABLE_INDEX and item == prev_item:
            alt_slot = calc_alt_slot(next_seed, count, slot)
            alt_item = TABLES[table_idx][alt_slot]
            entry.update({'alt_slot': alt_slot, 'alt_item': alt_item})

        track.append(entry)
        seed = next_seed

    return track_a, track_b


def dup_stream(table_idx: int, slot: int, seed2: int) -> Tuple[int, str]:
    """
    被りが発生した際の代替スロット及びアイテムを取得。
    """
    alt_slot = xorshift32_u32(seed2) % (DUP_COUNT - 1)
    if slot <= alt_slot:
        alt_slot += 1
    alt_item = TABLES[table_idx][alt_slot]

    return alt_slot, alt_item


def stream(track: str) -> List[dict]:
    """指定したトラックから開始した移動を含むガチャ結果"""
    # 初期シードから1ステップ進めた値を取得
    next_seed = xorshift32_u32(initial_seed)
    track_A_result, track_B_result = rarity_and_slot(next_seed, rolls=200)

    stream_result = []
    current_track = track
    prev_track = track
    tr_prev_item = None

    #alt_itemが発生した後は、もう片方のtrackへジャンプする
    for i in range(len(track_A_result)):
        # BからAへの移動時は1回スキップ
        if current_track == 'A' and prev_track == 'B':
            prev_track = 'A'
            continue

        track_result = track_A_result if current_track == 'A' else track_B_result
        alt_track = 'B' if current_track == 'A' else 'A'

        prev_track = current_track
        entry: dict = {
            'No': track_result[i]['No.'],
            'item': track_result[i]['item'],
        }
        stream_result.append(entry)

        item = track_result[i]['item']

        #トラック移動後の場合
        if len(stream_result) >= 2 and alt_track in stream_result[-2]['No']:
            #被りあり
            if track_result[i]['table_idx'] == DUP_TABLE_INDEX and item == tr_prev_item:

                tr_alt_slot, tr_alt_item = dup_stream(
                    track_result[i]['table_idx'],
                    track_result[i]['slot'],
                    track_result[i]['seed2'],
                )
                entry.update({'tr_alt_slot': tr_alt_slot, 'tr_alt_item': tr_alt_item})
                current_track = alt_track  #トラック移動
                tr_prev_item = tr_alt_item
                prefix = ' + + '
            #被りなし
            else:
                tr_prev_item = item
                prefix = ''

        #alt_itemがある場合
        elif 'alt_item' in track_result[i]:
            alt_item = track_result[i]['alt_item']
            entry['alt_item'] = alt_item
            current_track = alt_track  #トラック移動
            tr_prev_item = alt_item
            prefix = ' + '
        #alt_itemがない場合
        else:
            tr_prev_item = item
            prefix = ''

        entry['result'] = prefix + tr_prev_item
        print(entry)

    return stream_result


# --- 実行部 ---
if __name__ == '__main__':
    initial_seed = find_seed(SEQUENCE)
    if initial_seed is None:
        exit(1)

    print('\n--- Stream A ---')
    stream_from_a = stream('A')

    print('\n--- Stream B ---')
    stream_from_b = stream('B')


def gacha_result(track: str):
    stream_result = stream_from_a if track == 'A' else stream_from_b
    results = [item['result'] for item in stream_result]
    print('\n'.join(results))
    return results


print('\n-------A-------')
gacha_result('A')

print('\n-------B-------')
gacha_result('B')
