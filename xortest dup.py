from typing import List

import z3

U32_MASK = 0xffffffff

SEQUENCE = [2224457259, 2991770626]

# テーブル定義
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


# Z3 用（BitVec）
def xorshift32_z3(x):
    """
    Z3用のxorshift32シフト演算。
    """
    x = x ^ (x << 13)
    x = x ^ z3.LShR(x, 17)
    x = x ^ (x << 15)
    return x


def xorshift32_u32(x):
    """
    具体値（32bit unsigned int）用のxorshift32。
    各演算は32bitに丸め込み。
    """
    x &= U32_MASK
    x ^= (x << 13) & U32_MASK
    x ^= (x >> 17)
    x ^= (x << 15) & U32_MASK
    return x & U32_MASK


def find_seed(sequence):
    """
    Z3を使って与えられた乱数列の内部シードを探索する。
    ユニーク性も判定。
    """
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


def select_table_n(rarity: int) -> tuple[int, int]:
    """
    レアリティ値に応じたテーブル番号とテーブル内の項目数を返す。
    """
    for idx, tbl in enumerate(TABLES):
        if rarity <= tbl['rarity_max']:
            return idx, tbl['count']
    raise ValueError('Rarity out of range')


def calc_alt_slot(seed,count,original_slot):
    """
    代替スロット計算。slotが重複する場合に使う。
    """
    alt_slot = xorshift32_u32(seed) % (count-1)
    if original_slot <= alt_slot:
        alt_slot += 1
    return alt_slot

def rarity_and_slot(initial_seed, rolls) -> tuple[list[dict],list[dict]]:
    """
    指定回数のロールに対して、レアリティ・テーブル選択・スロット選択を行い、
    トラックA、トラックBのそれぞれに結果を記録して返す。
    """
    track_a: List[dict] = []
    track_b: List[dict] = []
    
    seed = initial_seed
    for i in range(rolls):
        rarity = seed % 10000
        next_seed = xorshift32_u32(seed)
        table_idx, count = select_table_n(rarity)
        slot = next_seed % count
        item = TABLES[table_idx][str(slot)]

        current_track = track_a if i % 2 == 0 else track_b
        track_label = 'A' if current_track is track_a else 'B'
        number = (i + 2) // 2
        
        entry = {
            'track': track_label,
            'seed1': seed,
            'seed2': next_seed,
            'rarity': rarity,
            'table_n': table_idx,
            'slot': slot,
            'No.': f'{number}{track_label}',
            'item': item,
        }
        # 直前のアイテムを取得（必要に応じて）
        prev_item = current_track[-1]['item'] if len(current_track) >= 1 else None
        
        # 重複判定と代替アイテム設定
        if len(current_track) >= 1 and table_idx == DUP_TABLE_INDEX and item == prev_item:
            alt_slot = calc_alt_slot(next_seed, count, slot)
            alt_item = TABLES[table_idx][str(alt_slot)]
            entry.update({'alt_slot':alt_slot, 'alt_item':alt_item})
            
        current_track.append(entry)
        seed = next_seed
        
    return track_a, track_b


def dup_stream(table_n, slot, seed2) -> tuple[int, str]:
    """
    被りが発生した際の代替スロット及びアイテムを取得。
    """
    alt_slot = xorshift32_u32(seed2) % (DUP_COUNT - 1)
    if slot <= alt_slot:
        alt_slot += 1
    alt_item = TABLES[table_n][str(alt_slot)]

    return alt_slot, alt_item

if __name__ == '__main__':
    initial_seed = find_seed(SEQUENCE)
    if initial_seed is None:
        exit(1)
        
    # 初期シードから1ステップ進めた値を取得
    next_seed = xorshift32_u32(initial_seed)    
    track_A_result, track_B_result = rarity_and_slot(next_seed, rolls=400)

    stream_from_track_A = []
    current_track = 'A'
    prev_track = 'A'
    tr_prev_item = None
        
                

    #alt_itemが発生した後は、もう片方のtrackへジャンプする
    for i in range(len(track_A_result)):
        if current_track == 'A':
            # Bから移動してきた場合、1回スキップ
            if prev_track == 'B':
                prev_track = 'A'
                continue

            prev_track = 'A'
            num = track_A_result[i]['No.']
            item = track_A_result[i]['item']
            stream_from_track_A.append({'No':num, 'item':item})

            #トラック移動後の被り処理
            if (len(stream_from_track_A) >= 2 and 'B' in stream_from_track_A[-2][0]

                if (len(stream_from_track_A) >= 2 and 'B' in stream_from_track_A[-2][0] and
                        track_A_result[i]['table_n'] == DUP_TABLE_INDEX and item == tr_prev_item):

                    tr_alt_slot, tr_alt_item = dup_stream(track_A_result[i]['table_n'],
                                                            track_A_result[i]['slot'],
                                                            track_A_result[i]['seed2'])
                    stream_from_track_A[-1].update({'tr_alt_slot':tr_alt_slot, 'tr_alt_item':tr_alt_item})
                    current_track = 'B'  #トラック移動
                    tr_prev_item = tr_alt_item

                elif 'alt_item' in track_A_result[i]:
                    alt_item = track_A_result[i]['alt_item']
                    stream_from_track_A[-1].update({'alt_item':alt_item})
                    current_track = 'B'
                    tr_prev_item = alt_item
                
            print(stream_from_track_A[-1])

        else: # current_track == 'B'
            prev_track = 'B'
            num = track_B_result[i]['No.']
            item = track_B_result[i]['item']
            stream_from_track_A.append({'num':num, 'item':item})

            #トラック移動後の被り処理
            if (len(stream_from_track_A) >= 2 and 'A' in stream_from_track_A[-2][0] and
                    track_B_result[i]['table_n'] == DUP_TABLE_INDEX and item == tr_prev_item):

                tr_alt_slot, tr_alt_item = dup_stream(track_B_result[i]['table_n'],
                                                        track_B_result[i]['slot'],
                                                        track_B_result[i]['seed2'])
                stream_from_track_A[-1].append({'tr_alt_slot':tr_alt_slot, 'tr_alt_item':tr_alt_item})
                current_track = 'A'  #トラック移動
                tr_prev_item = tr_alt_item

            elif 'alt_item' in track_B_result[i]:
                alt_item = track_B_result[i]['alt_item']
                stream_from_track_A[-1].update({'alt_item':alt_item})
                current_track = 'A'  #トラック移動
                tr_prev_item = alt_item

            print(stream_from_track_A[-1])
print()

gacha_result = []
for item in stream_from_track_A:
    if 'tr_alt_item' in item:
        result = item['tr_alt_item']
    elif 'alt_item' in item:
        result = item['alt_item']
    else:
        result = item['item']
    print(gacha_result[-1])
