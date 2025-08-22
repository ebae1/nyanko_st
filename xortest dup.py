from typing import List

import z3

U32_MASK = 0xffffffff

SEQUENCE = [
    3704151401,
    134749915,
    3058334514,
    3674097572,
    1968852641,
    83718374,
    1479206294,
    393553882,
    2894118429,
    446514108,
    4287503049,
    2928026786,
    1120391595,
    3457936850,
    1235284994,
    417963922,
    3457436191,
    2207091124,
    3134165225,
    43160704,
    239221889,
    1441151080,
    711323933,
    2581142334,
    4186834529,
    606223040,
]


# Z3 用（BitVec）
def xorshift32_z3(x):
    """
    Z3用のxorshift32。
    Shift演算がz3流に。
    """
    x = x ^ (x << 13)
    x = x ^ z3.LShR(x, 17)
    x = x ^ (x << 15)
    return x


def xorshift32_u32(x):
    """
    具体値（32bit int）用のxorshift32。
    32bitで丸め込み。
    """
    x &= U32_MASK
    x ^= (x << 13) & U32_MASK
    x ^= (x >> 17)
    x ^= (x << 15) & U32_MASK
    return x & U32_MASK


def find_seed(sequence):
    """
    シード値をZ3で探索し、ユニーク判定まで行う。
    """
    s0 = z3.BitVec('s0', 32)
    solver = z3.Solver()
    solver.add(s0 != 0)

    s = s0
    for i in sequence:
        s = xorshift32_z3(s)
        solver.add(s == z3.BitVecVal(i, 32))

    if solver.check() == z3.sat:
        model = solver.model()
        seed = model[s0].as_long()
        print('seed:', seed)

        #重解チェック
        solver.push()
        solver.add(s0 != z3.BitVecVal(seed, 32))
        res2 = solver.check()
        if res2 == z3.unsat:
            print('unique: yes (proved)')
        elif res2 == z3.sat:
            print('unique: no (another seed exists)')
            m2 = solver.model()
            print('another seed:', m2[s0].as_long())
        else:
            print('unique: unknown:', solver.reason_unknown())
        solver.pop()
        return seed
    else:
        print("unsat (一致するシードなし)")
        return None


# テーブル定義
table_nS = [
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


def select_table_n(rarity):
    for idx, tbl in enumerate(table_nS, start=1):
        if rarity <= tbl['rarity_max']:
            return idx, tbl['count']
    raise ValueError('Rarity out of range')


def is_duplicate(track, table_n, item):
    # trackが同じ、かつtable_nが1、かつ、itemが前回結果のitem_alt（item_altが存在しない場合はitem）と同じかどうか
    return (len(track) != 0 and table_n == 1 and item == track[-1][-1])


def rarity_and_score(seed_1, rolls):
    track_a: List[List[int]] = []
    track_b: List[List[int]] = []
    for i in range(rolls):

        rarity = seed_1 % 10000
        seed_2 = xorshift32_u32(seed_1)
        table_n, count = select_table_n(rarity)
        score = seed_2 % count
        item = table_nS[table_n - 1][str(score)]

        current_track = track_a if i % 2 == 0 else track_b
        if is_duplicate(current_track, table_n, item):
            score_alt = xorshift32_u32(seed_2) % (count - 1)
            if score <= score_alt:
                score_alt += 1
            item_alt = table_nS[table_n - 1][str(score_alt)]
            current_track.append([rarity, table_n, score, item, score_alt, item_alt])

        else:
            current_track.append([
                rarity,
                table_n,
                score,
                item,
            ])
        # print(seed_1,current_track)
        seed_1 = seed_2
    return track_a, track_b


if __name__ == '__main__':
    seeds = xorshift32_u32(find_seed(SEQUENCE))  #初期シードから1ステップ進める
    if seeds is not None:
        track_A_result, track_B_result = rarity_and_score(seeds, rolls=100)
        print('track_A:', track_A_result)
        print()
        print('track_B:', track_B_result)
        print()

        current_track = 'A'
        stream_from_track_A = []
        stream_from_track_A.append(['1A', track_A_result[0][-1]])
        #item_altが発生した後は、もう片方のtrackへジャンプする
        for idx in range(1, len(track_A_result)):
            if current_track == 'A':
                stream_from_track_A.append([f'{idx+1}A', track_A_result[idx][-1]])
                if len(track_A_result[idx]) >= 5:
                    current_track = 'B'

            elif current_track == 'B':
                stream_from_track_A.append([f'{idx+1}B', track_B_result[idx][-1]])
                if len(track_B_result[idx]) >= 5:
                    current_track = 'A'

        print(stream_from_track_A)
