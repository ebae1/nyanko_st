#https://qiita.com/SatoshiTerasaki/items/e101d4c0e2e9e0e55663


def xor32(f: int) -> int:
    mask = 0xFFFFFFFF #32ビットの下位ビットだけを残すマスク
    f &= mask # 入力を32ビット符号なしに正規化
    f = f ^ (f << 13 & mask)
    f = f ^ (f >> 17 & mask)
    f = f ^(f << 15 & mask)
    result = f & mask # 念のため最終結果もマスク
    return result

f = 1657036013
seed = xor32(f)
print(seed)



#pip install z3-solver
#Z3で疑似乱数生成器(xorshift)の出力を予測する
#https://burion.net/entry/2023/09/24/232230
#numpy.randomの使い方
#https://note.nkmk.me/python-numpy-random/

import z3
import time
import numpy as np
rng = np.random.default_rng()


class xor32():
    def __init__(self,seed):
        self.s = seed
    
    def mask(self, x):
        return x & 0xffffffff #32ビットの下位ビットだけを残すマスク
    
    def rand(self):
        self.s = self.mask(self.s ^ (self.s << 13))
        self.s = self.mask(self.s ^ (self.s >> 17))
        self.s = self.mask(self.s ^ (self.s << 15))

        return self.s & mask

sequence = [rng.rand() for i in range(10)]

print('randoms', sequence)

print(f'next random is {rng.rand()}')

# st = time.time()
# solver = z3.Solver()

# s = z3.BitVec('x', 32)
# for i in sequence:
#     x = x