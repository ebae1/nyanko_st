import z3

sequence = [3704151401,134749915,3058334514,3674097572,1968852641,83718374,1479206294,393553882,2894118429,446514108]

# Z3 用（BitVec）
def xorshift32(x):
    x = x ^ (x << 13)
    x = x ^ z3.LShR(x, 17)
    x = x ^ (x << 15)
    return x

# 具体値用（Python int, 32bit で丸め込み）
def xorshift32_u32(x):
    x &= 0xffffffff
    x ^= (x << 13) & 0xffffffff
    x ^= (x >> 17)
    x ^= (x << 15) & 0xffffffff
    return x & 0xffffffff

s0 = z3.BitVec('s0', 32)
solver = z3.Solver()
solver.add(s0 != 0)

s = s0
for i in sequence:
    s = xorshift32(s)
    solver.add(s == z3.BitVecVal(i, 32))

if solver.check() == z3.sat:
    model = solver.model()
    seed = model[s0].as_long()
    print("seed:", seed)
    
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
        
else:
    print("unsat (一致するシードなし)")
    
    
    
def rarity_and_score(seed_1,rolls):
    table1 = {'count':20,'rarity_max':7399}
    table2 = {'count':3,'rarity_max':9499}
    table3 = {'count':1,'rarity_max':9999}
    
    track_A = []
    track_B = []
    tablelist =[]
    
    for i in range(rolls):
                
        seed_1 = seed_1 & 0xffffffff
        rarity = seed_1 % 10000
        
        seed_2 =  xorshift32_u32(seed_1)
        
        if rarity <= table1['rarity_max']:
            count = table1['count']
            table = 1
            score = seed_2 % count
        elif rarity <= table2['rarity_max']:
            count = table2['count']
            score = seed_2 % count
            table = 2
        elif rarity <= table3['rarity_max']:
            count = table3['count']
            score = seed_2 % count
            table = 3
        
        
        #被り処理 一応OK   
        # 課題 → スコアとアイテム名の対応表に基づいてアイテム名で被り判断処理
        
        if i % 2 ==0:
            if len(track_A) >= 1 and table == 1 and track_A[-1][1] == 1 and score == track_A[-1][2]:
                score_alt = xorshift32_u32(seed_2) % (count-1)
                if score <= score_alt:
                    score_alt = score_alt + 1

                track_A.append([rarity,table,score,score_alt])
            else:
                track_A.append([rarity,table,score])
           
            
        else:
            if len(track_B) >= 1 and table == 1 and track_B[-1][1] == 1 and score == track_B[-1][2]:
                score_alt = xorshift32_u32(seed_2) % (count-1)
                if score <= score_alt:
                    score_alt = score_alt + 1
                
                track_B.append([rarity,table,score,score_alt])
            else:
                track_B.append([rarity,table,score])
                
                
        seed_1 = seed_2
        
    return track_A, track_B

a,b = rarity_and_score(2894118429,20)
print(a)
print(b)