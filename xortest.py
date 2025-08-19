import z3

sequence = [3704151401,134749915,3058334514,3674097572,1968852641,83718374,1479206294,393553882,2894118429,446514108]

def xorshift32(x):
    x = x ^ (x << 13)
    x = x ^ z3.LShR(x, 17)
    x = x ^ (x << 15)
    return x

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