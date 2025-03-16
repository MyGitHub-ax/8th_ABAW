d_model = 8
import math
for pos in range(10):
    for i in range(4):  # 因d_model=4，i取0和1（对应2i=0,2）
        denominator = 10000 ** (2*i / d_model)
        PE_even = math.sin(pos / denominator)  # 偶数维度：0,2
        print('PE({},{})={}'.format(pos,2*i,PE_even),end=' ')
        PE_odd = math.cos(pos / denominator)   # 奇数维度：1,3
        print('PE({},{})={}'.format(pos, 2 * i+1,PE_odd),end=' ')
    print()