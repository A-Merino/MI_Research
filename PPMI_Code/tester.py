from mutual_info import calc_MI
import numpy as np



# for i in range(10,26):
#     t = calc_MI(([0, 0, 10, 6, 0], [0, 0, 1, 2, 0], i))
#     print(t)


print(calc_MI(([4, 6, 10, 6, 0], [3, 2, 1, 2, 0], 26)))