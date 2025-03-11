from sklearn.metrics import mutual_info_score
import numpy as np
import matplotlib.pyplot as plt

def calc_MI(args):
    x, y, bins = args
    c_xy = np.histogram2d(x, y, bins)[0]
    # plt.plot(c_xy)
    # plt.plot(x, y)
    plt.imshow(c_xy)
    plt.show()
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi