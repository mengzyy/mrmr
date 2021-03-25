import sys
import os
import random
import numpy as np
import scipy as sp
from operator import itemgetter
from data.secom.usb import readdata


def entropy(vec, base=2):
    " Returns the empirical entropy H(X) in the input vector."
    _, vec = np.unique(vec, return_counts=True)
    prob_vec = np.array(vec / float(sum(vec)))
    if base == 2:
        logfn = np.log2
    elif base == 10:
        logfn = np.log10
    else:
        logfn = np.log
    return prob_vec.dot(-logfn(prob_vec))


def conditional_entropy(x, y):
    "Returns H(X|Y)."
    uy, uyc = np.unique(y, return_counts=True)
    prob_uyc = uyc / float(sum(uyc))
    cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
    return prob_uyc.dot(cond_entropy_x)


def mutual_information(x, y):
    return entropy(x) - conditional_entropy(x, y)


def symmetrical_uncertainty(x, y):
    return 2.0 * mutual_information(x, y) / (entropy(x) + entropy(y))


# 改进后的mrmr算法 infomtrix
# 1. 根据su进行排序
# 2.
def improved_mrmr(infomtrix):
    score = []
    # 先根据su进行排名
    feacount = infomtrix.shape[1] - 1
    for i in range(0, feacount):
        score.append(symmetrical_uncertainty(infomtrix[:, i], infomtrix[:, -1]))
    # 标记数组 -1 则代表以及被剔除
    sorttuple = []
    for i in range(0, feacount):
        sorttuple.append((score[i], i))
    sorttuple.sort(key=itemgetter(0), reverse=True)
    M = []
    book = [0] * feacount
    for i in range(0, feacount):
        ind = sorttuple[i][1]
        if book[ind] == 0:
            # add
            M.append(ind)
            book[ind] = -1
            print("select success:", ind)
            # 向后迭代
            for j in range(i+1, feacount):
                indj = sorttuple[j][1]
                if book[indj] == 0:
                    mutal1 = 0
                    mutal2 = 0
                    for m in M:
                        mutal1 += mutual_information(infomtrix[:, m], infomtrix[:, ind])
                    for m in M:
                        mutal2 += mutual_information(infomtrix[:, m], infomtrix[:, indj])
                    if mutal2 > mutal1:
                        # 如果冗余性 大余 则剔除
                        book[indj] = -1
    return M


if __name__ == '__main__':
    # vec1 = np.linspace(1, 20, 20)
    # print("Vec 1:", vec1)
    # print("Entropy:", entropy(vec1))
    #
    # vec2 = np.tile([4, 5, 6, 7], 5)
    # print("Vec 2:", vec2)
    # print("Entropy:", entropy(vec2))
    #
    # mi = mutual_information(vec1, vec2)
    # print("Mutual information: {0}".format(mi))
    #
    # su = symmetrical_uncertainty(vec1, vec2)
    # print("Symmetrical uncertainty: {0}".format(su))
    fea, label = readdata()
    le = len(label)
    for i in range(0, le):
        fea[i].append(label[i])
    feaarray = np.array(fea)
    feaarray[np.isnan(feaarray)] = 0.0
    l = improved_mrmr(feaarray)
    print(l)
