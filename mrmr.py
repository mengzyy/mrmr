from sklearn import metrics
import numpy as np
import warnings
from data.secom.usb import readdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
from random import randint
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


# 标准化计算互信息
# A，B分别为某特征下的样布的全部取值
# 返回A，B间聚类程度
def computeMutualInfo(A, B):
    return metrics.normalized_mutual_info_score(A, B)


# a step in mrmr
# mtrix : 原始特征数组 默认最后一列为类别【np】
# M 已被加入mrmr最优集合的特征index【list】
def selectFeature(M, mtrix):
    mcount = len(M)
    featurecount = mtrix.shape[1]
    maxindex = -1
    maxValue = -1
    for i in range(0, featurecount - 1):
        if i not in M:
            max1 = computeMutualInfo(mtrix[:, i], mtrix[:, -1])
            max2 = 0
            for index in M:
                max2 += computeMutualInfo(mtrix[:, index], mtrix[:, i])
            if mcount != 0:
                max2 /= mcount
            if maxValue < max1 + max2:
                maxindex = i
                maxValue = max1 + max2
    if maxindex != -1:
        M.append(maxindex)
    print("select success")
    return M


# mrmr 算法的python实现
# mrate 控制最优mrmr特征子集的大小 (0,1)
# mtrix 特征矩阵：注意仅仅以下标来记录特征类型 最后使用set来做一一对应
def mrmr(mrate, mtrix):
    # 存放最优特征集合
    M = []
    featurecount = (mtrix.shape[1]) - 1
    m = int(featurecount * mrate)
    for i in range(0, m):
        M = selectFeature(M, mtrix)
    return M


# do ml apolgo
def ml(mtrix, label, m):
    ss = StandardScaler()

    newfea = mtrix[:, m]
    X_train = newfea[0:800, :]
    y_train = np.asarray(label[0:800])
    X_val = newfea[800:-1, :]
    y_test_true = np.asarray(label[800:-1])
    model = RandomForestClassifier()
    X_train = ss.fit_transform(X_train)
    X_val = ss.fit_transform(X_val)


    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_val)
    auc = roc_auc_score(y_test_true, y_test_pred)
    f1 = f1_score(y_test_true, y_test_pred)
    print(classification_report(y_test_true, y_test_pred))
    print("AC", accuracy_score(y_test_true, y_test_pred))
    print("auc,f1",auc,f1)
    return auc, f1


if __name__ == '__main__':
    # A = [1, 1.3, 1, 1, 3, 1]
    # B = [1, 1, 1, 1.2, 3, 0]
    # C = [1, 4, 2, 6.2, 7, 1]
    # D = [1, 2, 3, 1.2, 4, 0]
    # E = [1, 5, 2, 2.2, 2, 1]
    # F = [1, 2, 1, 3.2, 1, 1]
    # matrix = []
    # matrix.append(A)
    # matrix.append(B)
    # matrix.append(C)
    # matrix.append(D)
    # matrix.append(E)
    # matrix.append(F)
    # matrix = np.array(matrix)
    # print(mrmr(0.4, matrix))
    fea, label = readdata()
    le = len(label)
    for i in range(0, le):
        fea[i].append(label[i])
    feaarray = np.array(fea)
    feaarray[np.isnan(feaarray)] = 0.0
    l = mrmr(0.3, feaarray)
    ml(feaarray,label,l)