import numpy as np
import pandas as pd
import funs as Ifuns
from KSUMS import KSUMS
import matplotlib.pyplot as plt

knn = 1000

# load date
# X, y_true, N, dim, c_true=Ifuns.load_csv('./data/testdata_4class.csv')
# X, y_true, N, dim, c_true=Ifuns.load_har('./data/har')
# X, y_true, N, dim, c_true=Ifuns.load_pendigits('./data/pendigits')
X, y_true, N, dim, c_true=Ifuns.load_usps('./data/usps')
print(N, dim, c_true)

# c_true=3

D = Ifuns.EuDist2(X, X, squared=True)
np.fill_diagonal(D, -1)
ind_M = np.argsort(D, axis=1)
np.fill_diagonal(D, 0)

NN = ind_M[:, :knn]
NND = Ifuns.matrix_index_take(D, NN)

# print(ind_M)
# print(NN)
# print(NND)

# Clustering
obj = KSUMS(NN.astype(np.int32), NND, c_true)
obj.clu()
y_pred = obj.y_pre

# eval
acc = Ifuns.accuracy(y_true=y_true, y_pred=y_pred)
ari = Ifuns.ari(y_true=y_true, y_pred=y_pred)
nmi = Ifuns.nmi(y_true=y_true, y_pred=y_pred)

print(obj.time)
print("acc:",acc)
print("ari:",ari)
print("nmi:",nmi)


print(y_pred)

# for i in range(c_true):
#     x = []
#     y = []
#     for j in range(N):
#         if y_pred[j] == i:
#             x.append(X[j, 0])
#             y.append(X[j, 1])
#     plt.scatter(x, y)
#
# plt.show()