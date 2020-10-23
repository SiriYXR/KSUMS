import numpy as np
import funs as Ifuns
from KSUMS import KSUMS

knn = 20

# load date
X, y_true, N, dim, c_true = Ifuns.load_mat("/home/pei/DATA/Mpeg7_20200916.mat")
print(N, dim, c_true)

D = Ifuns.EuDist2(X, X, squared=True)
np.fill_diagonal(D, -1)
ind_M = np.argsort(D, axis=1)
np.fill_diagonal(D, 0)

NN = ind_M[:, :knn]
NND = Ifuns.matrix_index_take(D, NN)

# Clustering
obj = KSUMS(NN.astype(np.int32), NND, c_true)
obj.clu()
y_pred = obj.y_pre


# eval
acc = Ifuns.accuracy(y_true=y_true, y_pred=y_pred)
ari = Ifuns.ari(y_true=y_true, y_pred=y_pred)

print(obj.time)
print(acc)
print(ari)

