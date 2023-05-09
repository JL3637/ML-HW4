import numpy as np
import random
import math
from liblinear.liblinearutil import *

d = 10
Q = 4
filename2 = 'test.txt'
test_N = 500
X_test = np.ones([test_N, d+1])
y_test = np.zeros(test_N)
with open(filename2) as file:
    a = 0
    for line in file:
        line = line.strip().split()
        for i in range(d):
            X_test[a][i+1] = line[i]
        X_test[a][0] = 1
        y_test[a] = line[d]
        a += 1
file.close()

X_test_tran = np.ones([test_N, 1001])
for i in range(test_N):
    cnt = 1
    for j in range(d):
        X_test_tran[i][cnt] = X_test[i][j+1]
        cnt += 1
    x_tmp1 = X_test_tran[i][1:11]
    x_tmp_index = np.arange(d+1)
    x_tmp_index_n = np.arange(d+1)
    for j in range(d):
        x_tmp2 = X_test_tran[i][j+1] * x_tmp1[x_tmp_index[j]:]
        x_tmp_index_n[j+1] = np.size(x_tmp2) + x_tmp_index_n[j]
        for k in range(np.size(x_tmp2)):
            X_test_tran[i][cnt] = x_tmp2[k]
            cnt += 1
    x_tmp1 = X_test_tran[i][11:66]
    x_tmp_index_nn = np.arange(d+1)
    for j in range(d):
        x_tmp2 = X_test_tran[i][j+1] * x_tmp1[x_tmp_index_n[j]:]
        x_tmp_index_nn[j+1] = np.size(x_tmp2) + x_tmp_index_nn[j]
        for k in range(np.size(x_tmp2)):
            X_test_tran[i][cnt] = x_tmp2[k]
            cnt += 1
    x_tmp1 = X_test_tran[i][66:286]
    x_tmp_index_nnn = np.arange(d+1)
    for j in range(d):
        x_tmp2 = X_test_tran[i][j+1] * x_tmp1[x_tmp_index_nn[j]:]
        x_tmp_index_nnn[j+1] = np.size(x_tmp2) + x_tmp_index_nnn[j]
        for k in range(np.size(x_tmp2)):
            X_test_tran[i][cnt] = x_tmp2[k]
            cnt += 1

print(X_test_tran)


# prob = problem(y_array_test, x_vector_array_test)
# param = parameter('-s 0 -c 1 -e 0.000001')
# model_ptr = liblinear.train(prob, param)
# model_ = toPyModel(model_ptr)
# [W, b] = model_.get_decfun()
# print(W)
# print(b)
