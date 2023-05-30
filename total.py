import numpy as np
import random
import math
from liblinear.liblinearutil import *
import itertools

d = 10
Q = 4
C_list = [0.0000005, 0.0005, 0.5, 500, 500000]    #C = 1 / 2 * lambda

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

# X_test_tran = np.zeros([test_N, 1001])
# for i in range(test_N): 
#     A = X_test[i].tolist()
#     B = X_test[i][1:].tolist()
#     for j in range(2,5):
#         C = itertools.combinations_with_replacement(B, j)
#         for k in C:
#             a = 1
#             for q in k:
#                 a *= q
#             A.append(a)
#     X_test_tran[i] = np.array(A)

prob = problem(y_test, X_test_tran)
for i in range(5):
    param = parameter(f'-s 0 -c {(C_list[i])} -e 0.000001 -q')   #try for C_list[0,1,2,3,4]
    m = train(prob, param)
    p_labs, p_acc, p_vals = predict()
# model_ptr = liblinear.train(prob, param)
# model_ = toPyModel(model_ptr)
# [W_out, b_out] = model_.get_decfun()

# E_out = 0
# for i in range(test_N):
#     if np.sign(np.dot(W_out, X_test_tran[i])) != y_test[i]:
#         E_out += 1
# E_out = E_out / test_N

# print(E_out)
