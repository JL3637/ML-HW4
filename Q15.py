import numpy as np
import random
import math
from liblinear.liblinearutil import *

d = 10
Q = 4
C_list = [0.0000005, 0.0005, 0.5, 500, 500000]    #C = 1 / 2 * lambda

filename1 = 'train.txt'
train_N = 200
X = np.ones([train_N, d+1])
y = np.zeros(train_N)
with open(filename1) as file:
    a = 0
    for line in file:
        line = line.strip().split()
        for i in range(d):
            X[a][i+1] = line[i]
        X[a][0] = 1
        y[a] = line[d]
        a += 1
file.close()

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

X_tran = np.ones([train_N, 1001])
for i in range(train_N):
    cnt = 1
    for j in range(d):
        X_tran[i][cnt] = X[i][j+1]
        cnt += 1
    x_tmp1 = X_tran[i][1:11]
    x_tmp_index = np.arange(d+1)
    x_tmp_index_n = np.arange(d+1)
    for j in range(d):
        x_tmp2 = X_tran[i][j+1] * x_tmp1[x_tmp_index[j]:]
        x_tmp_index_n[j+1] = np.size(x_tmp2) + x_tmp_index_n[j]
        for k in range(np.size(x_tmp2)):
            X_tran[i][cnt] = x_tmp2[k]
            cnt += 1
    x_tmp1 = X_tran[i][11:66]
    x_tmp_index_nn = np.arange(d+1)
    for j in range(d):
        x_tmp2 = X_tran[i][j+1] * x_tmp1[x_tmp_index_n[j]:]
        x_tmp_index_nn[j+1] = np.size(x_tmp2) + x_tmp_index_nn[j]
        for k in range(np.size(x_tmp2)):
            X_tran[i][cnt] = x_tmp2[k]
            cnt += 1
    x_tmp1 = X_tran[i][66:286]
    x_tmp_index_nnn = np.arange(d+1)
    for j in range(d):
        x_tmp2 = X_tran[i][j+1] * x_tmp1[x_tmp_index_nn[j]:]
        x_tmp_index_nnn[j+1] = np.size(x_tmp2) + x_tmp_index_nnn[j]
        for k in range(np.size(x_tmp2)):
            X_tran[i][cnt] = x_tmp2[k]
            cnt += 1

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

E_out_avg = 0
for i in range(256):
    list1 = random.sample(range(train_N), train_N)
    D_train = np.zeros([120, 1001])
    D_y_train = np.zeros(120)
    D_val = np.zeros([80, 1001])
    D_y_val = np.zeros(80)
    for j in range(120):
        D_train[j] = X_tran[list1[j]]
        D_y_train[j] = y[list1[j]]
    for j in range(80):
        D_val[j] = X_tran[list1[120+j]]
        D_y_val[j] = y[list1[120+j]]
    
    E_val = 1
    w_b = 0
    for j in range(5):
        prob = problem(D_y_train, D_train)
        param = parameter(f'-s 0 -c {(C_list[j])} -e 0.000001 -q')
        model_ptr = liblinear.train(prob, param)
        model_ = toPyModel(model_ptr)
        [W_sam, b_sam] = model_.get_decfun()

        E_val_tmp = 0
        for k in range(80):
            if np.sign(np.dot(W_sam, D_val[k])) != D_y_val[k]:
                E_val_tmp += 1
        E_val_tmp = E_val_tmp / 80
        if(E_val_tmp < E_val):
            E_val = E_val_tmp
            w_b = W_sam
    E_out = 0
    for i in range(test_N):
        if np.sign(np.dot(w_b, X_test_tran[i])) != y_test[i]:
            E_out += 1
    E_out = E_out / test_N
    E_out_avg += E_out
E_out_avg = E_out_avg / 256
print(E_out_avg)