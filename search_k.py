# coding=UTF-8

import os
import numpy as np
import json
import knn2

def splitData(tensor,n_output,n_pred):
    #print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

def MAPE(preY,trueY):
    not_zero=0
    sum_loss=0.0
    for i in range(len(preY)):
        for j in range(len(preY[i])):
            if(trueY[i][j]!=0):
                not_zero+=1
                sum_loss+=abs(preY[i][j]-trueY[i][j])/trueY[i][j]
    return sum_loss/not_zero

rootdir="C:\\Users\\its\\Documents\\icdm\\0.2million_tensor\\zao\\tensor"
link_k={}
all_k=[]
link_dirs=os.listdir(rootdir)
for i in range(len(link_dirs)):
    link_path=os.path.join(rootdir,link_dirs[i])
    if(os.path.isdir(link_path)):
        link=os.path.basename(link_path)
        filepath=link_path+"\\tensor_fill.csv"
        #print(filepath)
        tensorFill=np.loadtxt(filepath,delimiter=',')
        trainX,trainY,preX=splitData(tensorFill,30,31)
        preY=[]
        trueY=tensorFill[92:,60:]
        '''
        for x_ in preX:
            y_=knn2.knn(trainX,trainY,x_,11)
            preY.append(y_)
        mape=MAPE(preY,trueY)
        print(mape)
        '''
        mape=100000
        optimal_k=3
        for k in range(3,93):
            preY=[]
            trueY=tensorFill[92:,60:]
            for x_ in preX:
                y_=knn2.knn(trainX,trainY,x_,k)
                preY.append(y_)
            tmp=MAPE(preY,trueY)
            if(mape>tmp):
                mape=tmp
                optimal_k=k
        print(mape,optimal_k)
        link_k[link]=optimal_k
        all_k.append(optimal_k)

print(link_k)
print(all_k)
                