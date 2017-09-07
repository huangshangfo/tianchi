'''
Created on 2017-9-4

@author: its
'''

import numpy as np
import os
import datetime
import json
import knn2

def loadPath():
    #with open("configSelfValid.json") as f:
    with open("config_wan.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["outputpath"],config["filepath"],config["startdate"],config["days"],config["k"]

datapath,outputpath,filepath,startdate,days,k=loadPath()

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

def weight(index,col):
    return (index+1)**0.9*(1/col)

timeId=[]
timeDay=[]
timeMin=[]
startDay = datetime.datetime.strptime(startdate,"%Y-%m-%d")
startTime = datetime.datetime.strptime("08:00:00","%H:%M:%S")
for i in range(0,days,1):
    endDay = startDay + datetime.timedelta(days=i)
    timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
for i in range(0,62,2):
    endTime = startTime + datetime.timedelta(minutes=i)
    timeMin.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
for i in range(len(timeDay)):
    for j in range(len(timeMin)-1):
        timeId.append("["+timeDay[i]+" "+timeMin[j]+","+timeDay[i]+" "+timeMin[j+1]+")")


ids = os.listdir(datapath)
'''
min_mape=100
min_k=3

for k in range(3,216):
       
        
print("min",min_k,min_mape) 
'''
mape_sum=0.0 
for id in ids:
    #print(id)
    taskpath=datapath+id+"/"
    tensorFill=np.loadtxt(taskpath+filepath,delimiter=',')
    trainX,trainY,preX=splitData(tensorFill,30,days)
    
    preY=[]
    trueY=tensorFill[-61:-31,60:]
    #print(trueY.shape[0],trueY.shape[1])
    
    '''
    row=trainX.shape[0]
    col=trainX.shape[1]
    for r in range(row):
        for c in range(col):
            trainX[r][c]*=weight(c,col)
            
    row=preX.shape[0]
    col=preX.shape[1]
    for r in range(row):
        for c in range(col):
            preX[r][c]*=weight(c,col)
    '''
            
    for i in range(len(preX)):
        y_=knn2.knn(trainX,trainY,preX[i],k)
        preY.append(y_)
    
    tmp=MAPE(preY,trueY)
    print(id,tmp)
    mape_sum+=tmp
    
print("all",mape_sum/float(len(ids)))
    