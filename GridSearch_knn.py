'''
Created on 2017-7-26

@author: Administrator
'''
import numpy as np
import os
import datetime
import json
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

def loadPath():
    #with open("config.json") as f:
    with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]

datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()

toppath=datapath
linkDict={}

toBeValided=datapath+"self_valid_knn_803.txt"

TrueY=datapath+"selfValid_TrueY.txt"

TrueYFill=datapath+"selfValid_TrueYFill.txt"

def mape(toBeValided,TrueY):
    YDict={}
    with open(TrueY) as f:
        f_all=f.read()
        lines=f_all.split("\n")
        for line in lines:
            ls=line.split("#")
            if(len(ls)==4):
                YDict[(ls[0],ls[1],ls[2])]=float(ls[3])
    
    with open(toBeValided) as f:
        sumloss=0
        notzero=0
        f_all=f.read()
        lines=f_all.split("\n")
        for line in lines:
            ls=line.split("#")
            if(len(ls)==4):
                prey=float(ls[3])
                truey=YDict[(ls[0],ls[1],ls[2])]
                if truey==0:
                    continue
                else:
                    notzero+=1
                    sumloss+=abs(prey-truey)/truey
    return sumloss/notzero

with open(toppath+"linkDict.json") as f:
    linkDict=json.loads(f.read())

def splitData(tensor,tensorFill,n_output,n_pred):
    #print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensorFill[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

def weight(index,col):
    return (index+1)**0.9*(1/col)

weights=[]

for i in range(1,61):
    weights.append(i/60)
    
valid_weights=weights[:]

path=rootpath

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


ids = os.listdir(path)

min=9999
minIndex=0

for w in range(60):
    with open(toppath+"self_valid_knn.txt", "w") as f1:
        for i in ids:
            taskpath=path+i+"/"
            tensorFill=np.loadtxt(taskpath+"tensor_fill.csv",delimiter=',')
            tensor=np.loadtxt(taskpath+"tensor.csv",delimiter=',')
            #print(tensor)
            knownX,knownY,preX=splitData(tensor,tensorFill,30,days)
            kf = KFold(n_splits=12)
            avgY=None
            first=True       
            for train_index, valid_index in kf.split(knownX):
                trainX, validX = knownX[train_index], knownX[valid_index]
                trainY, validY = knownY[train_index], knownY[valid_index]
                neigh = KNeighborsRegressor(n_neighbors=7)
                
                row=trainX.shape[0]
                col=trainX.shape[1]
                for r in range(row):
                    for c in range(col):
                        trainX[r][c]*=weight(c,col)
                
                neigh.fit(trainX, trainY)
                
                row=preX.shape[0]
                col=preX.shape[1]
                for r in range(row):
                    for c in range(col):
                        preX[r][c]*=weight(c,col)
                preY=neigh.predict(preX).reshape(-1,1)
                if(first):
                    avgY=preY
                    first=False
                else:
                    avgY+=preY
                #print(preY)
            avgY=avgY/12
            taskname=linkDict[i]
            for j in range(len(avgY)):
                f1.write(taskname+"#"+timeId[j].split(" ")[0][1:]+"#"+timeId[j]+"#"+str(avgY[j][0])+"\n")
    
    MAPE=mape(toBeValided,TrueY)
    if(MAPE<min):
        min=MAPE
        minIndex=w
    print(MAPE)

print(min,minIndex)
