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

toBeValided=datapath+"self_valid_knn.txt"

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

'''
def weight(index,col):
    return (index+1)**0.9*(1/col)
'''
weights=[]

for i in range(1,61):
    weights.append(i**0.9*(1/60))
    
valid_weights=weights[:]

#we=[0.0960366462991885, 0.04479792299203811, 0.0960366462991885, 0.05803670421974161, 0.07094499354339873, 0.323650391996189, 0.0960366462991885, 0.10830031951416474, 0.1204112342640346, 0.1323880391207136, 0.1204112342640346, 0.016666666666666666, 0.1442454644027416, 0.1442454644027416, 0.6640256795567926, 0.21342868022876108, 0.22469530824002307, 0.0960366462991885, 0.1442454644027416, 0.16764776449657207, 0.1323880391207136, 0.23589948375610928, 0.3128417904653896, 0.24704481636898257, 0.5938777196270132, 0.20209554220138642, 0.42976426530534817, 0.03110109971789358, 0.30199152653237316, 0.1204112342640346, 0.5022918811548709, 0.3982719999871872, 0.3771236166328254, 0.05803670421974161, 0.6640256795567926, 0.29109776241652047, 0.41929627126294755, 0.4402040020619023, 0.40879915509753567, 0.5736696948708219, 0.2691715543346121, 0.23589948375610928, 0.5125544542448763, 0.5022918811548709, 0.5022918811548709, 0.4816960787751129, 0.5837834232713071, 0.5227942449709373, 0.5432076321943138, 0.5432076321943138, 0.5635361148385345, 0.5938777196270132, 0.5938777196270132, 0.6039529869125293, 0.6039529869125293, 0.6240479687562565, 0.6340684144300638, 0.6440712948410663, 0.6540569428697867, 0.6640256795567926]

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

for cindex in range(60):
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
                            if(c==cindex):
                                trainX[r][c]*=weights[w]
                            else:
                                trainX[r][c]*=valid_weights[c]
                    
                    neigh.fit(trainX, trainY)
                    
                    row=preX.shape[0]
                    col=preX.shape[1]
                    for r in range(row):
                        for c in range(col):
                            if(c==cindex):
                                preX[r][c]*=weights[w]
                            else:
                                preX[r][c]*=valid_weights[c]
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

    print(minIndex,min)
    valid_weights[cindex]=weights[minIndex]
    
print(valid_weights)
