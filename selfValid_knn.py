'''
Created on 2017-7-26

@author: Administrator
'''
import numpy as np
import os
import datetime
import json
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

weights=[0.0960366462991885, 0.04479792299203811, 0.0960366462991885, 0.05803670421974161, 0.07094499354339873, 0.35584255089579936, 0.1442454644027416, 0.0960366462991885, 0.1323880391207136, 0.1204112342640346, 0.15599542838086067, 0.016666666666666666, 0.1204112342640346, 0.16764776449657207, 0.6640256795567926, 0.21342868022876105, 0.21342868022876105, 0.05803670421974161, 0.016666666666666666, 0.03110109971789358, 0.1204112342640346, 0.3128417904653896, 0.34514926089602166, 0.3128417904653896, 0.6340684144300637, 0.23589948375610928, 0.37712361663282534, 0.23589948375610928, 0.37712361663282534, 0.4816960787751129, 0.37712361663282534, 0.3877138323758875, 0.3877138323758875, 0.41929627126294755, 0.40879915509753567, 0.41929627126294755, 0.35584255089579936, 0.40879915509753567, 0.4506162986222866, 0.6039529869125293, 0.4816960787751129, 0.3665002490258257, 0.49200595539687497, 0.533011797623584, 0.5125544542448763, 0.22469530824002307, 0.6240479687562565, 0.5533822459437262, 0.5533822459437262, 0.5432076321943138, 0.5837834232713071, 0.6039529869125293, 0.5938777196270132, 0.6039529869125293, 0.6140096124430231, 0.40879915509753567, 0.6640256795567926, 0.6440712948410663, 0.6540569428697867, 0.6640256795567926]

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
                    #trainX[r][c]*=weight(c,col)
                    trainX[r][c]*=weights[c]
            
            neigh.fit(trainX, trainY)
            
            row=preX.shape[0]
            col=preX.shape[1]
            for r in range(row):
                for c in range(col):
                    #preX[r][c]*=weight(c,col)
                    preX[r][c]*=weights[c]
            preY=neigh.predict(preX).reshape(-1,1)
            if(first):
                avgY=preY
                first=False
            else:
                avgY+=preY
            print(preY)
        avgY=avgY/12
        taskname=linkDict[i]
        for j in range(len(avgY)):
            f1.write(taskname+"#"+timeId[j].split(" ")[0][1:]+"#"+timeId[j]+"#"+str(avgY[j][0])+"\n")    