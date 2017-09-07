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
    with open("config_zhong.json") as f:
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

timeId=[]
timeDay=[]
timeMin=[]
startDay = datetime.datetime.strptime(startdate,"%Y-%m-%d")
startTime = datetime.datetime.strptime("15:00:00","%H:%M:%S")
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

with open(outputpath+"knn904_zhong.txt", "w") as f1:
    for id in ids:
        print(id)
        taskpath=datapath+id+"/"
        tensorFill=np.loadtxt(taskpath+filepath,delimiter=',')
        trainX,trainY,preX=splitData(tensorFill,30,days)
                
        for i in range(len(preX)):
            preY=knn2.knn(trainX,trainY,preX[i],k)
            for j in range(len(preY)):
                f1.write(id+"#"+timeId[30*i+j].split(" ")[0][1:]+"#"+timeId[30*i+j]+"#"+str(preY[j])+"\n")
