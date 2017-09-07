'''
Created on 2017-7-31

@author: Administrator
'''
import numpy as np
import os
import datetime
import json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
TablePath="C:\\Users\\its\\Documents\\icdm\\tables\\"


shareResPath="C:\\Users\\its\\Documents\\icdm\\tianchiResult\\"

toBeValided=TablePath+"sv_fcn_meanzao.txt"

# TrueY=datapath+"sv811_TrueY.txt"
TrueYFill=TablePath+"sv_TrueYFill_zao.txt"

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
        link=None
        sumi=0.0
        for line in lines:
            ls=line.split("#")
            if(len(ls)==4):
                prey=float(ls[3])
                truey=YDict[(ls[0],ls[1],ls[2])]
                if(link is None or link==ls[0]):
                    sumi+=abs(prey-truey)/truey
                else:
                    print(sumi/900.0)
                    sumi=abs(prey-truey)/truey
                link=ls[0]
        print(sumi/900.0)
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
    return sumloss/118800.0

# print(mape(toBeValided,TrueY))
print(mape(toBeValided,TrueYFill))

