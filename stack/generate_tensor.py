'''
Created on 2017-9-7

@author: its
'''
import os

toppath="C:\\Users\\its\\Documents\\code\\stacking\\"
outpath="C:\\Users\\its\\Documents\\code\\stacking\\tensor\\"
filepath=["\\tensor_zao.csv","\\tensor_zhong.csv","\\tensor_wan.csv"]

model1=["sv_fcn_zao.txt","sv_fcn_zhong.txt","sv_fcn_wan.txt"]
model2=["sv_rnnv4_zao.txt","sv_rnnv4_zhong.txt","sv_rnnv4_wan.txt"]
trueY=["sv_TrueYFill_zao.txt","sv_TrueYFill_zhong.txt","sv_TrueYFill_wan.txt"]

days=30
times=30
for i in range(0,3):
    f1=open(toppath+model1[i])
    f2=open(toppath+model2[i])
    f3=open(toppath+trueY[i])
    lines1=f1.readlines(); lines2=f2.readlines(); lines3=f3.readlines()
    index=0
    for j in range(132):
        link=lines1[index].split('#')[0]
        tensor=[]
        for m in range(days):
            value=[]
            for n in range(times):
                value.append(lines1[index+n].split('#')[-1].strip('\n'))
            for n in range(times):
                value.append(lines2[index+n].split('#')[-1].strip('\n'))
            for n in range(times):
                value.append(lines3[index+n].split('#')[-1].strip('\n')) 
            index+=times
            tensor.append(value)
            if not os.path.exists(outpath+link):
                os.mkdir(outpath+link)
        
        with open(outpath+link+filepath[i],'w') as f:
            for k in range(days):
                f.write(",".join(tensor[k]) + "\n")
        
        
