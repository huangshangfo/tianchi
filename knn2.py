'''
Created on 2017-8-29

@author: its
'''

def calDistance(d1,d2):
    distance=0.00
    for i in range(len(d1)):
        distance+=(d1[i] - d2[i]) * (d1[i] - d2[i])
    return distance

def forChoose(v,arr):
    sum=0.0
    for d in arr:
        sum+=abs(v-d)/d
    return sum/len(arr)

def knn(x,y,x_,k):   
    dis_value={}
    for i in range(len(x)):
        currData=x[i]
        value=y[i]
        key=str(i)
        for m in range(len(value)):
            key += '#'+str(value[m])
        distance=calDistance(x_,currData)
        dis_value[key]=distance
    sorted_list=sorted(dis_value.items(),key=lambda item:item[1])
    pre_value=[0]*30
    for n in range(30):
        cols=[]
        for i in range(k):
            c=float(sorted_list[i][0].split('#')[n+1])
            cols.append(c)
        min=1000.0
        chosen=0.2
        for d in cols:
            fci=forChoose(d,cols)
            if(fci<min):
                min=fci
                chosen=d
        pre_value[n]=chosen
    return pre_value