'''
Created on 2017-9-7

@author: its
'''
import json
import tensorflow as tf
import os
import numpy as np

datapath="C:\\Users\\its\\Documents\\code\\stacking\\tensor\\"
filepath=["\\tensor_zao.csv","\\tensor_zhong.csv","\\tensor_wan.csv"]

def stack(outputnum):
    W=tf.Variable(tf.ones(outputnum))
    x1=tf.placeholder(tf.float32, [None])
    x2=tf.placeholder(tf.float32, [None])
    trueY=tf.placeholder(tf.float32, [None])
    bi= tf.Variable(tf.ones(outputnum))
    y=tf.add(x1*W+x2*(1-W),bi)
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y/trueY,1)))
    return (x1,x2,y,trueY,lossfun,W)

def run(X1,X2,Y):
    x1,x2,y,trueY,lossfun,W=stack(30)
    times=10000
    train_step=tf.train.AdamOptimizer(learning_rate=3e-4).minimize(lossfun)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(times):
            sess.run(train_step,feed_dict={x1:X1,x2:X2,trueY:Y})
        w=sess.run(W, feed_dict={x1:X1,x2:X2,trueY:Y})
        return w

def splitData(tensor,num):
    model1=tensor[:,0:num]
    model2=tensor[:,num:2*num]
    trueY=tensor[:,2*num:]
    return (model1,model2,trueY)

ids = os.listdir(datapath)

for id in ids:
    tensor=np.loadtxt(datapath+id+filepath[0],delimiter=',')
    model1,model2,trueY=splitData(tensor,30)
    days=model1.shape[0]
    for day in range(days):
        X1=model1[day,:]; X2=model2[day,:]; Y=trueY[day,:]
        W=run(X1,X2,Y)
        s=""
        for w in W:
            s+=str(w)+' '  
        print (id,s)