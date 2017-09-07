'''
Created on 2017-8-31

@author: its
'''
import os

path="C:\\Users\\its\\Desktop\\kv.txt"

with open(path) as f:
    while True:
        line=f.readline().replace('\n','')
        if(not line):
            break
        print(line.split(' ')[1])