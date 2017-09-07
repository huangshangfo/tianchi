'''
Created on 2017-9-6

@author: Administrator
'''
import json
file1="C:\\Users\\its\\Documents\\icdm\\tables\\fcn_zao906_mean.txt"
file2="C:\\Users\\its\\Documents\\icdm\\tables\\rnnv4_zao.txt"
file3="C:\\Users\\its\\Documents\\icdm\\tables\\int_zao.txt"

fileW="C:\\Users\\its\\Documents\\code\\weight_zao.txt"

with open(fileW) as f:
    Ws=json.loads(f.read())

result_dict_1={}
result_dict_2={}
# result_dict_3={}
with open(file1) as f1:
    all = f1.readlines()
    for i in range(len(all)):
        values = all[i].replace("\n","").split("#")
        result_dict_1[(values[0],values[1],values[2])] = float(values[3])

with open(file2) as f2:
    all = f2.readlines()
    for i in range(len(all)):
        values = all[i].replace("\n","").split("#")
        result_dict_2[(values[0],values[1],values[2])] = float(values[3])
# with open(path+file3) as f3:
#     all = f3.readlines()
#     for i in range(len(all)):
#         values = all[i].replace("\n","").split("#")
#         result_dict_3[(values[0],values[1],values[2])] = float(values[3])
with open(file3,"w") as f4:
    for i in result_dict_1:
        link = i[0]
        W=Ws[link]
        f4.write("#".join(i)+"#"+str(W*result_dict_1[i]+(1-W)*result_dict_2[i])+"\n")
