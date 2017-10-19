import numpy as np
import pandas as pd
import json
import pickle
import os
import sklearn.preprocessing as pp
from scipy.sparse import csc_matrix

def write_dic(dic,path):
    with open(path,'wb') as f:
        # json.dump(dic, f)
        pickle.dump(dic, f)
    
def read_dic(path):
    with open(path,'rb') as f:
        dic = pickle.load(f)
    return dic
    
def test():
    dic = {1:2,3:4}
    write_dic(dic,'1.txt')
    dic1 = dict(read_dic('1.txt'))
    print(dic1==dic)
    
def write_record(df,path):
    if os.path.exists(path):
        df.to_csv(path,mode = 'a',encoding = 'utf-8',index = False,header = False)
    else:
        df.to_csv(path,mode = 'w',encoding='utf-8',index = False)
        
def compute_cos(a,b):
    a = csc_matrix(a)
    b = csc_matrix(b)
    vec_a = pp.normalize(a, axis=1)
    vec_b = pp.normalize(b, axis=1)
    res = vec_a*vec_b.T
    res = res.todense()
    return res.item(0)
    
def listfiles(rootDir): 
    list_dirs = os.walk(rootDir) 
    filepath_list = []
    for root, dirs, files in list_dirs:
        for f in files:
            filepath_list.append(os.path.join(root,f))
    return filepath_list

if __name__=="__main__":
    a = [1,1,3,4,5]
    b = [0,1,2,4]
    print(compute_cos(a,b))
    
    

