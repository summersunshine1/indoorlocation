import numpy as np
import pandas as pd
import json
import pickle
import os
import sklearn.preprocessing as pp
from scipy.sparse import csc_matrix
from sklearn import preprocessing
from sklearn.externals import joblib


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
    
def compare_res(res1,res2):
    d1 = pd.read_csv(res1)
    rowids = d1['row_id']
    shopids = d1['shop_id']
    d2 =pd.read_csv(res2)
    rowids1 = d2['row_id']
    shopids1 = d2['shop_id']
    length = len(rowids)
    count=0
    dic = {}
    for i in range(length):
        if not rowids[i] in dic:
            dic[rowids[i]] = []
        dic[rowids[i]].append(shopids[i])
        if not rowids1[i] in dic:
            dic[rowids1[i]] = []
        dic[rowids1[i]].append(shopids1[i])
    for k,v in dic.items():
        if v[0]==v[1]:
            cpunt+=1
        
    print(count/length)
    
def convertLabels(arr,pickle_path):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(arr)
    joblib.dump(le,pickle_path)
    return labels
    
def getlabels_detail(labels,pickle_path):
    le = joblib.load(pickle_path)
    arr = le.inverse_transform(labels)
    return arr   
    
def process_wifi_info(wifi_info):
    f1 = lambda x : x.split('|')[0]
    f2 = lambda x : int(x.split('|')[1])
    arr = np.array(wifi_info.split(";"))
    bssids = list(np.fromiter((f1(xi) for xi in arr), arr.dtype, count=len(arr)))
    strengths= list(np.fromiter((f2(xi) for xi in arr), arr.dtype, count=len(arr)))
    return bssids,strengths
    
def get_mallid_from_mallpath(path):
    mallpath = os.path.basename(path)
    mall_id = mallpath[:-4]
    return mall_id
    

if __name__=="__main__":
    # a = [1,1,3,4,5]
    # b = [0,1,2,4]
    # print(compute_cos(a,b))
    arr = ['a','b','c','c']
    labels = convertLabels(arr,"1")
    print(getlabels_detail(labels,'1'))
    
    

