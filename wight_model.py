import numpy as np
import pandas as pd
import os
from collections import Counter
from analyze import *
from commonLib import *
from getPath import *
pardir = getparentdir()

mall_dir = pardir+'/data/mall/'
evaluate_public_path = pardir+'/data/evaluation_public.csv'
evaluate_a_path = pardir+'/data/evaluation_a.csv'
evaluate_b_path = pardir+'/data/evaluation_b.csv'
mall_dic_dir = pardir+'/data/malldic/'
respath = pardir+'/data/res.csv'

weightarr = [100,80,60,40,20,0]

def split_a_b():
    data = pd.read_csv(evaluate_public_path)
    dates = data['time_stamp']
    dates = pd.to_datetime(dates)
    length = len(dates)
    a_index= []
    b_index = []
    for i in range(length):
        if dates[i].day<=7:
            a_index.append(i)
        else:
            b_index.append(i)
    data.iloc[a_index].to_csv(evaluate_a_path,mode = 'w',encoding='utf-8',index = False)
    data.iloc[b_index].to_csv(evaluate_b_path,mode = 'w',encoding='utf-8',index = False)
    

def get_evaluate_data():
    data = pd.read_csv(evaluate_a_path)
    mall_ids = data['mall_id']
    wifi_infos = data['wifi_infos']
    row_ids = data['row_id']
   
def process_wifi_info(wifi_info):
    f1 = lambda x : x.split('|')[0]
    f2 = lambda x : int(x.split('|')[1])
    arr = np.array(wifi_info.split(";"))
    bssids = np.fromiter((f1(xi) for xi in arr), arr.dtype, count=len(arr))
    strengths= np.fromiter((f2(xi) for xi in arr), arr.dtype, count=len(arr))
    return bssids,strengths
    
def get_strength_range(v1):
    down = int(np.min(v1))-5
    up = int(np.max(v1))+5
    v1.append(down)
    v1.append(up)
    bins = pd.cut(v1,bins = int((up-down)/5),labels = False, retbins=True)
    labels = bins[0]
    ranges = bins[1]
    # print(v1)
    # print(labels)
    # print(ranges)
    c = dict(Counter(labels))
    # dict = sorted(c.items(),key=lambda d:d[0])
    # a = [d[1] for d in dict]
    # b = [d[0] for d in dict]
    binnum = len(bins)
    sum = len(v1)
    weightdic = {}
    for i in range(len(ranges)-1):
        r = str(ranges[i])+'|'+str(ranges[i+1])
        if not i in c.keys():
            weightdic[r] = 1/(binnum+sum)
        else:
            weightdic[r] = (1+c[i])/(binnum+sum)
    # print(weightdic)
    return weightdic,[down,up]

def process():
    filelist = listfiles(mall_dir)
    for file in filelist:
        mall_id = os.path.basename(file)[:-4]
        data = pd.read_csv(file)
        shop_ids = data['shop_id']
        wifi_infos = data['wifi_infos']
        length = len(shop_ids)
        shop_wifi_dic = {}
        shop_wifi_range_dic ={}
        for i in range(length):
            shop_id = shop_ids[i]
            wifi_info = wifi_infos[i]
            bssids,strengths = process_wifi_info(wifi_info)
            if not shop_id in shop_wifi_dic:
                shop_wifi_dic[shop_id] = {}
            for j in range(len(bssids)):
                if not bssids[j] in shop_wifi_dic[shop_id]:
                    shop_wifi_dic[shop_id][bssids[j]] = []
                shop_wifi_dic[shop_id][bssids[j]].append(strengths[j])
        # print(shop_wifi_dic.keys())
        for k,v in shop_wifi_dic.items():
            if not k in shop_wifi_range_dic:
                shop_wifi_range_dic[k] = {}
            for k1,v1 in v.items():
                v1 = [int(t) for t in v1]
                weightdic,arr = get_strength_range(v1)
                shop_wifi_range_dic[k][k1] = {}
                shop_wifi_range_dic[k][k1]['weight'] = weightdic
                shop_wifi_range_dic[k][k1]['range'] = arr
        
        write_dic(shop_wifi_range_dic,mall_dic_dir+mall_id)
        
def get_sides_from_dic(weight_dic):
    min = 0
    max = -1000
    minv = 0
    maxv = 0
    i = 0
    for k,v in weight_dic.items():
        arr = k.split('|')
        down = float(arr[0])
        up = float(arr[1])
        if down<min:
            min=down
            minv = float(v)
        if up>max:
            max = up
            maxv = float(v)
    return min,max,minv,maxv
        
def getweight(strength,weight_dic):
    # print(strength)
    # print(weight_dic)
    for k,v in weight_dic.items():
        arr = k.split('|')
        down = float(arr[0])
        up = float(arr[1])
        # downs.append(down)
        # ups.append(up)
        # print(down)
        # print(up)
        if strength>down and strength<=up:
            return float(v)
    min,max,minv,maxv = get_sides_from_dic(weight_dic)
    if np.abs(min-strength)<1:
        return minv
    if np.abs(max-strength)<1:
        return maxv

def create_model():
    data = pd.read_csv(evaluate_a_path)
    row_ids = data['row_id']
    mall_ids = data['mall_id']
    wifi_infos = data['wifi_infos']
    
    length = len(row_ids)
    resdic = {}
    for i in range(length):
        countdic = {}
        wifi_info = wifi_infos[i]
        bssids,strengths = process_wifi_info(wifi_info)
        mall_id = mall_ids[i]
        dic = read_dic(mall_dic_dir+mall_id)
        # print(dic.keys())
        for j in range(len(bssids)):
            bssid = bssids[j]
            strength = int(strengths[j])
            for shop_id,bssid_info in dic.items():
                if bssid in bssid_info.keys() and (strength<=int(bssid_info[bssid]['range'][1]) and strength>=int(bssid_info[bssid]['range'][0])):
                    if not shop_id in countdic:
                        countdic[shop_id]=0
                    weight_dic = bssid_info[bssid]['weight']
                    countdic[shop_id]+=getweight(strength,weight_dic)*weightarr[int(np.floor(strength/20))]
        dict = sorted(countdic.items(),key=lambda d:d[1])
        a = [d[1] for d in dict]
        b = [d[0] for d in dict]
        # print(a)
        # print(b)
        row_id = row_ids[i]
        if len(b)==0:
            resdic[row_id] = shop_id
            print("zero")
        else:
            resdic[row_id] = b[len(b)-1]
        
    return resdic
        # print(arr)
        # if len(arr)>0:
            # print(sorted(arr))
            # l = len(arr[np.where(arr == np.max(arr))])
            # if l>1:
                # plt.plot(list(countdic.values()))
                # plt.show()
        # else:
            # print("zero")
def getres(files):
    for file in files:
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        mall_id = get_mallid_from_mallpath(file)
        shop_ids = data['shop_id']
        countdic = {}
        c = 0
        for i in range(len(wifi_infos)):
            wifi_info = wifi_infos[i]
            bssids,strengths = process_wifi_info(wifi_info)
            dic = read_dic(mall_dic_dir+mall_id)
            # print(dic.keys())
            for j in range(len(bssids)):
                bssid = bssids[j]
                strength = int(strengths[j])
                for shop_id,bssid_info in dic.items():
                    if bssid in bssid_info.keys() and (strength<=int(bssid_info[bssid]['range'][1]) and strength>=int(bssid_info[bssid]['range'][0])):
                        if not shop_id in countdic:
                            countdic[shop_id]=0
                        weight_dic = bssid_info[bssid]['weight']
                        countdic[shop_id]+=getweight(strength,weight_dic)#*weightarr[int(np.floor(strength/20))]
            dict = sorted(countdic.items(),key=lambda d:d[1])
            a = [d[1] for d in dict]
            b = [d[0] for d in dict]
            if b[-1] == shop_ids[i]:
                c+=1
                
        print(c/len(wifi_infos))
                
    

            
def write_res_to_file(dic):
    with open(respath,mode = 'w',encoding='utf-8') as f:
        f.writelines("row_id,shop_id\n")
        for k,v in dic.items():
            lines = str(k)+','+str(v)+'\n'
            f.writelines(lines)
          
def append_res_file():
    data = pd.read_csv(evaluate_b_path)
    row_ids = data['row_id']
    with open(respath,mode = 'a',encoding='utf-8') as f:
        for row_id in row_ids:
            lines = str(row_id)+',\n'
            f.writelines(lines)

if __name__=="__main__":
    # split_a_b()
    # print(pardir)
    # dic = read_dic(pardir+'/data/mallwifi_dic.txt')
    # print(dic)
    # move_to_different_mallfiles()
    # process()
    # resdic = create_model()
    # write_res_to_file(resdic)
    # get_strength_range([1,5,7,8,15])
    # append_res_file()
    # getweight(-80,{'-90.1|-80.2':0.1})
    getres([mall_dir+'m_6803.csv'])   
    

