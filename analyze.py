import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import json

from commonLib import *

from getPath import *
pardir = getparentdir()
shop_info_path = pardir + '/data/ccf_first_round_shop_info.csv'
shop_behavior_path = pardir + '/data/ccf_first_round_user_shop_behavior.csv'
evaluate_path = pardir + '/data/evaluation_public.csv'

mall_wifi_dic_path = pardir+'/data/mallwifi_dic.txt'
malldir = pardir+'/data/mall/'

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000

def getshopinfo():
    data = pd.read_csv(shop_info_path)
    mall_shop_dic = {}
    shop_info_dic = {}
    shop_mall_dic = {}
    shop_ids = data['shop_id']
    mall_ids = data['mall_id']
    longitudes = data['longitude']
    latitudes = data['latitude']
    length = len(mall_ids)
    for i in range(length):
        shop_id = shop_ids[i]
        mall_id = mall_ids[i]
        if not mall_id in mall_shop_dic:
            mall_shop_dic[mall_id] = []
        mall_shop_dic[mall_id].append(shop_id)
        shop_info_dic[shop_id] = [longitudes[i],latitudes[i]]
        shop_mall_dic[shop_id] = mall_id
    return mall_shop_dic,shop_info_dic,shop_mall_dic
    
def get_user_shop_behavior():
    data = pd.read_csv(shop_behavior_path)
    user_shop_dic = {}
    user_info_dic = {}
    shop_ids = data['shop_id']
    user_ids = data['user_id']
    longitudes = data['longitude']
    latitudes = data['latitude']
    length = len(shop_ids)
    for i in range(length):
        shop_id = shop_ids[i]
        user_id = user_ids[i]
        user_shop_dic[user_id]=shop_id
        user_info_dic[user_id]=[longitudes[i],latitudes[i]]
    return user_shop_dic,user_info_dic
    
def print_mall_shop():
    mall_shop_dic,shop_info_dic,shop_mall_dic = getshopinfo()
    for k,v in mall_shop_dic.items():
        print("mall_id: "+k)
        print("shoplength: "+str(len(v)))
    
def get_user_shop_info():
    mall_shop_dic,shop_info_dic,shop_mall_dic = getshopinfo()
    data = pd.read_csv(shop_behavior_path)
    wifi_infos = data['wifi_infos']
    shop_ids = data['shop_id']
    # mall_ids = data['mall_id']
    length = len(shop_ids)
    mall_wifi_dic = {}
    f = lambda x : x.split('|')[0]
    for i in range(length):
        wifi_info = wifi_infos[i]
        arr = np.array(wifi_info.split(";"))
        bssids = np.fromiter((f(xi) for xi in arr), arr.dtype, count=len(arr))
        mall_id = shop_mall_dic[shop_ids[i]]
        if not mall_id in mall_wifi_dic:
            mall_wifi_dic[mall_id] = set()
        mall_wifi_dic[mall_id].update(list(bssids))

    write_dic(mall_wifi_dic,mall_wifi_dic_path)
    
# def get_mall_shop_count():
    
    
def getmall_wifi_dic_info():
    dic = read_dic(mall_wifi_dic_path)
    for k,v in dic.items():
        print("mall_id: "+k)
        print("length: "+str(len(v)))
        for k1,v1 in dic.items():
            if k1!=k:
                inter = v&v1
                if len(inter)!=0:
                    print(k+":"+k1)
                    print(inter)
                
 
def move_to_different_mallfiles():
    chunksizes = 100
    mall_shop_dic,shop_info_dic,shop_mall_dic = getshopinfo()
    data = pd.read_csv(shop_behavior_path, chunksize = chunksizes)
    for chunk in data:
        shop_ids = chunk['shop_id']
        length = len(shop_ids)
        for i in range(length):
            shop_id = shop_ids.iloc[i]
            mall_id = shop_mall_dic[shop_id]
            path = malldir+mall_id+".csv"
            df = pd.DataFrame(chunk.iloc[[i]])
            write_record(df,path)
            print(i)

def evaluate():
    mall_shop_dic,shop_info_dic,shop_mall_dic = getshopinfo()
    user_shop_dic,user_info_dic = get_user_shop_behavior()
    resdic = {}
    count = 0
    for k,v in user_shop_dic.items():
        user_id = k
        shop_id = v
        userinfo = user_info_dic[user_id]
        mall_id = shop_mall_dic[shop_id]
        shops = mall_shop_dic[mall_id]
        dis = 100
        finalshop = -1
        diss = []
        for shop in shops:
            shopinfo = shop_info_dic[shop]
            temp = haversine(userinfo[0], userinfo[1], shopinfo[0],shopinfo[1])
            diss.append(temp)
            if temp<dis:
                dis = temp
                finalshop = shop
        print(shops)
        print(finalshop+" "+ shop_id)
        plt.xticks(list(range(len(shops))),shops)
        plt.plot(diss)
        plt.show()
            
        if finalshop == shop_id:
            count+=1
    print(count*1.0/len(user_shop_dic))
    
def getevalueinfo():
    data = pd.read_csv(evaluate_path)
    row_mall_dic = {}
    row_ids = data['row_id']
    mall_ids = data['mall_id']
    longitudes = data['longitude']
    latitudes = data['latitude']
    
def detect_null_value():
    data = pd.read_csv(shop_behavior_path)
    print(data['wifi_infos'][data['wifi_infos'].isnull()])
    print(data['longitude'][data['longitude'].isnull()])
    print(data['latitude'][data['latitude'].isnull()])
    del data
    # data = pd.read_csv(shop_info_path)
    # print(data[data.isnull()])
    # del data

if __name__=="__main__":
    # mall_shop_dic,shop_info_dic = getshopinfo()
    # detect_null_value()
    # move_to_different_mallfiles()    
    # mall_wifi_dic = get_user_shop_info()
    # print(mall_wifi_dic)
    # getmall_wifi_dic_info()
    # print_mall_shop()
    move_to_different_mallfiles()
    
