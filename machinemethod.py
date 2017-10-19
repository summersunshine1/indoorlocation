import numpy as np
import pandas as pd
from commonLib import *
from getPath import *
pardir = getparentdir()

label_pkl_path = pardir+'/data/label.pkl'
mall_wifi_dic_path = pardir+'/data/mallwifi_dic.txt'
mall_dir = pardir+'/data/mall/'

def get_bssid_dic(bssids):
    dic = {}
    for i in range(bssids):
        dic[bssid[i]] = i
    return dic

def get_mall_data(mall_path,mall_wifi_dic):
    mall_id = get_mallid_from_mallpath(path)
    data = pd.read_csv(mall_path)
    wifi_infos = data['wifi_infos']
    shop_ids = data['shop_id']
    mall_bssids = mall_wifi_dic[mall_id]
    allbssid_len = len(mall_bssids)
    bssid_dic = get_bssid_dic(mall_bssids)
    for wifi_info in wifi_infos:
        train_d = np.array([0]*allbssid_len)
        bssids,strengths = process_wifi_info(wifi_info)
        length = len(bssids)
        for i in range(length):
            index = bssid_dic[bssids[i]] 
            if train_d
        
        
        
        
    