import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wight_model import *
from commonLib import *
import threading

malldir = pardir+'/data/mall/'
mallshop_dic_dir = pardir+'/data/mall_shop_dic/'
mallshop_train_dic_dir = pardir+'/data/mall_shop_dic_train/'
mall_wifi_dic_dir = pardir+'/data/mall_wifi_dic/'
mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic'
middle_path = pardir+'/data/middle'

mallshop_dic_add_conect_dir = pardir+'/data/mall_shop_add_connect/'


def get_mall_shop_info_train(paths):
    for path in paths:
        data = pd.read_csv(path)
        mall_id = os.path.basename(path)[:-4]
        shop_ids = data['shop_id']
        wifi_infos = data['wifi_infos']
        length = len(shop_ids)
        shop_wifi_dic = {}
        train_index,valid_index = get_fix_date(data['time_stamp'])
        for i in train_index:
            shop_id = shop_ids[i]
            wifi_info = wifi_infos[i]
            bssids,strengths,connects = process_wifi_info(wifi_info)
            if not shop_id in shop_wifi_dic:
                shop_wifi_dic[shop_id] = {}
            for j in range(len(bssids)):
                if not bssids[j] in shop_wifi_dic[shop_id]:
                    shop_wifi_dic[shop_id][bssids[j]] = []
                shop_wifi_dic[shop_id][bssids[j]].append(strengths[j])
        write_dic(shop_wifi_dic,mallshop_train_dic_dir+mall_id)    
        
def getmall_shop_add_connect(paths):
    for path in paths:
        data = pd.read_csv(path)
        mall_id = os.path.basename(path)[:-4]
        shop_ids = data['shop_id']
        wifi_infos = data['wifi_infos']
        length = len(shop_ids)
        shop_wifi_dic = {}
        train_index,valid_index = get_fix_date(data['time_stamp'])
        for i in train_index:
            shop_id = shop_ids[i]
            wifi_info = wifi_infos[i]
            bssids,strengths,connects = process_wifi_info(wifi_info)
            if not shop_id in shop_wifi_dic:
                shop_wifi_dic[shop_id] = {}
            for j in range(len(bssids)):
                if not bssids[j] in shop_wifi_dic[shop_id]:
                    shop_wifi_dic[shop_id][bssids[j]]={}
                if not 'strength' in shop_wifi_dic[shop_id][bssids[j]]:
                    shop_wifi_dic[shop_id][bssids[j]]['strength']=[]
                    shop_wifi_dic[shop_id][bssids[j]]['connect']=[]
                shop_wifi_dic[shop_id][bssids[j]]['strength'].append(strengths[j])
                shop_wifi_dic[shop_id][bssids[j]]['connect'].append(connects[j])
        write_dic(shop_wifi_dic,mallshop_dic_add_conect_dir+mall_id)   

def get_mall_shop_info(paths):
    for path in paths:
        data = pd.read_csv(path)
        mall_id = os.path.basename(path)[:-4]
        shop_ids = data['shop_id']
        wifi_infos = data['wifi_infos']
        length = len(shop_ids)
        shop_wifi_dic = {}
        for i in range(length):
            shop_id = shop_ids[i]
            wifi_info = wifi_infos[i]
            bssids,strengths,connects = process_wifi_info(wifi_info)
            if not shop_id in shop_wifi_dic:
                shop_wifi_dic[shop_id] = {}
            for j in range(len(bssids)):
                if not bssids[j] in shop_wifi_dic[shop_id]:
                    shop_wifi_dic[shop_id][bssids[j]] = []
                shop_wifi_dic[shop_id][bssids[j]].append(strengths[j])
        write_dic(shop_wifi_dic,mallshop_dic_dir+mall_id)
        
def get_mall_wifi_dic(files): 
    for file in files:
        mall_id = os.path.basename(file)
        dic = read_dic(file)
        path = mall_wifi_dic_dir+mall_id
        bssid_dic = {}
        for shop_id,bssiddic in dic.items():
            for bssid,strengths in bssiddic.items():
                if not bssid in bssid_dic:
                    bssid_dic[bssid]=[]
                bssid_dic[bssid]+=list(strengths)
            write_dic(bssid_dic,path)
            
def get_dic_sum(dic): 
    sum = 0
    for k,v in dic.items():
        sum+=len(v)
    return sum
    
def get_important_shop_bssid(filelist):
    bssiddics = {}
    missing = []
    co = 0
    for file in filelist:
        mall_id = os.path.basename(file) 
        mall_shop_dic = read_dic(file)
        # mall_wifi_dic = read_dic(path)
        # print(mall_wifi_dic)
        # totalsum = get_dic_sum(mall_wifi_dic)
        mall_bssid_set = set()
        for shop_id,bssiddic in mall_shop_dic.items():
            shop_impor_dic = {}
            shopbssidtotallen = get_dic_sum(mall_shop_dic[shop_id])
            for bssid,strengths in bssiddic.items():
                # bssidlen = len(mall_wifi_dic[bssid])
                str = [int(t) for t in strengths]
                # if len(str)>2:
                    # str.remove(np.max(str))
                    # str.remove(np.min(str))
                # shop_impor_dic[bssid]=(len(strengths))/((np.median(str))/(np.max(str)))
                # shop_impor_dic[bssid]=len(strengths)/np.square(np.max(str))
                shop_impor_dic[bssid]=len(strengths)/-np.median(str)
            dict = sorted(shop_impor_dic.items(),key=lambda d:d[1])
            a = [d[1] for d in dict]
            b = [d[0] for d in dict]
            for t in b[-15:]:
                mall_bssid_set.add(t)
        bssiddics[mall_id] = list(mall_bssid_set)
        print(mall_id)
        print(len(mall_bssid_set))
        # co+=1
        # if co>3:
            # break
    write_dic(bssiddics,mall_bssid_dic_path)
    
def get_important_shop_bssid_connect(filelist):
    bssiddics = {}
    missing = []
    co = 0
    for file in filelist:
        mall_id = os.path.basename(file) 
        mall_shop_dic = read_dic(file)
        # mall_wifi_dic = read_dic(path)
        # print(mall_wifi_dic)
        # totalsum = get_dic_sum(mall_wifi_dic)
        mall_bssid_set = set()
        tempset = set()
        for shop_id,bssiddic in mall_shop_dic.items():
            shop_impor_dic = {}
            shopbssidtotallen = get_dic_sum(mall_shop_dic[shop_id])
            for bssid,dic in bssiddic.items():
                # bssidlen = len(mall_wifi_dic[bssid])
                strengths = dic['strength']
                connects = dic['connect']
                connects = np.array(connects)
                # print(connects)
                # print("end")
                # break
                length = len(connects[connects=="true"])
                tlength = len(connects)
                if length/tlength>0.1 and tlength>3:
                    mall_bssid_set.add(bssid)
                str = [int(t) for t in strengths]
                # if len(str)>2:
                    # str.remove(np.max(str))
                    # str.remove(np.min(str))
                # shop_impor_dic[bssid]=(len(strengths))/((np.median(str))/(np.max(str)))
                # shop_impor_dic[bssid]=len(strengths)/np.square(np.max(str))
                shop_impor_dic[bssid]=len(strengths)/-np.max(str)
            dict = sorted(shop_impor_dic.items(),key=lambda d:d[1])
            a = [d[1] for d in dict]
            b = [d[0] for d in dict]
            for t in b[-15:]:
                mall_bssid_set.add(t)
                # tempset.add(t)
        bssiddics[mall_id] = list(mall_bssid_set)
        print(mall_id)
        print(len(mall_bssid_set))
        # print(len(tempset))
        # co+=1
        # if co>3:
            # break
    write_dic(bssiddics,mall_bssid_dic_path)
            
def get_mall_wifi_dic_threading():
    filelist = listfiles(mallshop_dic_dir)
    newfiles = []
    for file in filelist:
        mall_id = os.path.basename(file) 
        path = mall_wifi_dic_dir+mall_id
        if os.path.exists(path):
            continue
        newfiles.append(file)
    length = len(newfiles)    
    processThread1 = threading.Thread(target=get_mall_wifi_dic, args=[newfiles[:int(length*0.5)]]) # <- 1 element list
    processThread1.start()
    processThread2 = threading.Thread(target=get_mall_wifi_dic, args=[newfiles[int(length*0.5):]])  # <- 1 element list
    processThread2.start()
    processThread1.join()
    processThread2.join()
    
def get_mall_shop_dic_threading():
    filelist = listfiles(malldir)
    newfiles = []
    for file in filelist:
        mall_id = os.path.basename(file)[:-4]
        path = mallshop_dic_add_conect_dir+mall_id
        if os.path.exists(path):
            continue
        newfiles.append(file)
    length = len(newfiles)    
    processThread1 = threading.Thread(target=getmall_shop_add_connect, args=[newfiles[:int(length*0.5)]]) # <- 1 element list
    processThread1.start()
    processThread2 = threading.Thread(target=getmall_shop_add_connect, args=[newfiles[int(length*0.5):]])  # <- 1 element list
    processThread2.start()
    processThread1.join()
    processThread2.join()
    
        
if __name__=="__main__":
    # print(read_dic(mall_wifi_dic_dir+'m_6803'))
    # get_mall_wifi_dic([mallshop_dic_dir+'m_2270'])  
    # get_mall_wifi_dic_threading()
    # filelist = listfiles(mallshop_dic_dir)
    # malls = getlowmall(middle_path)
    # filelist = []
    # for mall in malls:
        # file = mallshop_dic_dir+mall
        # filelist.append(file)
    # get_important_shop_bssid(filelist)
    # print(read_dic(mall_bssid_dic_path))
    # get_mall_shop_dic_threading()
    filelist = listfiles(mallshop_dic_add_conect_dir)
    get_important_shop_bssid_connect(filelist)
    # filelist = listfiles(malldir)
    # getmall_shop_add_connect(filelist)