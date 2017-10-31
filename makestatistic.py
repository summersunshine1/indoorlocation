import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wight_model import *
from commonLib import *
import threading

malldir = pardir+'/data/mall/'
testmalldir = pardir+'/data/rawtestmall/'
mallshop_dic_dir = pardir+'/data/mall_shop_dic/'
mallshop_train_dic_dir = pardir+'/data/mall_shop_dic_train/'
mall_wifi_dic_dir = pardir+'/data/mall_wifi_dic/'
mall_wifi_dic_remove_dir = pardir+'/data/mall_wifi_dic_remove/'
mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic_reduce'
shop_bssid_dic_path = pardir +'/data/shop_choose_bssid_dic'
mall_mix_bssid_path = pardir +'/data/mall_mix_bssid_dic_remove_3_20_remove_hot'
middle_path = pardir+'/data/middle'
mall_wifi_test_dic_path = pardir+'/data/mall_wifi_dic_test_dic'
mall_wifi_valid_dic_path = pardir+'/data/mall_wifi_dic_valid_dic'

mallshop_dic_add_conect_dir_remove = pardir+'/data/mall_shop_add_connect_remove/'
mallshop_dic_add_conect_totaldir = pardir+'/data/mall_shop_add_connect_total/'
mall_bssid_arr_dir = pardir+'/data/mall_bssid/'
mallshop_dic_add_conect_dir = pardir+'/data/mall_shop_add_connect/'

def getmall_important_bssid(paths):
    for path in paths:
        bssidset = set()
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
            # str = [int(t) for t in strengths]
            dic = dict(zip(bssids, strengths))
            res = sorted(dic.items(),key=lambda d:d[1])
            list1, list2 = zip(*res)
            connects = np.array(connects)
            bssids = np.array(bssids)
            connect = bssids[connects=='true']
            # print(res)
            if len(list1)<3:
                for b in list1:
                    bssidset.add(b)
                continue
            else:
                for b in list1[:3]:
                    bssidset.add(b)
            for c in connects:
                bssidset.add(c)
        print(mall_id)    
        print(len(bssidset))  
        
        write_dic(bssidset,mall_bssid_arr_dir+mall_id)     


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
            temp_dic = {}
            if not shop_id in shop_wifi_dic:
                shop_wifi_dic[shop_id] = {}
            for j in range(len(bssids)):
                if bssids[j] in temp_dic:
                    continue
                temp_dic[bssids[j]]=1
                if not bssids[j] in shop_wifi_dic[shop_id]:
                    shop_wifi_dic[shop_id][bssids[j]]={}
                if not 'strength' in shop_wifi_dic[shop_id][bssids[j]]:
                    shop_wifi_dic[shop_id][bssids[j]]['strength']=[]
                    shop_wifi_dic[shop_id][bssids[j]]['connect']=[]
                shop_wifi_dic[shop_id][bssids[j]]['strength'].append(strengths[j])
                shop_wifi_dic[shop_id][bssids[j]]['connect'].append(connects[j])
        write_dic(shop_wifi_dic,mallshop_dic_add_conect_dir+mall_id)   
        
def gettotal_mall_shop_add_connect(paths):
    for path in paths:
        data = pd.read_csv(path)
        mall_id = os.path.basename(path)[:-4]
        shop_ids = data['shop_id']
        wifi_infos = data['wifi_infos']
        length = len(shop_ids)
        shop_wifi_dic = {}
        # train_index,valid_index = get_fix_date(data['time_stamp'])
        for i in range(length):
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
        write_dic(shop_wifi_dic,mallshop_dic_add_conect_totaldir+mall_id)  

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
        path = mall_wifi_dic_remove_dir+mall_id
        bssid_dic = {}
        for shop_id,bssiddic in dic.items():
            for bssid,dic in bssiddic.items():
                if not bssid in bssid_dic:
                    bssid_dic[bssid]=[]
                bssid_dic[bssid]+=list(dic['strength'])
        write_dic(bssid_dic,path)
            
def get_test_mall_wifi_dic(files):
    dic = {}
    for file in files:
        mall_id = get_mallid_from_mallpath(file)
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        idset = set()
        for wifi_info in wifi_infos:
            bssids,strengths,connects = process_wifi_info(wifi_info)
            for id in bssids:
                idset.add(id)
        dic[mall_id] = list(idset)
    write_dic(dic,mall_wifi_test_dic_path)
    
def get_valid_mall_wifi_dic(files):
    dic = {}
    for file in files:
        mall_id = get_mallid_from_mallpath(file)
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        idset = set()
        for wifi_info in wifi_infos:
            bssids,strengths,connects = process_wifi_info(wifi_info)
            for id in bssids:
                idset.add(id)
        dic[mall_id] = list(idset)
    write_dic(dic,mall_wifi_valid_dic_path)
    
def compare():
    test_dic = read_dic(mall_wifi_test_dic_path)
    all_dic = read_dic(mall_wifi_valid_dic_path)
    for k,v in test_dic.items():
        vallset = set(all_dic[k])
        vtestset = set(v)
        print(vallset-vtestset)
        print(k)
        break
     
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
    shop_bssiddic = {}
    missing = []
    co = 0
    testdic = read_dic(mall_wifi_test_dic_path)
    for file in filelist:
        mall_id = os.path.basename(file) 
        mall_shop_dic = read_dic(file)
        path = mall_wifi_dic_dir+mall_id
        mall_wifi_dic = read_dic(path)
        mall_bssid_set = set()
        dict = sorted(mall_wifi_dic.items(),key=lambda d:len(d[1]))
        a = [len(d[1]) for d in dict]
        b = [d[0] for d in dict]
        for t in b[-5:]:
            mall_bssid_set.add(t)
        tempset = set()
        if not mall_id in shop_bssiddic.items():
            shop_bssiddic[mall_id] = {}
            
        for shop_id,bssiddic in mall_shop_dic.items():
            shop_impor_dic = {}
            # bssidlen = len(list(bssiddic.keys()))
            shop_bssiddic[mall_id][shop_id]=set()
            tempdict = sorted(bssiddic.items(),key=lambda d:np.max([int(t) for t in d[1]['strength']]))
            a = [len(d[1]) for d in tempdict]
            b = [d[0] for d in tempdict]
            for t in b[-3:]:
                mall_bssid_set.add(t) 
                shop_bssiddic[mall_id][shop_id].add(t)
            flag = 0
            for bssid,dic in bssiddic.items():
                bssidlen = len(mall_wifi_dic[bssid])
                strengths = dic['strength']
                connects = dic['connect']
                connects = np.array(connects)
                # print(connects)
                # print("end")
                # break
                length = len(connects[connects=="true"])
                tlength = len(connects)
                if tlength>length and length>=1:
                    mall_bssid_set.add(bssid)
                    shop_bssiddic[mall_id][shop_id].add(bssid)
                # if not bssid in testdic[mall_id]:
                    # continue
                str = [int(t) for t in strengths]
                # if len(str)>2:
                    # str.remove(np.max(str))
                    # str.remove(np.min(str))
                # shop_impor_dic[bssid]=(len(strengths))/((np.median(str))/(np.max(str)))
                # shop_impor_dic[bssid]=len(strengths)/np.square(np.max(str))
                # str = [np.power(10,(t+50)/-20) for t in str]
                # if np.max(str)<-90:
                    # continue
                shop_impor_dic[bssid]=((len(strengths)))/-(np.max(str))
            dict = sorted(shop_impor_dic.items(),key=lambda d:d[1])
            a = [d[1] for d in dict]
            b = [d[0] for d in dict]
            for t in b[-15:]:
                mall_bssid_set.add(t)
                shop_bssiddic[mall_id][shop_id].add(t)
                tempset.add(t)
        bssiddics[mall_id] = list(mall_bssid_set)
        print(mall_id)
        print(len(mall_bssid_set))
        print(len(tempset))
        # co+=1
        # if co>3:
            # break

    write_dic(bssiddics,mall_bssid_dic_path)
    write_dic(shop_bssiddic,shop_bssid_dic_path)
            
def get_mall_wifi_dic_threading():
    filelist = listfiles(mallshop_dic_add_conect_dir_remove)
    newfiles = []
    for file in filelist:
        mall_id = os.path.basename(file) 
        path = mall_wifi_dic_remove_dir+mall_id
        # if os.path.exists(path):
            # continue
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
        # if os.path.exists(path):
            # continue
        newfiles.append(file)
    length = len(newfiles)    
    processThread1 = threading.Thread(target=getmall_shop_add_connect, args=[newfiles[:int(length*0.5)]]) # <- 1 element list
    processThread1.start()
    processThread2 = threading.Thread(target=getmall_shop_add_connect, args=[newfiles[int(length*0.5):]])  # <- 1 element list
    processThread2.start()
    processThread1.join()
    processThread2.join()
    
def get_mall_shop_bssid_threading():
    filelist = listfiles(malldir)
    newfiles = []
    for file in filelist:
        mall_id = os.path.basename(file)[:-4]
        path = mall_bssid_arr_dir+mall_id
        # if os.path.exists(path):
            # continue
        newfiles.append(file)
    length = len(newfiles)    
    processThread1 = threading.Thread(target=getmall_important_bssid, args=[newfiles[:int(length*0.5)]]) # <- 1 element list
    processThread1.start()
    processThread2 = threading.Thread(target=getmall_important_bssid, args=[newfiles[int(length*0.5):]])  # <- 1 element list
    processThread2.start()
    processThread1.join()
    processThread2.join()
    
def getunionset():
    mall_bssid_dic = read_dic(mall_bssid_dic_path)
    files =listfiles(mall_bssid_arr_dir)
    mall_dic = {}
    for file in files:
        mall_id = os.path.basename(file)
        bssid1 = set(mall_bssid_dic[mall_id])
        bssid2 = set(read_dic(file))
        mall_dic[mall_id] = bssid1&bssid2
        print(mall_id + ":" + str(len(bssid1)) +" "+ str(len(bssid2))+" union:"+str(len(mall_dic[mall_id])))
    write_dic(mall_dic,mall_mix_bssid_path)
    
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
    # filelist = [mallshop_dic_add_conect_dir+'m_6803']
    # print(read_dic(mallshop_dic_add_conect_dir_remove+'m_6803'))
    # get_important_shop_bssid_connect(filelist)
    # filelist = listfiles(malldir)
    # getmall_shop_add_connect(filelist)
    # get_mall_shop_bssid_threading()
    
    # getunionset()
    # files = listfiles(testmalldir)
    # get_test_mall_wifi_dic(files)
    # files = listfiles(malldir)
    # get_valid_mall_wifi_dic(files)
    # compare()
    dic = read_dic(shop_bssid_dic_path)
    print(sorted(dic['m_4079']['s_180598']))
    print(sorted(dic['m_4079']['s_181552']))
    print(sorted(dic['m_4079']['s_181558']))