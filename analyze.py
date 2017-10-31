import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import json
import threading
from sklearn.ensemble import RandomForestClassifier

from commonLib import *

from getPath import *
pardir = getparentdir()
shop_info_path = pardir + '/data/ccf_first_round_shop_info.csv'
shop_behavior_path = pardir + '/data/ccf_first_round_user_shop_behavior.csv'
evaluate_path = pardir + '/data/evaluation_public.csv'

mall_wifi_dic_path = pardir+'/data/mallwifi_dic.txt'
malldir = pardir+'/data/mall/'
evaluate_a_path = pardir+'/data/evaluation_a.csv'
testmalldir = pardir+'/data/rawtestmall/'

mall_lon_dis_train_dir = pardir+'/data/mall_lon_dis_train/'

shop_info_dir = pardir+'/data/shopinfo/'
mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic_reduce'
# mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic_try_5'
mallshop_dic_add_conect_dir = pardir+'/data/mall_shop_add_connect/'
shop_bssid_dic_path = pardir +'/data/shop_choose_bssid_dic'

def getshopinfo():
    data = pd.read_csv(shop_info_path)
    mall_shop_dic = {}
    shop_info_dic = {}
    shop_mall_dic = {}
    shop_cate_dic = {}
    shop_ids = data['shop_id']
    mall_ids = data['mall_id']
    longitudes = data['longitude']
    latitudes = data['latitude']
    cats = data['category_id']
    length = len(mall_ids)
    for i in range(length):
        shop_id = shop_ids[i]
        mall_id = mall_ids[i]
        if not mall_id in mall_shop_dic:
            mall_shop_dic[mall_id] = []
        mall_shop_dic[mall_id].append(shop_id)
        shop_info_dic[shop_id] = [longitudes[i],latitudes[i]]
        shop_mall_dic[shop_id] = mall_id
        shop_cate_dic[shop_id] = cats[i]
    return mall_shop_dic,shop_info_dic,shop_mall_dic,shop_cate_dic
    
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
                
 
def move_to_different_mallfiles(shop_behavior_path,malldir):
    chunksizes = 100
    mall_shop_dic,shop_info_dic,shop_mall_dic = getshopinfo()
    data = pd.read_csv(shop_behavior_path, chunksize = chunksizes)
    for chunk in data:
        shop_ids = chunk['shop_id']
        length = len(shop_ids)
        for i in range(length):
            shop_id = shop_ids.iloc[i]
            mall_id = shop_mall_dic[shop_id]
            path = malldir +mall_id+".csv"
            df = pd.DataFrame(chunk.iloc[[i]])
            write_record(df,path)
            print(i)
            
def move_to_different_test_mallfiles(shop_behavior_path,malldir):
    chunksizes = 100
    # mall_shop_dic,shop_info_dic,shop_mall_dic = getshopinfo()
    data = pd.read_csv(shop_behavior_path, chunksize = chunksizes)
    for chunk in data:
        mall_ids = chunk['mall_id']
        length = len(mall_ids)
        for i in range(length):
            mall_id = mall_ids.iloc[i]
            path = malldir +mall_id+".csv"
            df = pd.DataFrame(chunk.iloc[[i]])
            write_record(df,path)

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
    
def getshoplon_lat(mall_shop_dic,shop_info_dic,files):
    for file in files:
        data = pd.read_csv(file)
        longitudes = data['longitude']
        latitudes = data['latitude']
        shop_ids = data['shop_id']
        mall_id = get_mallid_from_mallpath(file)
        shops = mall_shop_dic[mall_id]
        length = len(longitudes)
        res = []
        for i in range(length):
            arr = [haversine(shop_info_dic[t][0],shop_info_dic[t][1],longitudes[i],latitudes[i]) for t in shops]
            res.append(arr)
        write_dic(res,mall_lon_dis_train_dir+mall_id)
        
def get_shop_lon_threading():
    filelist = listfiles(malldir)
    newfiles = []
    for file in filelist:
        mall_id = os.path.basename(file)[:-4]
        path = mall_lon_dis_train_dir+mall_id
        if not os.path.exists(path):
            newfiles.append(file)
    mall_shop_dic,shop_info_dic,shop_mall_dic =getshopinfo()
    length = len(newfiles)    
    processThread1 = threading.Thread(target=getshoplon_lat, args=[mall_shop_dic,shop_info_dic,newfiles[:int(length*0.5)]]) # <- 1 element list
    processThread1.start()
    processThread2 = threading.Thread(target=getshoplon_lat, args=[mall_shop_dic,shop_info_dic,newfiles[int(length*0.5):]])  # <- 1 element list
    processThread2.start()
    processThread1.join()
    processThread2.join()
    
def getshop_info():
    data = pd.read_csv(shop_info_path)
    mall_ids = data['mall_id']
    for i in range(len(mall_ids)):
        mall_id = mall_ids.iloc[i]
        path = shop_info_dir +mall_id+".csv"
        df = pd.DataFrame(data.iloc[[i]])
        write_record(df,path)
    
# def get_important_shop(mall_id):
    # data = pd.read_csv(shop_info_dir+mall_id+'.csv')
    # shop_ids = data['shop_id']
    # cat_ids = data['category_id']
    # prices = data['price']
    # d = data.groupby(['category_id'])['shop_id','price'].min().reset_index()
    # shop_ids = list(d['shop_id'])
    # return shop_ids
    
def get_important_shop(mall_id):
    data = pd.read_csv(shop_info_dir+mall_id+'.csv')
    shop_ids = data['shop_id']
    lonts = data['longitude']
    lats = data['latitude']
    maxlon = np.max(lonts)
    minlon = np.min(lonts)
    maxlats = np.max(lats)
    minlats = np.min(lats)
    mlon = np.median(lonts)
    mlat = np.median(lats)
    shops = []
    shops += list(data['shop_id'][data['longitude']==maxlon])
    shops += list(data['shop_id'][data['longitude']==minlon])
    shops += list(data['shop_id'][data['latitude']==minlats])
    shops += list(data['shop_id'][data['latitude']==maxlats])
    # shops += list(data['shop_id'][data['longitude']==mlon])
    # shops += list(data['shop_id'][data['latitude']==mlat])
    print(len(shops))
    return shops
    

    
def creat_partial_model(filelist,mall_bssid_dic):
    for file in filelist:
        mall_id = get_mallid_from_mallpath(file)
        data = pd.read_csv(file)
        lon_diss = read_dic(mall_lon_dis_train_dir+mall_id)
        lon_diss = np.array(lon_diss)
        shop_ids = data['shop_id']
        labels = convertLabels(shop_ids,pardir+'/data/labels/'+mall_id)
        labels = np.array([[t] for t in labels])
        dates = data['time_stamp']
        longitudes = np.array(data['longitude'])
        latitudes = np.array(data['latitude'])
        wifi_infos = data['wifi_infos']
        mallbssids = sorted(list(mall_bssid_dic[mall_id]))
        # mallbssids = read_dic(mall_bssid_arr_dir+mall_id)
        bssid_dic = get_bssid_dic(mallbssids)
        train_x = []
        a = 2.5
        d = np.power(10,a)
        for j in range(len(wifi_infos)):
            wifi_info = wifi_infos[j]

            train_d = np.array([d]*(len(mallbssids)))
            bssids,strengths,connects = process_wifi_info(wifi_info)
            length = len(bssids)
            bssid_strength_dic = {}
            for i in range(length):
                bssid = bssids[i]
                if not bssid in mallbssids:
                    continue
                if not bssid in bssid_strength_dic:
                    bssid_strength_dic[bssid]=[]
                bssid_strength_dic[bssid].append(strengths[i])
            for k,v in bssid_strength_dic.items():
                index = bssid_dic[k]
                v1 = [float(t) for t in v]
                strength = np.max(v1)
                t = (strength+50)/-25
                train_d[index] = np.power(10,t)
                # train_d[index] = round(strength/10)
            # print(train_d)
            train_x.append(train_d)
        # print(train_x)
        # break
        longitudes = np.array([[round(t,5)] for t in longitudes])
        latitudes = np.array([[round(t,5)] for t in latitudes])
        train_index,valid_index = get_fix_date(dates)
        # train_x = np.hstack((train_x,lon_diss))
        train_x = np.hstack((train_x,latitudes))
        train_x = np.hstack((train_x,longitudes))
        train_x = np.array(train_x)
        print(np.shape(train_x))
        X_train = train_x[train_index,:]
        y_train = labels[train_index,:]
        X_test = train_x[valid_index,:]
        # X_test=pca.transform(X_test)
        y_test = labels[valid_index,:]
        clf = RandomForestClassifier(random_state=0,max_features = 'auto',n_estimators=50,n_jobs=3)
        clf.fit(X_train, y_train) 
        p = clf.predict(X_test)
        y_test = [t[0] for t in y_test]
        loss = my_custom_loss_func(y_test,p)
        print(mall_id+":"+str(loss))
  
def get_bssid_dic(bssids):
    bssids = list(bssids)
    dic = {}
    for i in range(len(bssids)):
        dic[bssids[i]] = i
    return dic
    
if __name__=="__main__":
    mall_bssid_dic = read_dic(mall_bssid_dic_path)
    # mall_shop_dic,shop_info_dic = getshopinfo()
    # detect_null_value()
    # move_to_different_mallfiles(shop_behavior_path,malldir)    
    # mall_wifi_dic = get_user_shop_info()
    # print(mall_wifi_dic)
    # getmall_wifi_dic_info()
    # print_mall_shop()
    # move_to_different_test_mallfiles(evaluate_a_path,testmalldir)
    # get_shop_lon_threading()
    # get_user_shop_info()
    # getshop_info()
    # get_important_shop('m_615')
    filelist = listfiles(malldir)
    creat_partial_model(filelist,mall_bssid_dic)
    filelist = listfiles(mallshop_dic_add_conect_dir)
    for file in filelist:
        mall_shop_dic = read_dic(file)
        # tempdict = sorted(mall_shop_dic.items(),key=lambda d:len(d[1]))
        for shop_id,bssiddic in mall_shop_dic.items():
            tempdict = sorted(bssiddic.items(),key=lambda d:len(d[1]['strength']))
            # a = [np.max(d[1]['strength']) for d in tempdict]
            # c = [np.mean(d[1]['strength']) for d in tempdict]
            # d = [np.median(d[1]['strength']) for d in tempdict]
            e = [len(d[1]['strength']) for d in tempdict]
            b = [d[0] for d in tempdict]
            # print(len(a))
            # print(a)
            # print(b)
            # print(c)
            # print(d)
            print(e)
            break
        break
    
