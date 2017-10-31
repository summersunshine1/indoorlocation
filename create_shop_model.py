import numpy as np
import pandas as pd
from commonLib import *
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.externals import joblib
from sklearn.svm import SVR,LinearSVR,SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from getPath import *
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import copy

from analyze import *
pardir = getparentdir()
shop_bssid_dic_path = pardir +'/data/shop_choose_bssid_dic'
mallshop_dic_dir = pardir+'/data/mall_shop_dic/'
mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic_try_5'
mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic_reduce'
mall_mix_bssid_path = pardir +'/data/mall_mix_bssid_dic_remove_3_20_remove_hot'
newlabels_dir = pardir+'/data/newlabels/'
newtrain_dir = pardir+'/data/newtrain/'
mall_dir = pardir+'/data/mall/'
model_dir = pardir+'/data/model/'
test_mall_dir = pardir+'/data/rawtestmall/'
res_path = pardir+'/data/res/rf_dis_model_shop_tune_ratio.csv'
res_one_shop = pardir+'/data/res/rf_dis_model_shop.csv'
evaluate_b_path = pardir+'/data/evaluation_b.csv'
middle_path = pardir+'/data/modeloutput'
temp_path = pardir+'/data/tempout'

mall_bssid_arr_dir = pardir+'/data/mall_bssid/'
mall_lon_dis_train_dir = pardir+'/data/mall_lon_dis_train/'
mall_shop_path = pardir+'/data/mall_lon_dic'
onevsrestdir = pardir+'/data/onevsrest/'
splitresdir = pardir +'/data/res/splitres'

def getindex(mall_id,mall_shop_dic):
    shops = get_important_shop(mall_id)
    shop_ids = mall_shop_dic[mall_id]
    index = []
    for i in range(len(shop_ids)):
        if shop_ids[i] in shops:
            index.append(i)
    return index
    
def getnewindex(data,mall_shop_dic,shop_cate_dic,mall_id):
    df = pd.DataFrame({'count':data.groupby(['shop_id'])['user_id'].apply(set).map(len).astype(np.int32)}).reset_index()
    df = df.sort_values(['count'],ascending=False)
    # merchant['brand_set']=data.groupby("merchant_id")['brand_id'].apply(set)
    # merchant['brand_num']=(merchant['brand_set'].map(len)).astype(np.int16)
    # l = len(df['count'][df['count']<10])/len(df)
    shops = np.array(df['shop_id'])
    selectshops = shops[[0,-1]]
    # if l<0.2:
        # selectshops = shops[-1:]#[shops[0]]
    # else:
        # selectshops = shops[:3]
    shop_ids = mall_shop_dic[mall_id]
    index = []
    cats = set()
    for i in range(len(shop_ids)):
        if shop_ids[i] in selectshops:
            # if shop_cate_dic[shop_ids[i]] in cats:
                # continue
            # cats.add(shop_cate_dic[shop_ids[i]])
            index.append(i)
    return index,selectshops
    
def get_nearest_shops(londiss,mall_shop_dic,mall_id):
    dic = {}
    shops = mall_shop_dic[mall_id]
    for i in range(len(londiss)):
        dic[shops[i]] = londiss[i]
    dict = sorted(dic.items(),key=lambda d:d[1])
    a = [d[1] for d in dict]
    # print(a[:10])
    b = [d[0] for d in dict]
    # for i in range(len(a)):
        # if a[i]>=80:
            # break
    return b[:3]
    
def get_shop_ids_index(shop_ids):
    res_dic = {}
    for i in range(len(shop_ids)):
        if not shop_ids[i] in res_dic:
            res_dic[shop_ids[i]] = []
        res_dic[shop_ids[i]].append(i)
    return res_dic
    
def getcolindex(bssid_dic,shops,mall_id):
    col_index = []
    mall_bssids = getshop_bssid(shops,mall_id)
    for bssid in mall_bssids:
        col_index.append(bssid_dic[bssid])
    col_index = sorted(col_index)
    col_index += [-3,-2,-1]
    return col_index
    
def create_two_classification(clusterdic,bssid_dic,train_x,train_y,shop_ids_index,mall_id):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    dir = mkdir(cluster_model_dir,mall_id)
    # print(shop_ids_index)
    for k,v in clusterdic.items():
        row_indexs = []
        for shop in v:
            if not shop in shop_ids_index:
                continue
            index = shop_ids_index[shop]
            row_indexs += index
        row_indexs = sorted(row_indexs)
        col_indexs = getcolindex(bssid_dic,v,mall_id)
        tr_x = train_x[row_indexs,:]
        # tr_x = tr_x[:,col_indexs]
        tr_y = train_y[row_indexs,:]
        classes = len(set([t[0] for t in tr_y]))
        clf = XGBClassifier(nthread=4,seed = 0)
        # xgb_param = clf.get_xgb_params()
        # xgb_param['num_class'] = classes
        # xgb_param['objective'] = "multi:softmax"
        # objective="multi:softmax"
        # clf = RandomForestClassifier(random_state=0,max_features = 'auto',n_estimators=50,n_jobs=3)
        print(np.shape(tr_x))
        print(np.shape(tr_y))
        clf.fit(tr_x,tr_y)
        joblib.dump(clf,dir+"/"+str(k))
        
def getratio(shop):
    data = pd.read_csv(res_one_shop)
    hits = len(data[data['shop_id']==shop])
    if hits!=0:
        ratio = len(data[data['shop_id']!=shop])/(len(data[data['shop_id']==shop]))
    else:
        ratio = 1
    return ratio

        
def onevsrest(bssid_dic,train_x,train_y,shop_ids_train,mall_id,shop_ids):
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    dir = mkdir(onevsrestdir,mall_id)
    totalindex = list(range(len(train_y)))
    for shop in shop_ids:
        mainindex = (shop_ids_train==shop)
        restindex = (shop_ids_train!=shop)
        tr_y = np.array(copy.copy(train_y))
        col_indexs = getcolindex(bssid_dic,[shop],mall_id) 
        tr_x = np.array(copy.copy(train_x))
        # tr_x = tr_x[:,col_indexs]
        tr_y[mainindex,:] = 1
        tr_y[restindex,:] = 0
        ratio = getratio(shop)
        if len(mainindex)==0 or len(restindex)==0:  
            continue
        clf = XGBClassifier(nthread=4,seed = 0,subsample=0.8,colsample_bytree=0.8,max_depth=10,n_estimators=150)
        # clf = RandomForestClassifier(random_state=0,max_features = 'auto',n_estimators=50,n_jobs=4)
        clf.fit(tr_x,tr_y)
        joblib.dump(clf,dir+"/"+str(shop))

    
def getcluster(actual,predict,mall_id):
    clustershops = {}
    cluster = 1
    for i in range(len(actual)):
        k1 = -1
        k2 = -1
        if actual[i] in clustershops:
            k1 = clustershops[actual[i]]
        if predict[i] in clustershops:
            k2 = clustershops[predict[i]]
        if k1!=k2:
            if k1 == -1:
                clustershops[actual[i]] = k2
            elif k2==-1:
                clustershops[predict[i]] = k1
                
            else:
                for k,v in clustershops.items():
                    if v==k1:
                        clustershops[k] == k2
        else:
            if k1==k2==-1:
                clustershops[actual[i]]=cluster
                clustershops[predict[i]]=cluster
                cluster+=1
    res = {}
    for k,v in clustershops.items():
        if not v in res:
            res[v] = set()
        shop = getlabels_detail(k,newlabels_dir+mall_id)
        res[v].add(shop)
    for k,v in res.items():
        res[k] = list(v)
    return res,clustershops

def get_shop_same_location(mall):
    mall_shop_dic,shop_info_dic,shop_mall_dic,shop_cate_dic =getshopinfo()
    shops = mall_shop_dic[mall]
    resdic = {}
    for shop in shops:
        if not shop_info_dic[shop][0] in resdic:
            resdic[shop_info_dic[shop][0]] = [shop]
        else:
            resdic[shop_info_dic[shop][0]].append(shop)
    tempdict = sorted(resdic.items(),key=lambda d:len(d[1]))
    a = [d[1] for d in tempdict]
    return a[-1]
    
def getshop_bssid(shops,mall_id):
    shop_bssid_dic = read_dic(shop_bssid_dic_path)
    mall_bssid = set()
    for shop in shops:
        if not shop in shop_bssid_dic[mall_id]:
            continue
        for bssid in shop_bssid_dic[mall_id][shop]:
            mall_bssid.add(bssid)
    return mall_bssid
    
def get_test_cluster_index(ptest,resclusterdic):
    cluster_index_dic = {}
    for i in range(len(ptest)):
        plabel = ptest[i]
        if not plabel in resclusterdic:
            print(str(plabel)+" not exists")
            continue
        cluster = resclusterdic[plabel]
        if not cluster in cluster_index_dic:
            cluster_index_dic[cluster] = []
        cluster_index_dic[cluster].append(i)   
    return cluster_index_dic  

def getpredictindex(shops,labels,mall_id):
    res = []
    res = np.argmax(labels,axis = 1)
    shopres = []
    for r in res:
        shopres.append(shops[r])
    res = get_labels(shopres,newlabels_dir+mall_id) 
    return res
    
def getpredictres(shops,labels,mall_id):
    res = np.argmax(labels,axis = 1)
    shopres = []
    for r in res:
        shopres.append(shops[r])
    return shopres

def get_train(filelist,mall_bssid_dic,istest):
    print(len(filelist))
    # if not istest:
    if os.path.exists(middle_path):
        os.remove(middle_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    mall_shop_dic,shop_info_dic,shop_mall_dic,shop_cate_dic =getshopinfo()
    train_dic = {}
    d =np.power(10,2.5)
    # d=0
    kind = ['regular', 'borderline1', 'borderline2', 'svm']
    # sm = [SMOTE(kind=k) for k in kind]
    shop_bssid_dic = read_dic(shop_bssid_dic_path)
    for file in filelist:
        mall_id = get_mallid_from_mallpath(file)
        
        data = pd.read_csv(file)
        df = pd.DataFrame({'count':data.groupby(['shop_id']).size()}).reset_index()
        # shops = df['shop_id'][df['count']<2]
        # shops = get_shop_same_location(mall_id)
    
        indexs,selectshops = getnewindex(data,mall_shop_dic,shop_cate_dic,mall_id)

        train_dic[mall_id] = selectshops
        lon_diss = read_dic(mall_lon_dis_train_dir+mall_id)
        lon_diss = np.array(lon_diss)

        diss = lon_diss.astype(int)[:,indexs]
        wifi_infos = data['wifi_infos']
        shop_ids = data['shop_id']
        shop_ids = np.array(list(shop_ids))
        # print(shop_ids)
        dates = data['time_stamp']
        datetimes = pd.to_datetime(dates)
        user_ids = data['user_id']
        longitudes = np.array(data['longitude'])
        latitudes = np.array(data['latitude'])
        user_ids_c = np.array([[int(t[2:])] for t in user_ids])
        labels = convertLabels(shop_ids,newlabels_dir+mall_id)
        labels = np.array([[t] for t in labels])
        mallbssids = sorted(list(mall_bssid_dic[mall_id]))
        
        # print(mallbssids)
        # break
        # mallbssids = read_dic(mall_bssid_arr_dir+mall_id)
        bssid_dic = get_bssid_dic(mallbssids)
        train_x = []
        whetherconnect = []
        lenbssid = []
        maxbssid = []
        shoptimesbefore = []
        times = []
        
        shopdis = []
        for j in range(len(wifi_infos)):
            wifi_info = wifi_infos[j]
            dis = lon_diss[j]
            times.append([datetimes[j].hour])
            bssid_strength_dic = {}
            bssid_connect = {}

            train_d = np.array([d]*(len(mallbssids)))
            bssids,strengths,connects = process_wifi_info(wifi_info)
            length = len(bssids)
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
                # if strength<-90:
                    # strength = -100
                t = (strength+50)/-20
                train_d[index] = np.power(10,t)
                # train_d[index] = round(strength/10)
            train_x.append(train_d)
            lenbssid.append(length)
            maxbssid.append(round(np.max(strength)/10))
        longitudes = np.array([[round(t,5)] for t in longitudes])
        latitudes = np.array([[round(t,5)] for t in latitudes])
        connects = np.array(whetherconnect)
        lenbssid = np.array([[t] for t in lenbssid])
        maxbssid = np.array([[t] for t in maxbssid])
        train_x = np.hstack((train_x,latitudes))
        train_x = np.hstack((train_x,longitudes))

        train_x = np.hstack((train_x,diss))
        train_x = np.array(train_x)
        print(mall_id)
     
        dates = np.array(dates)
        train_index,valid_index = get_fix_date(dates)
        if istest:
            X_train = train_x[train_index,:]
            y_train = labels[train_index,:]
           
            X_test = train_x[valid_index,:]
            y_test = labels[valid_index,:]
            y_test = np.array([t[0] for t in y_test])
            train_shop_ids = np.array(shop_ids[train_index])
            test_shop_ids = np.array(shop_ids[valid_index])
        
        else:
            X_train = train_x
            y_train = labels
            train_shop_ids = np.array(shop_ids)
        
        
        shops = mall_shop_dic[mall_id]
        onevsrest(bssid_dic,X_train,y_train,train_shop_ids,mall_id,shops)
        
        if istest:
            tempres = []
            for shop in shops:
                colindex = getcolindex(bssid_dic,[shop],mall_id)
                path = onevsrestdir+'/'+mall_id+'/'+shop
                if not os.path.exists(path):
                    p = np.array([[-1]]*len(valid_index))
                    if len(tempres)==0:
                        tempres = p
                        continue
                    tempres = np.hstack((tempres,p))
                    continue
                clf = joblib.load(path)
                # p = clf.predict_proba(X_test[:,colindex])
                p = clf.predict_proba(X_test)
                if np.shape(p)[1] == 1:
                    continue
                p = [[t[1]] for t in p]
                if len(tempres)==0:
                    tempres = p
                else:
                    tempres = np.hstack((tempres,p))
            tempres = np.array(tempres)
            res = getpredictindex(shops,tempres,mall_id)  
            loss1 = my_custom_loss_func(y_test,res)
            print(loss1)
            df = pd.DataFrame()
            res = np.array(res)
            y_test = np.array(y_test)
            df['pred'] = getlabels_detail(res[y_test!=res],newlabels_dir+mall_id)
            df['actual'] = getlabels_detail(y_test[y_test!=res],newlabels_dir+mall_id)
            lines = mall_id+":"+str(loss1)#+" "+str(loss1)
            # if not istest:
            write_middle_res(lines)
       
    write_dic(train_dic,mall_shop_path)
    
        
def predict(filelist,mall_bssid_dic):
    mall_shop_dic,shop_info_dic,shop_mall_dic,shop_cate_dic =getshopinfo()
    res_dic = {}
    if os.path.exists(res_path):
        os.remove(res_path)
    train_dic = read_dic(mall_shop_path)
    for file in filelist:
        lenbssid = []
        mall_id = get_mallid_from_mallpath(file)
        print(mall_id)
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        row_ids = data['row_id']
        shops = train_dic[mall_id]
        shop_ids = mall_shop_dic[mall_id]
        tempdic = {}
        for t in range(len(shop_ids)):
            tempdic[shop_ids[t]]=t    
        mallbssids = sorted(list(mall_bssid_dic[mall_id]))
        longitudes = np.array(data['longitude'])
        latitudes = np.array(data['latitude'])
        # mallbssids = read_dic(mall_bssid_arr_dir+mall_id)
        bssid_dic = get_bssid_dic(mallbssids)
        train_x = []
        shopdis = []
        for j in range(len(wifi_infos)):
            wifi_info = wifi_infos[j]
            bssid_strength_dic = {}
            d = np.power(10,2.5)
            train_d = np.array([d]*len(mallbssids))
            bssids,strengths,connects = process_wifi_info(wifi_info)
            dis1 = int(haversine(longitudes[j], latitudes[j], shop_info_dic[shops[0]][0], shop_info_dic[shops[0]][1]))
            dis2 = int(haversine(longitudes[j], latitudes[j], shop_info_dic[shops[1]][0], shop_info_dic[shops[1]][1]))
            if tempdic[shops[0]]<tempdic[shops[1]]:
                shopdis.append([dis1,dis2])
            else:
                shopdis.append([dis2,dis1])
            length = len(bssids)
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
                t = (strength+50)/-20
                train_d[index] = np.power(10,t)
                # train_d[index] = int(round(strength/10))
            train_x.append(train_d)
            lenbssid.append([length])
  
        longitudes = np.array([[round(t,5)] for t in longitudes])
        latitudes = np.array([[round(t,5)] for t in latitudes])
        train_x = np.hstack((train_x,latitudes))
        train_x = np.hstack((train_x,longitudes))
        train_x = np.hstack((train_x,shopdis))
        user_ids = data['user_id']
        user_ids = np.array([[int(t[2:])] for t in user_ids])
        lenbssid = np.array(lenbssid)
        
        model_dir = onevsrestdir+'/'+mall_id
        model_paths = listfiles(model_dir) 
        tempres = []
        tempshops = []
        for path in model_paths:
            clf = joblib.load(path)
            shop = os.path.basename(path)
            colindex = getcolindex(bssid_dic,[shop],mall_id)
            p = clf.predict_proba(train_x[:,colindex])
            tempshops.append(shop)
            p = [[t[1]] for t in p]
            if len(tempres)==0:
                tempres = p
            else:
                tempres = np.hstack((tempres,p))
        tempres = np.array(tempres)
        res = getpredictres(tempshops,tempres,mall_id) 
        df = pd.DataFrame()
        df['row_id'] = row_ids
        df['shop_id'] = res
        write_record(df,splitresdir+'/'+mall_id+'.csv')
    

def create_model_threading(mall_bssid_dic,istest):
    filelist = listfiles(mall_dir)
    newfiles = []
    for file in filelist:
        mall_id = get_mallid_from_mallpath(file)
        path = model_dir+mall_id
        if os.path.exists(path):
            continue
        newfiles.append(file)
    length = len(newfiles)    
    processThread1 = threading.Thread(target=get_train, args=[newfiles[:int(length*0.5)],mall_bssid_dic,istest]) # <- 1 element list
    processThread1.start()
    processThread2 = threading.Thread(target=get_train, args=[newfiles[int(length*0.5):],mall_bssid_dic,istest])  # <- 1 element list
    processThread2.start()
    processThread1.join()
    processThread2.join()

        
def write_middle_res(lines):
    with open(pardir+'/data/modeloutput', 'a',encoding = 'utf-8') as f:
        f.writelines(lines+'\n')
        
def get_bssid_dic(bssids):
    bssids = list(bssids)
    dic = {}
    for i in range(len(bssids)):
        dic[bssids[i]] = i
    return dic
    
def get_temp_train_list():
    filelist = listfiles(mall_lon_dis_train_dir)
    files = []
    for file in filelist:
        mall_id = os.path.basename(file)
        files.append(mall_dir+mall_id+'.csv')
    return files
    
def combineres():
    
    files = listfiles(splitresdir)
    for file in files:
        data = pd.read_csv(file)
        write_record(data,res_path)
    append_res_file(res_path,evaluate_b_path)
    remove_replicate_res(res_path)
        
if __name__=="__main__":
    mall_bssid_dic = read_dic(mall_bssid_dic_path)
    # mall_bssid_dic = read_dic(mall_mix_bssid_path)
    # print(sorted(list(mall_bssid_dic['m_4079'])))
    # print(sorted(list(mall_bssid_dic['m_4079'])))
    # print(len(mall_bssid_dic.keys()))
    # files = []
    # for k,v in mall_bssid_dic.items():
        # files.append(mall_dir+k+'.csv')
    files = listfiles(mall_dir)
    # dic = {}
    # print(len(files))
    # for file in files:
        # print(file)
        
    # print(files)
    # print(len(files))
    # files = get_temp_train_list()
    # print(files)
    # files = [mall_dir+'m_4168.csv']
    # files = [mall_dir+'m_6803.csv']
    get_train(files,mall_bssid_dic,1)
    # create_model_threading(mall_bssid_dic,0)
    # choosefiles = []
    # files = listfiles(test_mall_dir)
    # splitfiles = listfiles(splitresdir)
    # for file in files:
        # mall_id = get_mallid_from_mallpath(file)
        # splitpath = splitresdir+'/'+mall_id+'.csv'
        # if os.path.exists(splitpath):
            # continue
        # choosefiles.append(file)
    # print(choosefiles)
    # print(len(files))
    # predict(choosefiles,mall_bssid_dic)
    # remove_replicate_res(res_path)
    # combineres()
        
                

    