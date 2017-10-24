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
from sklearn.ensemble import RandomForestClassifier
from getPath import *
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
pardir = getparentdir()

mallshop_dic_dir = pardir+'/data/mall_shop_dic/'
mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic'
mall_mix_bssid_path = pardir +'/data/mall_mix_bssid_dic_remove_3_20_remove_hot'
newlabels_dir = pardir+'/data/newlabels/'
newtrain_dir = pardir+'/data/newtrain/'
mall_dir = pardir+'/data/mall/'
model_dir = pardir+'/data/model/'
test_mall_dir = pardir+'/data/rawtestmall/'
res_path = pardir+'/data/res/rf_max_divde10_convert_feature_add_ll.csv'
evaluate_b_path = pardir+'/data/evaluation_b.csv'
middle_path = pardir+'/data/modeloutput'
temp_path = pardir+'/data/tempout'

mall_bssid_arr_dir = pardir+'/data/mall_bssid/'

def get_train(filelist,mall_bssid_dic,istest):
    print(len(filelist))
    # if not istest:
    if os.path.exists(middle_path):
        os.remove(middle_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    for file in filelist:
        mall_id = get_mallid_from_mallpath(file)
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        shop_ids = data['shop_id']
        dates = data['time_stamp']
        user_ids = data['user_id']
        user_ids = np.array([[int(t[2:])] for t in user_ids])
        labels = convertLabels(shop_ids,newlabels_dir+mall_id)
        labels = np.array([[t] for t in labels])
        mallbssids = sorted(list(mall_bssid_dic[mall_id]))
        # mallbssids = read_dic(mall_bssid_arr_dir+mall_id)
        bssid_dic = get_bssid_dic(mallbssids)
        train_x = []
        whetherconnect = []
        lenbssid = []
        maxbssid = []
        for wifi_info in wifi_infos:
            bssid_strength_dic = {}
            bssid_connect = {}
            # d = 1/np.power(10,10)
            train_d = np.array([-10]*(len(mallbssids)))
            bssids,strengths,connects = process_wifi_info(wifi_info)
            if len(connects[connects=="true"])>=1:
                whetherconnect.append([1])
            else:
                whetherconnect.append([0])
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
                train_d[index] = round(strength/10)
            train_x.append(train_d)
            lenbssid.append(length)
            maxbssid.append(round(np.max(strength)/15))
        longitudes = np.array(data['longitude'])
        latitudes = np.array(data['latitude'])
        longitudes = np.array([[round(t,5)] for t in longitudes])
        latitudes = np.array([[round(t,5)] for t in latitudes])
        # print(np.shape(train_x))
        # print(np.shape(longitudes))
        connects = np.array(whetherconnect)
        lenbssid = np.array([[t] for t in lenbssid])
        maxbssid = np.array([[t] for t in maxbssid])
        train_x = np.hstack((train_x,latitudes))
        train_x = np.hstack((train_x,longitudes))
        # train_x = np.hstack((train_x,maxbssid))
        # data = np.hstack((data,latitudes))
        # data = np.hstack((data,labels))
        # write_dic(data,newtrain_dir+mall_id)
        # train_x = np.power(10,np.array(train_x)/np.max(train_x,))
        train_x = np.array(train_x)
       
        #  print(max)
        # train_x =  np.power(10,np.array(train_x)/10)
        # print(data)
        # mean = [[t] for t in np.mean(train_x,1)]
        # stds = np.std(train_x,1)
        print(mall_id)
        
        # print(shop_ids[arr==0])
        # r = len(arr[arr==0])/np.shape(train_x)[0]
        # l = len(arr)
        # lines = mall_id+":"+ "ratio:"+str(r)+" toatl:"+str(l)+'\n'
        # b = list(shop_ids[arr==0])
        # lines+=','.join(b)+'\n'
       
        # with open(temp_path,'a',encoding='utf-8') as f:
            # f.writelines(lines)
        # train_x = train_x[arr!=0]
        # labels = labels[arr!=0]
        # train_x = (train_x - min)/arr
        
        train_index,valid_index = get_fix_date(dates)
        # X_train, X_test, y_train, y_test = train_test_split(train_x, labels, test_size=0.2, random_state=42)
        features = np.shape(train_x)[1]
        print(features)
        # ch = SelectKBest(chi2, k=int(features*0.8))
        # X_new = ch.fit_transform(X_train, y_train)
        # clf = SVC(kernel ='linear')
        X_train = train_x[train_index,:]
        y_train = labels[train_index,:]
        max = np.array([[t] for t in np.max(X_train,1)])
        min = np.array([[t] for t in np.min(X_train,1)])
        
        arr = max-min
        r = len(arr[arr==0])/len(arr)
        l = len(arr)
        lines = mall_id+":"+ "ratio:"+str(r)+" toatl:"+str(l)+'\n'
        # b = list(shop_ids[arr==0])
        # lines+=','.join(b)+'\n'
       
        with open(temp_path,'a',encoding='utf-8') as f:
            f.writelines(lines)
        
        # X_train = X_train[arr!=0]
        # y_train = y_train[arr!=0]
        # pca=PCA(n_components=int(np.shape(X_train)[1]*0.8))
        # X_train=pca.fit_transform(X_train)
        X_test = train_x[valid_index,:]
        # X_test=pca.transform(X_test)
        y_test = labels[valid_index,:]

        clf = RandomForestClassifier(random_state=0,max_features = 'auto',n_estimators=50,n_jobs=-1)
        clf.fit(X_train, y_train) 

        # print(clf.oob_score_)
        list1 = list(clf.feature_importances_)
        list1 = sorted(list1)
        # print(list1[-10:])
        # X_test_new = ch.transform(X_test)
        p = clf.predict(X_test)
        if not istest:
            clf.fit(train_x, labels)
            joblib.dump(clf, model_dir+mall_id)
        y_test = [t[0] for t in y_test]
        loss = my_custom_loss_func(y_test,p)
        print(mall_id+":"+str(loss))
        lines = mall_id+":"+str(loss)
        # if not istest:
        write_middle_res(lines)
        
def predict(filelist,mall_bssid_dic):
    res_dic = {}
    if os.path.exists(res_path):
        os.remove(res_path)
    for file in filelist:
        lenbssid = []
        mall_id = get_mallid_from_mallpath(file)
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        row_ids = data['row_id']
        
        print(mall_id)
        mallbssids = sorted(list(mall_bssid_dic[mall_id]))
        longitudes = np.array(data['longitude'])
        latitudes = np.array(data['latitude'])
        # mallbssids = read_dic(mall_bssid_arr_dir+mall_id)
        bssid_dic = get_bssid_dic(mallbssids)
        train_x = []
        for wifi_info in wifi_infos:
            bssid_strength_dic = {}
            # d = np.power(10,-10)
            train_d = np.array([-10]*len(mallbssids))
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
                train_d[index] = int(round(strength/10))
            train_x.append(train_d)
            lenbssid.append([length])
           
        longitudes = np.array([[round(t,5)] for t in longitudes])
        latitudes = np.array([[round(t,5)] for t in latitudes])
        train_x = np.hstack((train_x,latitudes))
        train_x = np.hstack((train_x,longitudes))
        
        # maxarr = np.max(train_x,1)
        # max = np.array([[t] for t in maxarr])
        # min = np.array([[t] for t in np.min(train_x,1)])
        # arr = max-min
        # arr = np.array([t[0] for t in arr])
        # print(len(arr[arr==0]))
        user_ids = data['user_id']
        user_ids = np.array([[int(t[2:])] for t in user_ids])
        lenbssid = np.array(lenbssid)
        # train_x = np.hstack((train_x,lenbssid))
        clf = joblib.load(model_dir+mall_id)
        p = clf.predict(train_x)
        labels = getlabels_detail(p,newlabels_dir+mall_id)
        df = pd.DataFrame()
        df['row_id'] = row_ids
        df['shop_id'] = labels
        # print(len(df))
        write_record(df,res_path)
    append_res_file(res_path,evaluate_b_path)
    
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
        
if __name__=="__main__":
    # mall_bssid_dic = read_dic(mall_bssid_dic_path)
    mall_bssid_dic = read_dic(mall_mix_bssid_path)
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
    # get_train(files,mall_bssid_dic,0)
    # create_model_threading(mall_bssid_dic,0)
    files = listfiles(test_mall_dir)
    # print(len(files))
    predict(files,mall_bssid_dic)
    remove_replicate_res(res_path)
    # read_dic(new
        
                

    