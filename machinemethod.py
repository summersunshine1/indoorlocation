import numpy as np
import pandas as pd
import _thread
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.externals import joblib
from sklearn.svm import SVR,LinearSVR,SVC

from commonLib import *
from getPath import *
pardir = getparentdir()

label_pkl_dir = pardir+'/data/labels/'
mall_wifi_dic_path = pardir+'/data/mallwifi_dic.txt'
mall_dir = pardir+'/data/mall/'
testmalldir = pardir+'/data/rawtestmall/'
mall_train_dir = pardir+'/data/mall_train/'
mall_test_dir = pardir+'/data/mall_test/'
pca_test_dir = pardir+'/data/pca_test/'
pca_train_dir = pardir+'/data/pca_train/'
model_dir = pardir+'/data/model/'

res_path = pardir+'/data/res/pca_res.csv'

def get_bssid_dic(bssids):
    bssids = list(bssids)
    dic = {}
    for i in range(len(bssids)):
        dic[bssids[i]] = i
    return dic

def get_mall_data(mall_path,mall_wifi_dic):
    mall_id = get_mallid_from_mallpath(mall_path)
    data = pd.read_csv(mall_path)
    wifi_infos = data['wifi_infos']
    shop_ids = data['shop_id']
    mall_bssids = mall_wifi_dic[mall_id]
    allbssid_len = len(mall_bssids)
    bssid_dic = get_bssid_dic(mall_bssids)
    train_data = []
    for wifi_info in wifi_infos:
        bssid_strength_dic = {}
        train_d = np.array([0]*allbssid_len)
        bssids,strengths = process_wifi_info(wifi_info)
        length = len(bssids)
        for i in range(length):
            bssid = bssids[i]
            if not bssid in bssid_strength_dic:
                bssid_strength_dic[bssid]=[]
            bssid_strength_dic[bssid].append(strengths[i])
        for k,v in bssid_strength_dic.items():
            index = bssid_dic[k]
            v1 = [float(t) for t in v]
            strength = np.mean(v1)
            train_d[index] = strength
        train_data.append(train_d)
    labels = convertLabels(list(shop_ids),label_pkl_dir+mall_id)
    train_data = np.array(train_data)
    labels = [[t] for t in labels]
    data = np.hstack((train_data,labels))
    write_dic(data,mall_train_dir+mall_id)
    
def get_train_data(mall_paths):
    mall_wifi_dic = read_dic(mall_wifi_dic_path)
    for mall_path in mall_paths:
        get_mall_data(mall_path,mall_wifi_dic)
    
def write_train_data():
    files = listfiles(mall_dir)
    length = len(files)
    locks=[]    
    i = 0
    try:
        _thread.start_new_thread(get_train_data, (files[0:24],))
        _thread.start_new_thread(get_train_data, (files[24:48],))
        _thread.start_new_thread(get_train_data, (files[48:72],))
        _thread.start_new_thread(get_train_data, (files[72:97],))
    except:
        print ("Error: unable to start thread")
    while 1:
        time.sleep(1000) 
        
def get_mall_test_data(mall_path,mall_wifi_dic):
    mall_id = get_mallid_from_mallpath(mall_path)
    data = pd.read_csv(mall_path)
    wifi_infos = data['wifi_infos']
    row_ids = data['row_id']
    mall_bssids = mall_wifi_dic[mall_id]
    allbssid_len = len(mall_bssids)
    bssid_dic = get_bssid_dic(mall_bssids)
    train_data = []
    length = len(wifi_infos)
    for i in range(length):
        wifi_info = wifi_infos.iloc[i]
        bssid_strength_dic = {}
        train_d = np.array([0]*allbssid_len)
        bssids,strengths = process_wifi_info(wifi_info)
        length = len(bssids)
        for i in range(length):
            bssid = bssids[i]
            if not bssid in bssid_strength_dic:
                bssid_strength_dic[bssid]=[]
            bssid_strength_dic[bssid].append(strengths[i])
        for k,v in bssid_strength_dic.items():
            if not k in bssid_dic:
                continue
            index = bssid_dic[k]
            v1 = [float(t) for t in v]
            strength = np.mean(v1)
            train_d[index] = strength
        train_data.append(train_d)
    row_ids = [[t] for t in row_ids]
    data = np.hstack((train_data,row_ids))
    write_dic(data,mall_test_dir+mall_id)
    
def get_test_data(mall_paths):
    mall_wifi_dic = read_dic(mall_wifi_dic_path)
    for mall_path in mall_paths:
        get_mall_test_data(mall_path,mall_wifi_dic)
    
def write_test_data():
    files = listfiles(testmalldir)
    length = len(files)
    locks=[]    
    i = 0
    try:
        _thread.start_new_thread(get_test_data, (files[0:24],))
        _thread.start_new_thread(get_test_data, (files[24:48],))
        _thread.start_new_thread(get_test_data, (files[48:72],))
        _thread.start_new_thread(get_test_data, (files[72:97],))
    except:
        print ("Error: unable to start thread")
    while 1:
        time.sleep(1000)
    
def get_pca(mall_train_paths):
    for path in mall_train_paths:
        mall_id = os.path.basename(path)
        train_data = read_dic(path)
        train_data = np.array(train_data)
        train = train_data[:,:-1]
        labels = train_data[:,-1]
        labels = [[t] for t in labels]
        features = np.shape(train)[1]
        if features>5000:
            c = int(features*0.05)
        else:
            c = int(features*0.1)
        pca=PCA(n_components=c)
        newdata=pca.fit_transform(train)
        data = np.hstack((newdata,labels))
        write_dic(data,pca_dir+mall_id)
        
def get_all_pca(mall_train_paths):
    for path in mall_train_paths:
        mall_id = os.path.basename(path)
        if os.path.exists(pca_train_dir+mall_id):
            continue
        train_data = read_dic(path)
        test_mall_path = mall_test_dir+mall_id
        test_data = read_dic(test_mall_path)
        
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_len = np.shape(train_data)[0]
        test_len = np.shape(test_data)[0]
        
        all_data = np.vstack((train_data,test_data))
        train = all_data[:,:-1]
        features = np.shape(train)[1]
        if features>5000:
            c = int(features*0.05)
        else:
            c = int(features*0.1)
        pca=PCA(n_components=c)
        newdata=pca.fit_transform(train)
        
        train_t = newdata[:train_len,:]
        test_t = newdata[train_len:,:]
        train_label = train_data[:,-1]
        train_label = [[t] for t in train_label]
        test_label = test_data[:,-1]
        test_label = [[t] for t in test_label]
        
        train_data = np.hstack((train_t,train_label))
        write_dic(train_data,pca_train_dir+mall_id)
        test_data = np.hstack((test_t,test_label))
        write_dic(test_data,pca_test_dir+mall_id)
        
        
def write_pca():
    files = listfiles(mall_train_dir)
    length = len(files) 
    newfiles = []
    for file in files:
        mall_id = os.path.basename(file)
        if os.path.exists(pca_train_dir+mall_id):
            continue
        newfiles.append(file)
    length = len(newfiles)    
    try:
        _thread.start_new_thread(get_all_pca, (newfiles[0:int(length*0.25)],))
        _thread.start_new_thread(get_all_pca, (newfiles[int(length*0.25):int(length*0.5)],))
        _thread.start_new_thread(get_all_pca, (newfiles[int(length*0.5):int(length*0.75)],))
        _thread.start_new_thread(get_all_pca, (newfiles[int(length*0.75):],))
    except Exception as e: 
        print(e)
    while 1:
        time.sleep(1000)
    
def my_custom_loss_func(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    print(ground_truth)
    print(predictions)
    predictions = np.array(predictions)
    return len(ground_truth[ground_truth == predictions])/len(ground_truth)
        
def create_model(paths):
    for path in paths:
        mall_id = os.path.basename(path)
        pca_data = read_dic(path)
        pca_data = np.array(pca_data)
        x = pca_data[:,:-1]
        y = pca_data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        clf = LogisticRegression()
        clf.fit(x, y)   
        score = make_scorer(my_custom_loss_func, greater_is_better=True)
        scores = -cross_val_score(clf, x, y,cv=10,scoring=score)
        print(scores)
        print(np.mean(scores))
        joblib.dump(clf, model_dir+mall_id)
        
def create_model_split(paths):
    for path in paths:
        mall_id = os.path.basename(path)
        if os.path.exists(model_dir+mall_id):
            continue

        pca_data = read_dic(path)
        pca_data = np.array(pca_data)
        x = pca_data[:,:-1]
        y = pca_data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        clf = SVC(kernel ='linear')
        clf.fit(X_train, y_train) 
        # print(y_train[y_train!=0])
        p = clf.predict(X_test)
        loss = my_custom_loss_func(p,y_test)
        print(mall_id+":"+str(loss))
        lines = mall_id+":"+str(loss)
        write_middle_res(lines)
        joblib.dump(clf, model_dir+mall_id)
        
def write_middle_res(lines):
    with open(pardir+'/data/modeloutput', 'a',encoding = 'utf-8') as f:
        f.writelines(lines+'\n')
        
def train():
    files = listfiles(pca_train_dir)
    length = len(files) 
    newfiles = []
    for file in files:
        mall_id = os.path.basename(file)
        if os.path.exists(model_dir+mall_id):
            continue
        newfiles.append(file)
    length = len(newfiles) 
    try:
        _thread.start_new_thread(create_model_split, (newfiles[0:int(length*0.25)],))
        _thread.start_new_thread(create_model_split, (newfiles[int(length*0.25):int(length*0.5)],))
        _thread.start_new_thread(create_model_split, (newfiles[int(length*0.5):int(length*0.75)],))
        _thread.start_new_thread(create_model_split, (newfiles[int(length*0.75):],))
    except Exception as e: 
        print(e)
        # print ("Error: unable to start thread")
    while 1:
        time.sleep(1000)
        
def predict(test_dir):
    filelist = listfiles(test_dir)
    for file in filelist:
        df = pd.DataFrame()
        mall_id = os.path.basename(file)
        model_path = model_dir+mall_id
        clf = joblib.load(model_path)
        test_data = read_dic(file)
        test_data = np.array(test_data)
        row_ids = test_data[:,-1]
        test = test_data[:,:-1]
        p = clf.predict(test)
        df['row_id'] = [int(t) for t in row_ids]
        p = [int(t) for t in p]
        df['shop_id'] = getlabels_detail(p,label_pkl_dir+mall_id)
        write_record(df,res_path)
        
def append_res_file(respath):
    data = pd.read_csv(evaluate_b_path)
    row_ids = data['row_id']
    with open(respath,mode = 'a',encoding='utf-8') as f:
        for row_id in row_ids:
            lines = str(row_id)+',\n'
            f.writelines(lines)


if __name__=="__main__":
    # write_pca()
    # train()
    predict(pca_test_dir)
    # append_res_file(res_path)
    # train()
    # create_model_split([pca_dir+'m_625'])
    # labels = getlabels_detail([1],label_pkl_dir+'m_625')
    # print(labels)
    # data = read_dic(mall_ta+'m_6167')
    # print(np.shape(data))
    # data = read_dic(pca_dir+'m_4495')
    # print(np.shape(data))
    # y = data[:,-1]
    # i = y[y!=0]
    # labels = getlabels_detail([i[0]],label_pkl_dir+'m_625')
    # print(read_dic(mall_train_dir+'m_625'))
    # write_test_data()
    # get_all_pca([mall_train_dir+'m_3019'])
    # print(read_dic(pca_test_dir+'m_3019'))
    # write_pca()
    
    
        
        
        
        
    