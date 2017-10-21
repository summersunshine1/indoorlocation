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
pardir = getparentdir()

mallshop_dic_dir = pardir+'/data/mall_shop_dic/'
mall_bssid_dic_path = pardir +'/data/mall_choose_bssid_dic'
newlabels_dir = pardir+'/data/newlabels/'
newtrain_dir = pardir+'/data/newtrain/'
mall_dir = pardir+'/data/mall/'
model_dir = pardir+'/data/model/'
test_mall_dir = pardir+'/data/rawtestmall/'
label_dir = pardir+'/data/labels/'
res_path = pardir+'/data/res/rf.csv'
evaluate_b_path = pardir+'/data/evaluation_b.csv'
middle_path = pardir+'/data/modeloutput'

def get_train(filelist,mall_bssid_dic):
    print(len(filelist))
    if os.path.exists(middle_path):
        os.remove(middle_path)
    for file in filelist:
        mall_id = get_mallid_from_mallpath(file)
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        shop_ids = data['shop_id']
        labels = convertLabels(shop_ids,newlabels_dir+mall_id)
        labels = [[t] for t in labels]
        mallbssids = mall_bssid_dic[mall_id]
        bssid_dic = get_bssid_dic(mallbssids)
        train_x = []
        for wifi_info in wifi_infos:
            bssid_strength_dic = {}
            train_d = np.array([0]*len(mallbssids))
            bssids,strengths = process_wifi_info(wifi_info)
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
                strength = np.mean(v1)
                train_d[index] = strength
            train_x.append(train_d)
        data = np.hstack((train_x,labels))
        write_dic(data,newtrain_dir+mall_id)
        # print(data)
        X_train, X_test, y_train, y_test = train_test_split(train_x, labels, test_size=0.2, random_state=42)
        # clf = SVC(kernel ='linear')
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train, y_train) 
        # list1 = list(clf.feature_importances_)
        # list1 = sorted(list1)
        # print(list1[-10:])
        p = clf.predict(X_test)
        clf.fit(train_x, labels)
        joblib.dump(clf, model_dir+mall_id)
        y_test = [t[0] for t in y_test]
        loss = my_custom_loss_func(y_test,p)
        print(mall_id+":"+str(loss))
        lines = mall_id+":"+str(loss)
        write_middle_res(lines)
        
def predict(filelist,mall_bssid_dic):
    res_dic = {}
    if os.path.exists(res_path):
        os.remove(res_path)
    for file in filelist:
        mall_id = get_mallid_from_mallpath(file)
        data = pd.read_csv(file)
        wifi_infos = data['wifi_infos']
        row_ids = data['row_id']
        print(len(row_ids))
        mallbssids = mall_bssid_dic[mall_id]
        bssid_dic = get_bssid_dic(mallbssids)
        train_x = []
        for wifi_info in wifi_infos:
            bssid_strength_dic = {}
            train_d = np.array([0]*len(mallbssids))
            bssids,strengths = process_wifi_info(wifi_info)
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
                strength = np.mean(v1)
                train_d[index] = strength
            train_x.append(train_d)
        clf = joblib.load(model_dir+mall_id)
        p = clf.predict(train_x)
        labels = getlabels_detail(p,label_dir+mall_id)
        df = pd.DataFrame()
        df['row_id'] = row_ids
        df['shop_id'] = labels
        print(len(df))
        write_record(df,res_path)
    append_res_file(res_path,evaluate_b_path)   
    
        
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
    mall_bssid_dic = read_dic(mall_bssid_dic_path)
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
    # get_train(files,mall_bssid_dic)
    files = listfiles(test_mall_dir)
    # print(len(files))
    predict(files,mall_bssid_dic)

        
                

    