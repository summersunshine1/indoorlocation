import pandas as pd
import numpy as np
from analyze import *
from wight_model import *
import matplotlib.pyplot as plt
from commonLib import *
from getPath import *
pardir = getparentdir()
mall_dir = pardir+'/data/mall/'


def one_record_detail():
    data = pd.read_csv(shop_behavior_path)
    wifi_infos = data['wifi_infos']
    totalcount = 0
    for info in wifi_infos:
        bssids,strengths = process_wifi_info(info)
        dic = {}
        count = 0
        for bssid in bssids:
            if bssid in dic:
                count += 1
                continue
            dic[bssid]=1 
        totalcount+=count
    print(totalcount)

def one_shop_detail():
    data = pd.read_csv(shop_behavior_path)
    shop_ids = data['shop_id']

    shop_id_set = set(list(shop_ids))
    for shop_id in shop_id_set:
        wifi_infos = data['wifi_infos'][data['shop_id']==shop_id]
        dic = {}
        for info in wifi_infos:
            bssids,strengths = process_wifi_info(info)
            for i in range(len(bssids)):
                if not bssids[i] in dic:
                    dic[bssids[i]] = []
                dic[bssids[i]].append(strengths[i])
        print(len(dic))
        for k,v in dic.items():
            if len(v)==1:
                print("rssi"+str(v))
                continue
            v = [float(t) for t in v]
            plt.hist(v,bins = int(np.max(v))-int(np.min(v))+1)
            plt.title(shop_id+" "+k)
            plt.show()
            
def get_class_distribute():
    files = listfiles(mall_dir)
    i = 0
    for file in files:
        data = pd.read_csv(file)
        mall_id = get_mallid_from_mallpath(file)
        wifi_infos = data['wifi_infos']
        dates = data['time_stamp']
        dates = pd.to_datetime(dates)
        bssid_dic ={}
        bssid_count={}
        print(mall_id)
        morning = []
        afternoon = []
        evening = []
        for i in range(len(wifi_infos)):
            info = wifi_infos[i]
            bssids,strengths,connects= process_wifi_info(info)
            bssids = np.array(bssids)
            if dates[i].hour<11:
                morning.append(len(bssids))
            elif dates[i].hour<18:
                afternoon.append(len(bssids))
            else:
                evening.append(len(bssids))
            connects = np.array(connects)
            strengths = np.array(strengths)
            connect_bssid = bssids[connects=='true']
            connect_strength = strengths[connects=='true']
            for i in range(len(connect_bssid)):
                bssid = connect_bssid[i]
                strength = connect_strength[i]
                if not bssid in bssid_dic:
                    bssid_dic[bssid]={}
                    bssid_dic[bssid]['count'] = 0
                    bssid_dic[bssid]['strengths'] = []
                bssid_dic[bssid]['count'] +=1
                bssid_dic[bssid]['strengths'].append(strength)
        # print(bssid_dic)
        print(len(morning))
        print(len(afternoon))
        print(len(evening))
        break
        
def getbssidnum():
    files = listfiles(mall_dir)
    i = 0
    for file in files:
        data = pd.read_csv(file)
        mall_id = get_mallid_from_mallpath(file)
        wifi_infos = data['wifi_infos']
        dates = data['time_stamp']
        dates = pd.to_datetime(dates)
        datedic = {}
        totalbssid = set()
        for i in range(len(wifi_infos)):
            info = wifi_infos[i]
            bssids,strengths,connects= process_wifi_info(info)
            date = dates[i].day
            if not date in datedic:
                datedic[date] = set()
            for b in bssids:
                datedic[date].add(b)
                totalbssid.add(b)  
        bssids = list(datedic.values())
        print(len(bssids[0]&bssids[len(bssids)-1]))
        for i in range(1,len(bssids)):
            print(str(len(bssids[i-1]))+":"+str(len(bssids[i]))+":"+str(len(bssids[i-1]&bssids[i])))
        # print(datedic)
        # print(len(totalbssid))
        break


if __name__=="__main__":
    # one_record_detail()
    # one_shop_detail()    
    # compare_res(pardir+'/data/oldres.csv',pardir+'/data/res.csv')
    getbssidnum()           
        
    
    
    
    