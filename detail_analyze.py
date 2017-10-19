import pandas as pd
import numpy as np
from analyze import *
from wight_model import *
import matplotlib.pyplot as plt

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
            

if __name__=="__main__":
    # one_record_detail()
    one_shop_detail()    
                
        
    
    
    
    