import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from commonLib import *

malldir = pardir+'/data/mall/'
 
def get_mall_info(file):
    data = pd.read_csv(file)
    mall_id = os.path.basename(file)[:-4]
    shop_ids = data['shop_id']
    wifi_infos = data['wifi_infos']
    print(len(wifi_infos))
    d = pd.DataFrame(data.groupby(['user_id'])['wifi_infos'].count().reset_index())
    d1 = pd.DataFrame(data.groupby(['user_id','shop_id'])['wifi_infos'].count().reset_index())
    res = pd.merge(d,d1,on='user_id')
    d = res.groupby(['user_id']).count()
    da = d['wifi_infos_x'][d['wifi_infos_x']==1]
    print(len(da)/len(d))
    
    
    # print(d)
    # plt.hist(d[d==1])
    # plt.show()
    # length = len(shop_ids)
    # shop_wifi_dic = {}
    # for i in range(length):
        # shop_id = shop_ids[i]
        # wifi_info = wifi_infos[i]
        # bssids,strengths = process_wifi_info(wifi_info)
       
if __name__=="__main__":
    malls = ['m_7800','m_4187','m_7168','m_1409','m_623']
    files = []
    for i in range(len(malls)):
        file = malldir+malls[i]+'.csv'
        files.append(file)
        get_mall_info(file)