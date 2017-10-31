import pandas as pd
import numpy as np
from analyze import *
from wight_model import *
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap,cm

from commonLib import *
from getPath import *
pardir = getparentdir()
mall_dir = pardir+'/data/mall/'
mall_lon_dis_train_dir = pardir+'/data/mall_lon_dis_train/'
shop_info_dir = pardir+'/data/shopinfo/'
mall_wifi_dic_dir = pardir+'/data/mall_wifi_dic/'
mall_wifi_dic_remove_dir = pardir+'/data/mall_wifi_dic_remove/'

def one_record_detail(shop_behavior_path):
    data = pd.read_csv(shop_behavior_path)
    mall_id = get_mallid_from_mallpath(shop_behavior_path)
    wifi_infos = data['wifi_infos']
    totalcount = 0
    bss = set()
    for info in wifi_infos:
        bssids,strengths ,connects= process_wifi_info(info)
        dic = {}
        count = 0

        for bssid in bssids:
            if bssid in dic:
                bss.add(bssid)
                count += 1
                continue
            dic[bssid]=1 
        totalcount+=count
    print(mall_id)
    print(bss)
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
        
def getinfo():
    file = mall_dir+'m_6803.csv'
    data = pd.read_csv(file)
    df = pd.DataFrame({'count':data.groupby(['shop_id']).size()}).reset_index()
    l = len(df['shop_id'][df['count']<3])
    print(df['shop_id'][df['count']<3])
    # l = len(a[a==1])
    print(l/len(df))
    
# def plotmap():
    # filelist = listfiles(shop_info_dir)
    # for file in filelist:
        # data = pd.read_csv(file)
        # mall_id = get_mallid_from_mallpath(file)
        # lons = data['longitude']
        # lats = data['latitude']
        # map = Basemap(llcrnrlon=100,llcrnrlat=20,urcrnrlon=130,urcrnrlat=50,projection='merc',resolution='l')
        # map.drawcoastlines()   
        # map.drawcountries()    
        # map.drawmapboundary()
        # x, y = map(lons, lats)
        # cm = plt.cm.get_cmap('RdYlBu')
        # sc = map.scatter(x,y,cmap=cm)
        # plt.colorbar(sc)
        # plt.title(mall_id)
        # plt.show()

if __name__=="__main__":
    # one_record_detail()
    # one_shop_detail()    
    # compare_res(pardir+'/data/oldres.csv',pardir+'/data/res.csv')
    # getbssidnum()  
    # getinfo()
    # plotmap()
    mallshop_dic_add_conect_dir_remove = pardir+'/data/mall_shop_add_connect_remove/'
    # print(read_dic(mall_wifi_dic_dir+'m_6803'))
    dic = read_dic(mallshop_dic_add_conect_dir_remove+'m_4079')
    list1 = set(dic['s_181894'].keys())
    list2 = set(dic['s_181895'].keys())
    print(list1-list2)
    print(list2-list1)
    
    # filelist = listfiles(mall_dir)
    # for file in filelist:
        # one_record_detail(file)
    
    
    
    