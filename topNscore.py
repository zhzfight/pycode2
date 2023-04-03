# encoding: utf-8
import os
import sys
import pandas as pd
import numpy as np
from numpy import array
import time
import calendar
import math
from progress.bar import Bar
import datetime
import holidays
from utils import get_time_slot,get_words,get_timestamp,get_region
poi_dim = 128
N = 10
target_ratio = 0.1  # 第25天权重降至比例
delta_time_weight = 1 * 24 * 3600

delta_t = 1 * 24 * 3600
sample_num = 300
test_data = pd.read_csv('./test_data.txt', sep='\t').sample(n=sample_num, random_state=88, axis=0)
train_data = pd.read_csv('./train_data.txt',  sep='\t')

poi_vec_data = pd.read_csv('./net_POI_vec.txt', header=None, sep='\s+')
time_vec_data = pd.read_csv('./net_time_vec.txt', header=None, sep='\s+')
region_vec_data = pd.read_csv('./net_reg_vec.txt', header=None, sep='\s+')
user_vec_data = pd.read_csv('./net_user_vec.txt', header=None, sep='\s+')

poi_vec_dict = {}
time_vec_dict = {}
region_vec_dict = {}
user_vec_dict = {}
poi_cat_dict={}


def write_in_file(f_out_path, str_out):
    f_out = open(f_out_path, 'w+')
    f_out.write(str_out)
    f_out.close()


def dicts_gen():
    print('dicts_gen')
    # poi_vec_dict
    #print(train_data.columns)
    for index, row in poi_vec_data.iterrows():
        if row[0] not in poi_vec_dict:
            poi_vec_dict[row[0]] = np.array(row[1:])
        else:
            print('poi_vec_dict重复:', row[0])
    # time_vec_dict
    for index, row in time_vec_data.iterrows():
        if row[0] not in time_vec_dict:
            time_vec_dict[row[0]] = np.array(row[1:])
        else:
            print('time_vec_dict重复:', row[0])
    # region_vec_dict
    for index, row in region_vec_data.iterrows():
        if row[0] not in region_vec_dict:
            region_vec_dict[row[0]] = np.array(row[1:])
        else:
            print('region_vec_dict重复:', row[0])
    for index, row in user_vec_data.iterrows():
        if row[0] not in user_vec_dict:
            user_vec_dict[row[0]] = np.array(row[1:])
        else:
            print('user_vec_dict重复:', row[0])

    pc_df=train_data.drop_duplicates(['VenueId'])
    for index,row in pc_df.iterrows():
        poi_cat_dict[row['VenueId']]=get_words(row['VenueCategory'])

def cur_preference_vec_gen():
    print('cur_pregerence_vec gen')
    cur_preference_vec = pd.DataFrame(
        columns=['userID', 'VenueId', 'Time_Slot', 'Time', 'Region','VenueCategory'] + [x for x in range(poi_dim)])
    cur_preference_vec['userID'] = test_data['userID']
    cur_preference_vec['VenueId'] = test_data['VenueId']
    cur_preference_vec['Time_Slot'] = test_data['Time(GMT)'].apply(get_time_slot)
    cur_preference_vec['Time'] = test_data['Time(GMT)'].apply(get_timestamp)
    cur_preference_vec['Region'] = test_data['VenueLocation'].apply(get_region)
    cur_preference_vec['VenueCategory']=test_data['VenueCategory']

    train_data['Time'] = train_data['Time(GMT)'].apply(get_timestamp)
    bar = Bar('user_vec_gen', max=sample_num)
    for index, row in cur_preference_vec.iterrows():
        vec = np.zeros(poi_dim)
        pois_time = train_data[(train_data['userID'] == row['userID']) & (row['Time'] - train_data['Time'] < delta_t) & (
                    row['Time'] - train_data['Time'] > 0)][['VenueId', 'Time']]
        # 方法1：除以delta_time_weight
        if len(pois_time) == 0:
            bar.next()
            continue
        p_vs=[]
        timeIntervals=[]
        for index_, row_ in pois_time.iterrows():
            p_vs.append( poi_vec_data[poi_vec_data[0] == row_['VenueId']][poi_vec_data.columns[1:]].iloc[0])
            timeIntervals.append(row['Time']-row_['Time'])
            # vec = vec + math.exp(-(row['Time'] - row_['Time']) / delta_time_weight) * p_v
        total_timeIntervals=sum(timeIntervals)
        timeIntervals_ratio=[math.exp(-(x+1)/total_timeIntervals) for x in timeIntervals]
        for i in range(len(p_vs)):
            vec+=(timeIntervals_ratio[i]/sum(timeIntervals_ratio))*np.array(p_vs[i])
        cur_preference_vec.loc[index, [x for x in range(poi_dim)]] = vec
        bar.next()
    bar.finish()
    cur_preference_vec.to_csv('./cur_preference_vec.txt', sep='\t')


def topNscore():
    print('topNscore')
    print(len(poi_vec_dict))
    hit_num = 0
    sample_num = 0
    cur_preference_vec = pd.read_csv('./cur_preference_vec.txt', sep='\t', index_col=0)


    pt_df=pd.read_csv('./net_POI_time.txt',sep='\t',header=None)
    pr_df=pd.read_csv('./net_POI_reg.txt',sep='\t',header=None)
    pt_df.columns=['poi','t','tf']
    pr_df.columns=['poi','r','tf']
    c1=0
    c2=0
    c3=0
    c4=0
    c5=0
    c6=0
    for index, row in cur_preference_vec.iterrows():

        cp_vec = np.array(row[6:])
        if np.isnan(cp_vec[6]):
            continue
        userID = row['userID']
        poiID = row['VenueId']
        time_slot = row['Time_Slot']
        region = row['Region']
        word=row['VenueCategory']
        if region not in region_vec_dict:
            c1+=1
            continue
        if userID not in user_vec_dict:
            c2+=1
            continue
        if poiID not in poi_vec_dict:
            c3+=1
            continue
        words=poi_cat_dict[poiID]
        if str(time_slot)+'_'+words not in time_vec_dict:
            c4+=1
            continue

        sample_num += 1



        # # poiID_vec = np.array(poi_vec_data[poi_vec_data[0]==poiID][poi_vec_data.columns[1:]].iloc[0])
        # time_slot_vec = np.array(time_vec_data[time_vec_data[0]==int(time_slot)][time_vec_data.columns[1:]].iloc[0])
        # region_vec = np.array(region_vec_data[region_vec_data[0]==region][region_vec_data.columns[1:]].iloc[0])
        region_vec = region_vec_dict[region]

        user_vec = user_vec_dict[userID]
        topN = [('', -sys.maxsize - 1,0.0,0.0,0.0,0.0,0.0,0.0)] * N
        # for index_, row_ in poi_vec_data.iterrows():

        for key, value in poi_vec_dict.items():
            # poi_vec = np.array(row_[1:])
            #score = (e_x[0]*cp_vec+e_x[1]*region_vec+e_x[2]*time_slot_vec).dot(value)
            #score=cosine_similarity(cp_vec,value)+cosine_similarity(time_slot_vec,value)+cosine_similarity(region_vec,value)
            # score = a1 * cp_vec.dot(value) + a2 * time_slot_vec.dot(value) + a3 * region_vec.dot(value)
            if key not in poi_cat_dict:
                c5+=1
                continue

            words = poi_cat_dict[key]
            score=''
            if str(time_slot) + '_' + words not in time_vec_dict:
                c6+=1
                continue

            time_slot_vec = time_vec_dict[str(time_slot) + '_' + words]
            score = cp_vec.dot(value) + time_slot_vec.dot(value)+region_vec.dot(value)
            if key == poiID:
                print('target ', cosine_similarity(cp_vec, value),cp_vec.dot(value), cosine_similarity(time_slot_vec, value),time_slot_vec.dot(value),
                      cosine_similarity(region_vec, value),region_vec.dot(value))
                print('user ',cosine_similarity(cp_vec, user_vec), cosine_similarity(time_slot_vec, user_vec),
                      cosine_similarity(region_vec, user_vec))
                print(str(time_slot) + '_' + words ,'\n',pt_df[pt_df['poi']==key])
                print(region,'\n',pr_df[pr_df['poi']==key])
            topN.sort(key=lambda x: x[1])
            if topN[0][1] < score:
                topN[0] = (key, score,cosine_similarity(cp_vec, value),cp_vec.dot(value),cosine_similarity(time_slot_vec, value),time_slot_vec.dot(value),cosine_similarity(region_vec, value),region_vec.dot(value))
        topN_pois = [x[0] for x in topN]
        print('threshold ',topN[0][2],topN[0][3],topN[0][4],topN[0][5],topN[0][6],topN[0][7])
        if poiID in topN_pois:
            hit_num += 1
        print('第', sample_num, '个样本', hit_num, sample_num, '{:.3}'.format(hit_num / sample_num))
    print(c1,c2,c3,c4,c5,c6)


def cosine_similarity(x, y):
    # x and y are numpy vectors of the same length
    dot_product = np.dot(x, y)  # calculate the dot product
    norm_x = np.linalg.norm(x)  # calculate the norm of x
    norm_y = np.linalg.norm(y)  # calculate the norm of y
    return dot_product / (norm_x * norm_y)  # return the cosine similarity


if __name__ == '__main__':
    dicts_gen()
    cur_preference_vec_gen()
    topNscore()
