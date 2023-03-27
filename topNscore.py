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

poi_dim = 128
N = 10
target_ratio = 0.1  # 第25天权重降至比例
delta_time_weight = 25 * 24 * 3600

delta_t = 25 * 24 * 3600
sample_num = 300
test_data = pd.read_csv('./test_data.txt', header=None, sep='\t').sample(n=sample_num, random_state=88, axis=0)
train_data = pd.read_csv('./train_data.txt', header=None, sep='\t')

poi_vec_data = pd.read_csv('./net_POI_vec.txt', header=None, sep='\s+')
time_vec_data = pd.read_csv('./net_time_vec.txt', header=None, sep='\s+')
region_vec_data = pd.read_csv('./net_reg_vec.txt', header=None, sep='\s+')
user_vec_data = pd.read_csv('./net_user_vec.txt', header=None, sep='\s+')

poi_vec_dict = {}
time_vec_dict = {}
region_vec_dict = {}
user_vec_dict = {}

month_dict = dict((v, k) for k, v in enumerate(calendar.month_abbr))


def get_time_slot(time_str):
    return time_str[11:13]


def get_timestamp(time_read):
    week = time_read[0:3]
    month = time_read[4:7]
    day = time_read[8:10]
    hour = time_read[11:13]
    minute = time_read[14:16]
    second = time_read[17:19]
    year = time_read[-4:]
    try:
        format_time = year + '-' + str(month_dict[month]) + '-' + day + ' ' + hour + ':' + minute + ':' + second
        ts = time.strptime(format_time, "%Y-%m-%d %H:%M:%S")
        time_ = time.mktime(ts)
        print_str = str(int(time_)) + '\t' + str(hour) + '\n'
        return time_
    except:
        print(time_read)


def get_region(loc_str):
    location = loc_str[1:-1]
    locations = location.split(',')
    state, country = locations[3], locations[4]
    if len(locations) is 5:
        city = locations[2]
    else:  # len(locations) is 6:
        city = locations[-4] + locations[-3]
    region = ''.join(city.split(' ')) + '_' + ''.join(state.split(' ')) + '_' + ''.join(country.split(' '))
    return region


def write_in_file(f_out_path, str_out):
    f_out = open(f_out_path, 'w+')
    f_out.write(str_out)
    f_out.close()


def dicts_gen():
    print('dicts_gen')
    # poi_vec_dict
    for index, row in poi_vec_data.iterrows():
        if row[0] not in poi_vec_dict:
            poi_vec_dict[row[0]] = np.array(row[1:])
        else:
            print('poi_vec_dict重复:', row[0])
    # time_vec_dict
    for index, row in time_vec_data.iterrows():
        if int(row[0]) not in time_vec_dict:
            time_vec_dict[int(row[0])] = np.array(row[1:])
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


def cur_preference_vec_gen():
    print('cur_pregerence_vec gen')
    cur_preference_vec = pd.DataFrame(
        columns=['userID', 'VenueId', 'Time_Slot', 'Time', 'Region'] + [x for x in range(poi_dim)])
    cur_preference_vec['userID'] = test_data[1]
    cur_preference_vec['VenueId'] = test_data[3]
    cur_preference_vec['Time_Slot'] = test_data[2].apply(get_time_slot)
    cur_preference_vec['Time'] = test_data[2].apply(get_timestamp)
    cur_preference_vec['Region'] = test_data[5].apply(get_region)

    train_data['Time'] = train_data[2].apply(get_timestamp)
    bar = Bar('user_vec_gen', max=sample_num)
    for index, row in cur_preference_vec.iterrows():
        vec = np.zeros(poi_dim)
        pois_time = train_data[(train_data[1] == row['userID']) & (row['Time'] - train_data['Time'] < delta_t) & (
                    row['Time'] - train_data['Time'] > 0)][[3, 'Time']]
        # 方法1：除以delta_time_weight
        if len(pois_time) == 0:
            bar.next()
            continue
        for index_, row_ in pois_time.iterrows():
            p_v = np.array(poi_vec_data[poi_vec_data[0] == row_[3]][poi_vec_data.columns[1:]].iloc[0])
            # vec = vec + math.exp(-(row['Time'] - row_['Time']) / delta_time_weight) * p_v
            vec = vec + math.exp(-(row['Time']-row_['Time'])/delta_time_weight) * p_v
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
    bar = Bar('eval', max=len(cur_preference_vec))
    for index, row in cur_preference_vec.iterrows():

        cp_vec = np.array(row[5:])
        if np.isnan(cp_vec[5]):
            bar.next()
            continue
        userID = row['userID']
        poiID = row['VenueId']
        time_slot = row['Time_Slot']
        region = row['Region']
        if region not in region_vec_dict:
            bar.next()
            continue
        sample_num += 1



        # # poiID_vec = np.array(poi_vec_data[poi_vec_data[0]==poiID][poi_vec_data.columns[1:]].iloc[0])
        # time_slot_vec = np.array(time_vec_data[time_vec_data[0]==int(time_slot)][time_vec_data.columns[1:]].iloc[0])
        # region_vec = np.array(region_vec_data[region_vec_data[0]==region][region_vec_data.columns[1:]].iloc[0])
        time_slot_vec = time_vec_dict[int(time_slot)]
        region_vec = region_vec_dict[region]
        user_vec = user_vec_dict[userID]
        topN = [('', -sys.maxsize - 1)] * N
        # for index_, row_ in poi_vec_data.iterrows():
        a1 = cosine_similarity(cp_vec, user_vec)
        a2 = cosine_similarity(time_slot_vec, user_vec)
        a3 = cosine_similarity(region_vec, user_vec)
        e_x = np.exp([a1, a2, a3])
        e_x = e_x / np.sum(e_x)
        #print(a1, a2, a3)
        # print(e_x[0], e_x[1], e_x[2])
        for key, value in poi_vec_dict.items():
            # poi_vec = np.array(row_[1:])
            # score = (e_x[0]*cp_vec+e_x[1]*region_vec+e_x[2]*time_slot_vec).dot(value)
            # score=a1*cosine_similarity(cp_vec,value)+a2*cosine_similarity(time_slot_vec,value)+a3*cosine_similarity(region_vec,value)
            # score = a1 * cp_vec.dot(value) + a2 * time_slot_vec.dot(value) + a3 * region_vec.dot(value)
            score = ( cp_vec + region_vec +  time_slot_vec).dot(value)
            topN.sort(key=lambda x: x[1])
            if topN[0][1] < score:
                topN[0] = (key, score)
        topN_pois = [x[0] for x in topN]
        if poiID in topN_pois:
            hit_num += 1
        print('第', sample_num, '个样本', hit_num, sample_num, '{:.3}'.format(hit_num / sample_num))
        bar.next()
    bar.finish()


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
