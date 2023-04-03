import datetime
import holidays
import re
from geopy.geocoders import Nominatim
import time
import calendar
geolocator = Nominatim(user_agent="zhz's pycode")
us_ca_holidays = holidays.US(state='CA')
def get_time_slot(time_str):
    date = datetime.datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
    hour=date.hour
    # 创建美国CA节假日对象

    res = 0
    if hour < 5:
        res = 0
    elif hour < 7:
        res = 1
    elif hour<9:
        res=2
    elif hour < 11:
        res = 3
    elif hour < 13:
        res = 4
    elif hour<15:
        res=5
    elif hour < 16:
        res= 6
    elif hour < 17:
        res= 7
    elif hour < 19:
        res= 8
    elif hour<21:
        res=9
    elif hour<23:
        res=10
    else:
        res=0
    if date.date() in us_ca_holidays:
        res+=11
    return res

def get_words(words_str):

    words = re.split(r",", words_str[1:-1])  # 去掉首尾的方括号，然后用逗号或&分割
    words=[x.replace(' ', '') for x in words]
    words = [word for word in words if word != '']
    unique_words=set(words)
    return '_'.join(unique_words)
'''
    words = words_str[1:-1].split(',')[:-1]
    return '_'.join([x.replace(' ', '') for x in words])
    '''
month_dict = dict((v, k) for k, v in enumerate(calendar.month_abbr))
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
    else: # len(locations) is 6:
        city = locations[-4] + locations[-3]
    region = ''.join(city.split(' ')) + '_' + ''.join(state.split(' ')) + '_' + ''.join(country.split(' '))
    return region
