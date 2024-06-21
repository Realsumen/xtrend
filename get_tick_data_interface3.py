#coding: utf8
"""
获取分钟数据的接口--cilckhouse
"""
import datetime as dt
import sys
import pymongo as po
import pandas as pd
import numpy as np
import argparse
import re
from clickhouse_driver import Client as ck_client


coll_save_name = 'stockdata_1Min_'
db_save_name = 'stock_1min_data2_'
DATACOLUMNS = ['price', 'acc_volume', 'acc_turnover', 'high', 'low', 'open', 'pre_close']
BAPRICE_LIST = ['bid_price1', 'bid_volume1', 'ask_price1', 'ask_volume1',
                'bid_price2', 'bid_volume2', 'ask_price2', 'ask_volume2',
                'bid_price3', 'bid_volume3', 'ask_price3', 'ask_volume3',
                'bid_price4', 'bid_volume4', 'ask_price4', 'ask_volume4',
                'bid_price5', 'bid_volume5', 'ask_price5', 'ask_volume5',
                'bid_price6', 'bid_volume6', 'ask_price6', 'ask_volume6',
                'bid_price7', 'bid_volume7', 'ask_price7', 'ask_volume7',
                'bid_price8', 'bid_volume8', 'ask_price8', 'ask_volume8',
                'bid_price9', 'bid_volume9', 'ask_price9', 'ask_volume9',
                'bid_price10', 'bid_volume10', 'ask_price10', 'ask_volume10']
DATACOLUMNS.extend(BAPRICE_LIST)
PRICECOLUMNS = ['price', 'high', 'low', 'open', 'pre_close', 'bid_price1',
                'ask_price1', 'bid_price2', 'ask_price2', 'bid_price3',
                'ask_price3', 'bid_price4', 'ask_price4', 'bid_price5',
                'ask_price5', 'bid_price6', 'ask_price6', 'bid_price7',
                'ask_price7', 'bid_price8', 'ask_price8', 'bid_price9',
                'ask_price9', 'bid_price10', 'ask_price10']
DICT_TYPE_YEAR_LIST = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

host_server2_ip = '172.16.80.27'
host_server1_ip = '192.168.1.111' # '172.16.80.29'
host_local_ip = 'localhost'
is_server_1 = True  # !!本地地址!!!!!!!!!!!!!!!!!!!!!!
if is_server_1:
    host_local = host_server1_ip  # host_local_ip
    host_remote = host_server1_ip
else:
    host_local = host_server1_ip
    host_remote = host_local_ip


def inter_read_1id_1day_clickhouse(year, theid, thedayin, timelist, columns_list):
    if len(timelist) > 1:
        time_sql = 'IN '
        for ii, ele in enumerate(timelist):
            if ii == 0:
                time_sql += '(' + str(ele)
            else:
                time_sql += ',' + str(ele)
        time_sql += ')'
    else:
        time_sql = '= ' + str(timelist[0])
    clickhouse_host_sq = host_remote
    clickhouse_database = 'stock_data_' + str(year)
    day1 = str(thedayin[0])
    day2 = str(thedayin[1])
    begin_time = day1[:4] + '-' + day1[4:6] + '-' + day1[6:]
    end_time = day2[:4] + '-' + day2[4:6] + '-' + day2[6:]
    # client = ck_client(host=clickhouse_host_sq, user=clickhouse_user, database=clickhouse_database, password=clickhouse_pwd)
    client = ck_client(host=clickhouse_host_sq)
    temp_col = ''
    for ele in columns_list:
        temp_col += (', ' + ele)
    api_interface_sql2 = "SELECT TradingDay, InstrumentID, Time" + temp_col + \
                         " FROM {}.test as a WHERE a.TradingDay >= '{}' and a.TradingDay <='{}' and a.InstrumentID = '{}' and a.Time {} order by TradingDay ".format(clickhouse_database, begin_time, end_time, theid, time_sql)
    # print api_interface_sql2
    a = client.execute(api_interface_sql2)
    temp_col_list = ['TradingDay', 'InstrumentID', 'Time']
    temp_col_list.extend(columns_list)
    res_df = pd.DataFrame(a, columns=temp_col_list)
    client.disconnect()
    res_df['TradingDay'] = [dt.datetime.strftime(ele, '%Y%m%d') for ele in list(res_df['TradingDay'])]
    res_df = res_df.sort_values(['TradingDay', 'Time'])
    return res_df


def inter_get_timelist(day, min_num):
    frep_str = str(min_num)+'Min'
    timestart = day + ' 09:30:00'
    timeend = day + ' 11:30:00'
    timelist = pd.date_range(pd.Timestamp(timestart), pd.Timestamp(timeend), freq = frep_str)
    timestart = day + ' 13:00:00'
    timeend = day + ' 15:00:00'
    timelist_pm = pd.date_range(pd.Timestamp(timestart), pd.Timestamp(timeend), freq=frep_str)
    timelist = np.concatenate([timelist, timelist_pm])

    def gettime(datetimein):
        tt = pd.to_datetime(datetimein)
        tmptime = tt.strftime('%H%M%S')
        return int(tmptime)
    # timelist = map(gettime, timelist)
    timelist = [gettime(ele) for ele in timelist]
    return timelist


def inter_read_1id_days(theid, thedayin, timelist, need_columns=['price', 'high', 'low', 'open']):
    theday = thedayin[0]
    theday_end = thedayin[1]
    tempday = str(theday)
    tempday_end = str(theday_end)
    comp = re.compile(r'\d{8}')
    daymatch = comp.match(tempday)
    if (not daymatch) or (len(tempday) != 8):
        print('day input error: %s.' % tempday)
        return None
    year = tempday[0:4]
    year_end = tempday_end[0:4]
    if year_end != year:
        print('year not same: %s, %s.' % (year, year_end))
    res_df = inter_read_1id_1day_clickhouse(year, theid, thedayin, timelist, need_columns)
    return res_df


def inter_get_tick_data_byinter(theid, thedayin, time_list, merge, mod, need_col=None):
    # mod: 0 原始值；1 后复权；2前复权
    if isinstance(thedayin, list):
        # 按照时间列表==========
        temp_day_list = thedayin[:]
        temp_day_list.sort()
        theday_list = [temp_day_list[0], temp_day_list[-1]]
    elif isinstance(thedayin, tuple):
        # 按照元组=============
        # daylist = pd.date_range(str(thedayin[0]), str(thedayin[1]), freq='D')
        temp_day_list = []
        theday_list = [thedayin[0], thedayin[1]]
    else:
        print('dayin type wrong.')
        return None
    res_temp_list = []
    all_year_list = splite_year(theday_list)
    for year_cut in all_year_list:
        # print year_cut
        if need_col is None:
            res = inter_read_1id_days(theid, year_cut, time_list)
        else:
            res = inter_read_1id_days(theid, year_cut, time_list, need_col)
        # print res
        if res is not None:
            res_temp_list.append(res)
    # print 'year num:', len(res_temp_list)
    if len(res_temp_list) <= 0:
        return None
    res = pd.concat(res_temp_list, axis=0)
    res = res.sort_values(['TradingDay', 'Time'])
    # print res
    if mod == 0:  # 原始值
        ratio_res = res
    elif mod == 1:  # 后复权
        ratio_df = backwardRatio(theid)
        ratio_res = process_backward_ratio(res, ratio_df)
    elif mod == 2:  # 前复权
        ratio_df = backwardRatio(theid)
        ratio_res = process_forward_ratio(res, ratio_df)
    else:
        print('backwardRatio mod wrong.%d'% mod)
        return None
    if merge:
        return ratio_res
    else:
        res_df_list = []
        id_days = ratio_res.groupby('TradingDay')
        for day, day_data in id_days:
            if isinstance(thedayin, list):
                if int(day) in temp_day_list:
                    res_df_list.append(day_data)
            else:
                res_df_list.append(day_data)
        return res_df_list


def splite_year(thedayin):
    # 将时间段按照年拆分成分割的时间段
    theday_1 = thedayin[0]
    theday_2 = thedayin[1]
    daylist = pd.date_range(str(theday_1), str(theday_2), freq='D')
    all_year_dict = []
    for ii, ele in enumerate(daylist):
        if ii == 0:
            start = ele
            end = ele
        if ele.year != end.year:
            start_int = int(dt.datetime.strftime(start, '%Y%m%d'))
            end_int = int(dt.datetime.strftime(end, '%Y%m%d'))
            all_year_dict.append([start_int, end_int])
            start = ele
            end = ele
        else:
            end = ele
    start_int = int(dt.datetime.strftime(start, '%Y%m%d'))
    end_int = int(dt.datetime.strftime(end, '%Y%m%d'))
    all_year_dict.append([start_int, end_int])
    return all_year_dict


def backwardRatio(theid):
    client1 = po.MongoClient(host_local, 39017)
    db1 = client1['admin']
    db1.authenticate("reader", "123456")
    db1 = client1['stocks_data']
    coll1 = db1['BackwardRatio']
    res = coll1.find({"SecurityID": theid})
    res_list = list(res)
    client1.close()
    if len(res_list) <= 0:
        # print('backwardRatio not find data. %s' % theid)
        return None
    df = pd.DataFrame(res_list)
    df.sort_values('Date', inplace=True)
    return df


def get_all_backwardRatio():  # 获取所有复权信息，用于配合获取全市场价格
    client1 = po.MongoClient(host_local, 39017)
    db1 = client1['admin']
    db1.authenticate("reader", "123456")
    db1 = client1['stocks_data']
    coll1 = db1['BackwardRatio']
    res = coll1.find()
    res_list = list(res)
    client1.close()
    if len(res_list) <= 0:
        # print('backwardRatio not find data. %s' % theid)
        return None
    df = pd.DataFrame(res_list)
    del df['_id']
    return df


def process_backward_ratio(res, ratio_df):
    if ratio_df is None:
        # print('has no ratio data')
        return res
    the_column_list = list(res.columns)
    tradingday_list = np.array(list(res['TradingDay']))
    all_len = len(tradingday_list)
    tt_day = np.array(list(ratio_df['Date']))
    tt_ratio = np.array(list(ratio_df['Ratio']))
    tt_num = np.searchsorted(tt_day, tradingday_list, side='right')
    ratio_list = np.zeros(all_len)
    for ii, ee in enumerate(tt_num):
        if ee < 1:
            ratio_list[ii] = 1.0
        else:
            ratio_list[ii] = tt_ratio[ee-1]
    for ele in PRICECOLUMNS:
        if ele in the_column_list:
            temp_a_list = list(res[ele])
            temp_a_np = np.array(temp_a_list)
            temp = temp_a_np * ratio_list
            res[ele] = temp
    return res


def process_forward_ratio(res, ratio_df):
    if ratio_df is None:
        print('has no ratio data')
        return res
    the_column_list = list(res.columns)
    # 原始数据日期
    tradingday_list = np.array(list(res['TradingDay']))
    all_len = len(tradingday_list)
    # 比例数据
    tt_day = np.array(list(ratio_df['Date']))
    tt_ratio = np.array(list(ratio_df['Ratio']))

    tt_num = np.searchsorted(tt_day, tradingday_list, side='right')
    # print tt_num
    ratio_list = np.zeros(all_len)
    for ii, ee in enumerate(tt_num):
        if ee < 1:
            ratio_list[ii] = 1.0
        else:
            ratio_list[ii] = tt_ratio[ee-1]
    # 前复权===========================
    ratio_list = ratio_list / float(ratio_list[-1])
    # print ratio_list
    for ele in PRICECOLUMNS:
        if ele in the_column_list:
            temp_a_list = list(res[ele])
            temp_a_np = np.array(temp_a_list)
            temp = temp_a_np * ratio_list
            res[ele] = temp
    return res


def get_tick_data_bytime(theid, thedayin, time, merge=True, mod_num=0, need_col=None):
    # 处理时间点，获取时间===================
    time_in = int(time)
    time_list_all = inter_get_timelist('20160420', 1)
    if time_in not in time_list_all:
        print('Input time is not in date list. %s' % time)
        return None
    time_list = []
    time_list.append(time_in)
    res = inter_get_tick_data_byinter(theid, thedayin, time_list, merge, mod_num, need_col)
    return res


def get_tick_data_byfreq(theid, thedayin, freq, merge=True, mod_num=0, need_col=None):
    # 处理频率，获取时间===================
    freq_str = str(freq)
    if freq_str.endswith('min'):
        temp = freq_str[:-3]
        freq_num = int(temp)
    elif freq_str.endswith('h'):
        temp = freq_str[:-1]
        freq_num = int(temp)*60
    else:
        print('freq input wrong. %s' % freq)
        return None
    time_list = inter_get_timelist('20160420', freq_num)
    res = inter_get_tick_data_byinter(theid, thedayin, time_list, merge, mod_num, need_col)
    return res



def main(argv=None):
    # ===============================
    dayin = (20190213, 20190713)
    print(type(dayin))
    theid = '002807.SZ' # '601113.SH'
    print('========================================')
    # mod: 0 原始值；1 后复权；2前复权
    res = get_tick_data_byfreq(theid, dayin, '1min', True, mod_num=0)


    print(res)
    thedayin = (20190213, 20190713)
    res2 = get_tick_data_bytime(theid, thedayin, 150000, merge=True, mod_num=0)


    #with open('./dict_data.pickle', 'w') as p:
    #    pickle.dump(res, p)
    print('========================================')

    print('ok')


if __name__ == "__main__":
    sys.exit(main())
