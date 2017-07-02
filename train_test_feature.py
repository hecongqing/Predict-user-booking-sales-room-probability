# encoding=utf8
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn import preprocessing

import re
import numpy as np


def del_str_data(key):
    return int(re.search("\d+", key).group())


def data_deal(actions):
    for key in ["orderid", "uid", "hotelid", "basicroomid", "roomid",
                "orderid_lastord", "hotelid_lastord", "roomid_lastord", "basicroomid_lastord"]:
        actions[key] = actions[key].fillna("0")
        actions[key] = actions[key].apply(del_str_data)
    return actions


def last_now_data_compare_basic(actions):
    dt1 = pd.to_datetime(actions["orderdate"])
    dt2 = pd.to_datetime(actions["orderdate_lastord"])
    actions['orderdate_interval_week'] = dt1 - dt2
    actions['orderdate_interval_week'] = actions['orderdate_interval_week'].apply(lambda x: x.days)
    actions["orderdate_week"] = dt1.dt.weekday
    actions["orderdate_lastord_week"] = dt2.dt.weekday

    actions["hotelid_eq"] = actions["hotelid"] - actions["hotelid_lastord"]
    actions["hotelid_eq"] = actions["hotelid_eq"].map(lambda x: int(x == 0))


    actions["roomid_eq"] = actions["roomid"] - actions["roomid_lastord"]
    actions["roomid_eq"] = actions["roomid_eq"].map(lambda x: int(x == 0))

    actions["basicroomid_eq"] = actions["basicroomid"] - actions["basicroomid_lastord"]
    actions["basicroomid_eq"] = actions["basicroomid_eq"].map(lambda x: int(x == 0))


    actions["rank_eq"] = actions["rank"] - actions["rank_lastord"]
    actions["rank_eq"] = actions["rank_eq"].map(lambda x: int(x == 0))


    actions["star_eq"] = actions["star"] - actions["star_lastord"]
    actions["star_eq"] = actions["star_eq"].map(lambda x: int(x == 0))


    for i in [2, 3, 4, 5, 6, 8]:
        actions["roomservice_eq_" + str(i)] = actions["roomservice_" + str(i)] - actions[
            "roomservice_" + str(i) + "_lastord"]
        actions["roomservice_eq_" + str(i)] = actions["roomservice_eq_" + str(i)].map(lambda x: int(x == 0))
        del actions["roomservice_" + str(i) + "_lastord"]


    for i in [2, 3, 4, 5, 6]:
        actions["roomtag_eq_" + str(i)] = actions["roomtag_" + str(i)] - actions["roomtag_" + str(i) + "_lastord"]
        actions["roomtag_eq_" + str(i)] = actions["roomtag_eq_" + str(i)].map(lambda x: int(x == 0))
        del actions["roomtag_" + str(i) + "_lastord"]
    return actions


def last_avg_now_data_compare_data(actions):
    actions["price_deduct_rate_with_last"] = actions["price_deduct"] / actions["price_last_lastord"]
    actions["returnvalue_rate_with_last"] = actions["returnvalue"] / actions["return_lastord"]

    actions["price_deduct_rate_with_avg"] = actions["price_deduct"] / actions["user_avgdealprice"]
    actions["returnvalue_rate_with_avg"] = actions["returnvalue"] / actions["user_avgpromotion"]
    return actions


def last_price_rate(actions):
    actions["price_last_lastord_min_rate_hotel"] = actions["price_last_lastord"] / actions["hotel_minprice_lastord"]
    actions["price_last_lastord_min_rate_basic"] = actions["price_last_lastord"] / actions["basic_minprice_lastord"]
    actions['return_lastord_rate'] = actions['return_lastord'] / actions['price_last_lastord']
    return actions


def now_price_rate(actions):
    hotel_now_min_price = actions[['orderdate', 'hotelid', 'price_deduct']].groupby(['orderdate', 'hotelid'],
                                                                                    as_index=False).min()
    hotel_now_min_price.columns = ['orderdate', 'hotelid', 'hotel_now_min_price']
    hotel_basicroomid_now_min_price = actions[['orderdate', 'hotelid', 'basicroomid', 'price_deduct']].groupby(
        ['orderdate', 'hotelid', 'basicroomid'], as_index=False).min()
    hotel_basicroomid_now_min_price.columns = ['orderdate', 'hotelid', 'basicroomid', 'hotel_basicroomid_now_min_price']
    actions = pd.merge(actions, hotel_now_min_price, on=['orderdate', 'hotelid'], how='left')
    actions = pd.merge(actions, hotel_basicroomid_now_min_price, on=['orderdate', 'hotelid', 'basicroomid'], how='left')

    actions["price_deduct_min_rate_hotel"] = actions["price_deduct"] / actions["hotel_now_min_price"]
    actions["price_deduct_min_rate_basic"] = actions["price_deduct"] / actions["hotel_basicroomid_now_min_price"]
    actions['returnvalue_rate'] = actions['returnvalue'] / actions['price_deduct']
    actions["price_hotel_min_rate_with_last"] = actions["hotel_now_min_price"] / actions["hotel_minprice_lastord"]
    actions["price_basic_min_rate_with_last"] = actions["hotel_basicroomid_now_min_price"] / actions[
        "basic_minprice_lastord"]

    actions['price_user_min_rate'] = actions['price_deduct'] / actions['user_minprice']
    actions['price_user_max_rate'] = actions['price_deduct'] / actions['user_maxprice']
    actions['price_user_avg_rate'] = actions['price_deduct'] / actions['user_avgprice']
    actions['rank_rate'] = actions['rank'] / actions['user_rank_ratio']

    actions["orderid_basicroomid_price_rate_min"] = actions["orderid_basicroomid_price_deduct_min"] / actions[
        "price_deduct"]
    actions["orderid_basicroomid_hotel_rate_min"] = actions["orderid_basicroomid_price_deduct_min"] / actions[
        "hotel_now_min_price"]
    actions["orderid_basicroomid_basic_rate_min"] = actions["orderid_basicroomid_price_deduct_min"] / actions[
        "hotel_basicroomid_now_min_price"]

    actions["orderid_price_rate_min"] = actions["orderid_price_deduct_min"] / actions["price_deduct"]
    actions["orderid_hotel_rate_min"] = actions["orderid_price_deduct_min"] / actions["hotel_now_min_price"]
    actions["orderid_basic_rate_min"] = actions["orderid_price_deduct_min"] / actions["hotel_basicroomid_now_min_price"]

    actions["orderid_basicroomid_price_rate_median"] = actions["orderid_basicroomid_price_deduct_median"] / actions[
        "price_deduct"]
    actions["orderid_basicroomid_hotel_rate_median"] = actions["orderid_basicroomid_price_deduct_median"] / actions[
        "hotel_now_min_price"]
    actions["orderid_basicroomid_basic_rate_median"] = actions["orderid_basicroomid_price_deduct_median"] / actions[
        "hotel_basicroomid_now_min_price"]

    actions["orderid_price_rate_median"] = actions["orderid_price_deduct_median"] / actions["price_deduct"]
    actions["orderid_hotel_rate_median"] = actions["orderid_price_deduct_median"] / actions["hotel_now_min_price"]
    actions["orderid_basic_rate_median"] = actions["orderid_price_deduct_median"] / actions[
        "hotel_basicroomid_now_min_price"]

    actions["orderid_basicroomid_price_rate_mean"] = actions["orderid_basicroomid_price_deduct_mean"] / actions[
        "price_deduct"]
    actions["orderid_basicroomid_hotel_rate_mean"] = actions["orderid_basicroomid_price_deduct_mean"] / actions[
        "hotel_now_min_price"]
    actions["orderid_basicroomid_basic_rate_mean"] = actions["orderid_basicroomid_price_deduct_mean"] / actions[
        "hotel_basicroomid_now_min_price"]

    actions["orderid_price_rate_mean"] = actions["orderid_price_deduct_mean"] / actions["price_deduct"]
    actions["orderid_hotel_rate_mean"] = actions["orderid_price_deduct_mean"] / actions["hotel_now_min_price"]
    actions["orderid_basic_rate_mean"] = actions["orderid_price_deduct_mean"] / actions[
        "hotel_basicroomid_now_min_price"]

    actions["orderid_basicroomid_price_rate_max"] = actions["orderid_basicroomid_price_deduct_max"] / actions[
        "price_deduct"]
    actions["orderid_basicroomid_hotel_rate_max"] = actions["orderid_basicroomid_price_deduct_max"] / actions[
        "hotel_now_min_price"]
    actions["orderid_basicroomid_basic_rate_max"] = actions["orderid_basicroomid_price_deduct_max"] / actions[
        "hotel_basicroomid_now_min_price"]

    actions["orderid_price_rate_max"] = actions["orderid_price_deduct_max"] / actions["price_deduct"]
    actions["orderid_hotel_rate_max"] = actions["orderid_price_deduct_max"] / actions["hotel_now_min_price"]
    actions["orderid_basic_rate_max"] = actions["orderid_price_deduct_max"] / actions["hotel_basicroomid_now_min_price"]
    return actions


def get_now_hotel_different_time_rate(actions):
    cols = ["user_ordnum_", "user_avgprice_", "user_medprice_", "user_minprice_", "user_maxprice_",
            "user_roomservice_3_123ratio_", "user_roomservice_7_1ratio_", "user_roomservice_7_0ratio_",
            "user_roomservice_4_5ratio_", "user_roomservice_4_4ratio_",
            "user_roomservice_4_2ratio_", "user_roomservice_4_3ratio_", "user_roomservice_4_0ratio_"
            ]
    times = ['1week', '1month', '3month']
    for col in cols:
        for i in range(len(times) - 1):
            actions[col + times[i] + "rate"] = actions[col + times[i]] / actions[col + times[i + 1]]
    return actions



def orderid_price_deduct_min_mean_median_max(actions):
    orderid_basicroomid_price_deduct_min = actions[["orderid", "basicroomid", "price_deduct"]].groupby(
        ["orderid", "basicroomid"], as_index=False).min()
    orderid_basicroomid_price_deduct_min.columns = ["orderid", "basicroomid", "orderid_basicroomid_price_deduct_min"]
    orderid_basicroomid_price_deduct_median = actions[["orderid", "basicroomid", "price_deduct"]].groupby(
        ["orderid", "basicroomid"], as_index=False).median()
    orderid_basicroomid_price_deduct_median.columns = ["orderid", "basicroomid",
                                                       "orderid_basicroomid_price_deduct_median"]
    orderid_basicroomid_price_deduct_mean = actions[["orderid", "basicroomid", "price_deduct"]].groupby(
        ["orderid", "basicroomid"], as_index=False).mean()
    orderid_basicroomid_price_deduct_mean.columns = ["orderid", "basicroomid", "orderid_basicroomid_price_deduct_mean"]
    orderid_basicroomid_price_deduct_max = actions[["orderid", "basicroomid", "price_deduct"]].groupby(
        ["orderid", "basicroomid"], as_index=False).max()
    orderid_basicroomid_price_deduct_max.columns = ["orderid", "basicroomid", "orderid_basicroomid_price_deduct_max"]
    actions = pd.merge(actions, orderid_basicroomid_price_deduct_max, on=["orderid", "basicroomid"], how="left")
    actions = pd.merge(actions, orderid_basicroomid_price_deduct_mean, on=["orderid", "basicroomid"], how="left")
    actions = pd.merge(actions, orderid_basicroomid_price_deduct_median, on=["orderid", "basicroomid"], how="left")
    actions = pd.merge(actions, orderid_basicroomid_price_deduct_min, on=["orderid", "basicroomid"], how="left")

    orderid_price_deduct_min = actions[['orderid', 'price_deduct']].groupby(["orderid"], as_index=False).min()
    orderid_price_deduct_min.columns = ["orderid", "orderid_price_deduct_min"]
    actions = pd.merge(actions, orderid_price_deduct_min, on=["orderid"], how="left")
    orderid_price_deduct_median = actions[['orderid', 'price_deduct']].groupby(["orderid"], as_index=False).median()
    orderid_price_deduct_median.columns = ["orderid", "orderid_price_deduct_median"]
    actions = pd.merge(actions, orderid_price_deduct_median, on=["orderid"], how="left")
    orderid_price_deduct_mean = actions[['orderid', 'price_deduct']].groupby(["orderid"], as_index=False).mean()
    orderid_price_deduct_mean.columns = ["orderid", "orderid_price_deduct_mean"]
    actions = pd.merge(actions, orderid_price_deduct_mean, on=["orderid"], how="left")
    orderid_price_deduct_max = actions[['orderid', 'price_deduct']].groupby(["orderid"], as_index=False).max()
    orderid_price_deduct_max.columns = ["orderid", "orderid_price_deduct_max"]
    actions = pd.merge(actions, orderid_price_deduct_max, on=["orderid"], how="left")
    return actions





def del_nan_much(actions):
    cols = actions.columns.values
    print("删除前的维度为%s" % len(cols))
    rows = actions.shape[0]
    for col in cols:
        if actions[col].count() / rows <= 0.1:
            del actions[col]
    cols = actions.columns.values
    print("删除后的维度为%s" % len(cols))
    return actions


def luo_feat_1(actions):
    train_set = actions
    train_set['preferential_ratio'] = (train_set['returnvalue'] / train_set['price_deduct']) / (
    train_set['return_lastord'] / train_set['price_last_lastord'])
    train_set['star_user_avgstar_ratio'] = train_set['star'] / train_set['user_avgstar']
    train_set['star_user_avggoldstar_ratio'] = train_set['star'] / train_set['user_avggoldstar']
    train_set['star_star_lastord_ratio'] = train_set['star'] / train_set['star_lastord']
    train_set['price_deduct_user_avgdealpriceholiday_ratio'] = train_set['price_deduct'] / train_set[
        'user_avgdealpriceholiday']
    train_set['price_deduct_user_avgdealpriceworkday_ratio'] = train_set['price_deduct'] / train_set[
        'user_avgdealpriceworkday']
    train_set['price_deduct_user_avgdealprice_ratio'] = train_set['price_deduct'] / train_set['user_avgdealprice']
    train_set['price_deduct_user_avgprice_star_ratio'] = train_set['price_deduct'] / train_set['user_avgprice_star']
    train_set['price_deduct_user_avgprice_ratio'] = train_set['price_deduct'] / train_set['user_avgprice']
    train_set['price_deduct_user_maxprice_ratio'] = train_set['price_deduct'] / train_set['user_maxprice']
    train_set['price_deduct_user_minprice_ratio'] = train_set['price_deduct'] / train_set['user_minprice']
    train_set['price_deduct_user_avgprice_1week_ratio'] = train_set['price_deduct'] / train_set['user_avgprice_1week']
    train_set['price_deduct_user_medprice_1week_ratio'] = train_set['price_deduct'] / train_set['user_medprice_1week']
    train_set['price_deduct_user_minprice_1week_ratio'] = train_set['price_deduct'] / train_set['user_minprice_1week']
    train_set['price_deduct_user_maxprice_1week_ratio'] = train_set['price_deduct'] / train_set['user_maxprice_1week']
    train_set['price_deduct_user_avgprice_1month_ratio'] = train_set['price_deduct'] / train_set['user_avgprice_1month']
    train_set['price_deduct_user_medprice_1month_ratio'] = train_set['price_deduct'] / train_set['user_medprice_1month']
    train_set['price_deduct_user_minprice_1month_ratio'] = train_set['price_deduct'] / train_set['user_minprice_1month']
    train_set['price_deduct_user_maxprice_1month_ratio'] = train_set['price_deduct'] / train_set['user_maxprice_1month']
    train_set['price_deduct_user_avgprice_3month_ratio'] = train_set['price_deduct'] / train_set['user_avgprice_3month']
    train_set['price_deduct_user_medprice_3month_ratio'] = train_set['price_deduct'] / train_set['user_medprice_3month']
    train_set['price_deduct_user_minprice_3month_ratio'] = train_set['price_deduct'] / train_set['user_minprice_3month']
    train_set['price_deduct_user_maxprice_3month_ratio'] = train_set['price_deduct'] / train_set['user_maxprice_3month']
    train_set['price_deduct_hotel_minprice_lastord_ratio'] = train_set['price_deduct'] / train_set[
        'hotel_minprice_lastord']
    train_set['price_deduct_basic_minprice_lastord_ratio'] = train_set['price_deduct'] / train_set[
        'basic_minprice_lastord']
    train_set['price_deduct_returnvalue_ratio'] = train_set['price_deduct'] / train_set['returnvalue']
    train_set['price_deduct_user_avgpromotion_ratio'] = train_set['price_deduct'] / train_set['user_avgpromotion']
    train_set['returnvalue_user_avgpromotion_ratio'] = train_set['returnvalue'] / train_set['user_avgpromotion']

    train_set['rank_user_rank_ratio_ratio'] = train_set['rank'] / train_set['user_rank_ratio']

    train_set['basic_minarea_user_avgroomarea_ratio'] = train_set['basic_minarea'] / train_set['user_avgroomarea']
    train_set['basic_maxarea_user_avgroomarea_ratio'] = train_set['basic_maxarea'] / train_set['user_avgroomarea']
    train_set['basic_minarea_basic_maxarea_ratio'] = train_set['basic_minarea'] / train_set['basic_maxarea']
    return train_set


def luo_feat_2(actions):
    train_set = actions
    columns_ = [('basicroomid', 'basic_week_ordernum_ratio'),
                ('basicroomid', 'basic_recent3_ordernum_ratio'),
                ('basicroomid', 'basic_comment_ratio'),
                ('basicroomid', 'basic_30days_ordnumratio'),
                ('basicroomid', 'basic_30days_realratio'),
                ('roomid', 'room_30days_ordnumratio'),
                ('roomid', 'room_30days_realratio')]

    for room_column, ratio_column in columns_:
        sample = train_set[['hotelid', room_column, 'orderdate', ratio_column]]
        sample.drop_duplicates(['hotelid', room_column, 'orderdate'], inplace=True)

        dic_basic_hotel = {}
        for row in sample.values:
            hotelid = row[0]
            basicroomid = row[1]
            orderdate = row[2]
            basic_week_ordernum_ratio = row[3]

            key = (hotelid, orderdate)
            if key not in dic_basic_hotel:
                dic_basic_hotel[key] = {}
            dic_basic_hotel[key][basicroomid] = basic_week_ordernum_ratio

        dic = {}
        dic['hotelid'] = []
        dic[room_column] = []
        dic['orderdate'] = []
        dic[ratio_column] = []
        rank_column = ratio_column + '_rank_in_hotelid'
        dic[rank_column] = []

        for key, rooms in dic_basic_hotel.items():
            rooms = sorted(rooms.items(), key=lambda x: x[1], reverse=True)
            before_counts = np.inf
            order_index = 0
            for room, ratio in rooms:
                if before_counts > ratio:
                    order_index += 1
                    before_counts = ratio
                dic['hotelid'].append(key[0])
                dic[room_column].append(room)
                dic['orderdate'].append(key[1])
                dic[ratio_column].append(ratio)
                dic[rank_column].append(order_index)

        basic_hote_df = pd.DataFrame(dic)

        sample = train_set[['hotelid', room_column, 'orderdate', ratio_column]]
        train_sample_part = pd.merge(sample, basic_hote_df, on=['hotelid', room_column, 'orderdate'], how='left')
        train_set[rank_column] = train_sample_part[rank_column].values
    return train_set


def luo_feat_3(actions):
    train_set = actions
    roomservice_2_like = []
    roomservice_3_like = []
    roomservice_4_like = []
    roomservice_5_like = []
    roomservice_6_like = []
    roomservice_7_like = []
    roomservice_8_like = []

    roomservice_data = train_set[
        ['roomservice_2', 'user_roomservice_2_1ratio', 'roomservice_3', 'user_roomservice_3_123ratio', 'roomservice_4',
         'user_roomservice_4_0ratio', 'user_roomservice_4_1ratio', 'user_roomservice_4_2ratio',
         'user_roomservice_4_3ratio',
         'user_roomservice_4_4ratio', 'user_roomservice_4_5ratio', 'roomservice_5', 'user_roomservice_5_1ratio',
         'user_roomservice_5_345ratio',
         'roomservice_6', 'user_roomservice_6_0ratio', 'user_roomservice_6_1ratio', 'user_roomservice_6_2ratio',
         'roomservice_7',
         'user_roomservice_7_0ratio', 'roomservice_8', 'user_roomservice_8_1ratio']]
    flag = 0
    for row in roomservice_data.values:
        if flag % 50000 == 0:
            print('已经处理完：%d' % (flag))
        flag += 1

        roomservice_2 = row[0]
        user_roomservice_2_1ratio = row[1]
        user_roomservice_2_0ratio = 1 - row[1]
        if roomservice_2 == 1:
            roomservice_2_like.append(user_roomservice_2_1ratio)
        else:
            roomservice_2_like.append(user_roomservice_2_0ratio)

        roomservice_3 = row[2]
        user_roomservice_3_123ratio = row[3]

        roomservice_4 = row[4]
        user_roomservice_4_0ratio = row[5]
        user_roomservice_4_1ratio = row[6]
        user_roomservice_4_2ratio = row[7]
        user_roomservice_4_3ratio = row[8]
        user_roomservice_4_4ratio = row[9]
        user_roomservice_4_5ratio = row[10]
        if roomservice_4 == 0:
            roomservice_4_like.append(user_roomservice_4_0ratio)
        elif roomservice_4 == 1:
            roomservice_4_like.append(user_roomservice_4_1ratio)
        elif roomservice_4 == 2:
            roomservice_4_like.append(user_roomservice_4_2ratio)
        elif roomservice_4 == 3:
            roomservice_4_like.append(user_roomservice_4_3ratio)
        elif roomservice_4 == 4:
            roomservice_4_like.append(user_roomservice_4_4ratio)
        elif roomservice_4 == 5:
            roomservice_4_like.append(user_roomservice_4_5ratio)
        else:
            roomservice_4_like.append(0)

        roomservice_5 = row[11]
        user_roomservice_5_1ratio = row[12]
        user_roomservice_5_0ratio = 1 - row[12]
        user_roomservice_5_345ratio = row[13]
        if roomservice_5 == 0:
            roomservice_5_like.append(user_roomservice_5_0ratio)
        else:
            roomservice_5_like.append(user_roomservice_5_1ratio)

        roomservice_6 = row[14]
        user_roomservice_6_0ratio = row[15]
        user_roomservice_6_1ratio = row[16]
        user_roomservice_6_2ratio = row[17]
        if roomservice_6 == 0:
            roomservice_6_like.append(user_roomservice_6_0ratio)
        elif roomservice_6 == 1:
            roomservice_6_like.append(user_roomservice_6_1ratio)
        elif roomservice_6 == 2:
            roomservice_6_like.append(user_roomservice_6_2ratio)
        else:
            roomservice_6_like.append(0)

        roomservice_7 = row[18]
        user_roomservice_7_0ratio = row[19]
        user_roomservice_7_1ratio = 1 - row[19]
        if roomservice_7 == 0:
            roomservice_7_like.append(user_roomservice_7_0ratio)
        else:
            roomservice_7_like.append(user_roomservice_7_1ratio)

        roomservice_8 = row[20]
        user_roomservice_8_1ratio = row[21]

    train_set['roomservice_2_like'] = np.array(roomservice_2_like)
    train_set['roomservice_4_like'] = np.array(roomservice_4_like)
    train_set['roomservice_5_like'] = np.array(roomservice_5_like)
    train_set['roomservice_6_like'] = np.array(roomservice_6_like)
    train_set['roomservice_7_like'] = np.array(roomservice_7_like)
    return train_set


def train_test_feat(actions):
    actions = orderid_price_deduct_min_mean_median_max(actions)
    actions["orderid_basicroomid_price_deduct_rank"] = actions['price_deduct'].groupby([actions['orderid'], actions['basicroomid']]).rank()
    actions["orderid_price_deduct_min_rank"] = actions['orderid_price_deduct_min'].groupby(actions['orderid']).rank()
    actions["orderid_price_deduct_median_rank"] = actions['orderid_price_deduct_median'].groupby(actions['orderid']).rank()
    actions["orderid_price_deduct_mean_rank"] = actions['orderid_price_deduct_mean'].groupby(actions['orderid']).rank()
    actions["orderid_price_deduct_max_rank"] = actions['orderid_price_deduct_max'].groupby(actions['orderid']).rank()
    actions = last_now_data_compare_basic(actions)
    actions = last_avg_now_data_compare_data(actions)
    actions = last_price_rate(actions)
    actions = now_price_rate(actions)
    actions = luo_feat_1(actions)
    actions = luo_feat_2(actions)
    actions = luo_feat_3(actions)

    actions = actions.drop(['orderdate', 'orderid_lastord', 'orderdate_lastord', 'hotelid_lastord'
                               , 'roomid_lastord', 'basicroomid_lastord', 'rank_lastord', 'star_lastord'], axis=1)
    actions = actions.fillna(0)
    return actions
#path='/vol6/home/hnu_hcq/xiecheng/'
actions = pd.read_table("data/competition_train.txt",encoding='gb2312')
actions = data_deal(actions)
#actions=actions.iloc[:100,:]
actions =train_test_feat(actions)
actions.to_csv('cache/train_feature.csv',index=False)
df = pd.read_table("data/competition_test.txt",encoding='gb2312')
df = data_deal(df)
#df=df.iloc[:100,:]
df = train_test_feat(df)
df.to_csv('cache/test_feature.csv',index=False)
