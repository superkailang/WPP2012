import pandas as pd;
import os
import matplotlib.pyplot as plt;
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import numpy as np;
from math import ceil
import tqdm
from sklearn.metrics import mean_squared_error
from keras.utils import np_utils


def plot_result(res_data, x_features):
    for name in x_features:
        dataframe = res_data;
        # ax = plt.subplot(nfig, 1 ,idx);
        ax = dataframe.plot(x='date', y=name, kind='line', title=name);
        dataframe.plot(x='date', y=name, kind='line', ax=ax, title=name);
        plt.show();


def normalize(dataset, featrure, benchmark=None):
    minV, maxV = dataset.min(), dataset.max();
    dataset[featrure] = ((dataset[featrure] - minV[featrure]) / (maxV[featrure] - minV[featrure]));
    if (benchmark is None):
        return dataset;
    else:
        benchmark[featrure] = ((benchmark[featrure] - minV[featrure]) / (maxV[featrure] - minV[featrure]));
        return dataset, benchmark


def get_csv_data(train_file, format="%Y%m%d%H"):
    train_data = pd.read_csv(train_file, encoding='utf');
    train_data['date'] = pd.to_datetime(train_data['date'], format=format);
    return train_data;


def pre_Process_Wind_data(wind_datas, train_data, offset=36):
    set_seq = 12
    columns = list(wind_datas[0].drop(columns=['date', 'hors']).columns)
    train_data = train_data.rename(columns={"date": "merge_start"}).copy()
    new_col_len, new_f_col_len = 6, 0
    for idx, item in enumerate(tqdm.tqdm(wind_datas)):
        farm_name = 'wp' + str(idx + 1)
        item = item.rename(columns={"date": "start", "hors": "dist"})
        end_time = item.loc[item.index.values[-1], 'start']
        new_items = None
        for time_idx in range(offset // set_seq):
            new_item = item[-48:].copy();
            new_item[columns] = np.nan;
            new_item['start'] = new_item['start'].apply(lambda x: end_time + timedelta(hours=set_seq * (time_idx + 1)));
            new_items = new_item if (new_items is None) else pd.concat([new_items, new_item], axis=0);
        item = pd.concat([item, new_items], axis=0)
        item['new_date'] = item.apply(lambda item: item['start'] + timedelta(hours=(int)(item['dist'])), axis=1)
        start_time = item.loc[item.index.values[offset], 'new_date']
        item = item[item['new_date'] >= start_time]
        item.sort_values(['start', 'new_date'], ascending=[1, 1], inplace=True)
        item['turn'] = pd.DatetimeIndex(item['start']).hour
        item['wd_cut_left'] = item['wd'].apply(lambda x: (x // 30 * 30) / 360)
        # item['wd_cut_right'] = item['wd'].apply(lambda x:( (x//30+1)*30)/360 );
        item['ws_2'] = item['ws'].apply(lambda x: x ** 2)
        item['ws_3'] = item['ws'].apply(lambda x: x ** 3)  # Deal_Normalize;
        # #cutoff = item['ws'].mean() + 2 * item['ws'].std(); #item['ws'] = item['ws'].apply(lambda x: cutoff if x > cutoff else x);
        # methods = 'linear'; #item['ws']= item['ws'].interpolate(method = methods, axis=0);
        item = item.dropna(axis=0)
        history_feature = []
        for iter in tqdm.tqdm(range(0, new_col_len)):
            h_col_name = "wp_hn_" + str(iter + 1)
            f_col_name = "wp_hp_" + str(iter + 1)
            item['merge_start'] = item.apply(lambda item: item['start'] - timedelta(hours=iter), axis=1)
            item = item.merge(train_data[['merge_start', farm_name]], left_on='merge_start', right_on='merge_start',how="left")
            item = item.rename(columns={farm_name: h_col_name})
            item['merge_start'] = item.apply(lambda item: item['start'] + timedelta(hours=(49 + iter)),axis=1)
            item = item.merge(train_data[['merge_start', farm_name]], left_on='merge_start', right_on='merge_start',how="left")
            item = item.rename(columns={farm_name: f_col_name})
            item.loc[item[f_col_name].isna(), f_col_name] = item.loc[item[f_col_name].isna(), h_col_name]
            item.loc[item['dist']>=24,h_col_name] = item.loc[item['dist']>=24,f_col_name]
            #item['merge_start'] = item.apply(lambda item: item['start'] - timedelta(hours=iter),item['start'] + timedelta(hours=(49 + iter))), axis=1)
            item.drop(columns=[f_col_name],inplace=True)
            history_feature.append(h_col_name)
        wind_datas[idx] = item.drop(columns=['merge_start']);
    return wind_datas, history_feature


def get_time_attribute(data, key_time):
    time_features = ['year', 'month', 'week', 'hour']
    for time_feature in time_features:
        if (time_feature == 'year'):
            data[time_feature] = pd.DatetimeIndex(data[key_time]).year
        elif time_feature == 'month':
            data[time_feature] = pd.DatetimeIndex(data[key_time]).month
        elif time_feature == 'week':
            data[time_feature] = pd.DatetimeIndex(data[key_time]).week
        elif time_feature == 'hour':
            data[time_feature] = pd.DatetimeIndex(data[key_time]).hour
        elif time_feature == 'weekday':
            data[time_feature] = pd.DatetimeIndex(data[key_time]).weekday
    return data


def normalize_Data(train_data, wind_feature):
    # Normalization
    power_scaler = MinMaxScaler(feature_range=(0, 1));
    power_features = ['year', 'month', 'week', 'hour'];
    power_scaler.fit_transform(train_data[~(train_data['ws'].isnull())][power_features].values);
    train_data[power_features] = power_scaler.transform(train_data[power_features]);
    scaler = MinMaxScaler(feature_range=(0, 1));
    columns = wind_feature;
    scaler.fit_transform(train_data[columns].values);
    train_data[columns] = scaler.transform(train_data[columns]);
    return train_data;


def get_data(train_file, benchmark_file, wind_files, test_file,fill_Nan = None):
    try:
        time_offset = 36  # Filter Forcast_time;

        train_data = get_csv_data(train_file)
        benchmark_data = get_csv_data(benchmark_file)
        test_data = get_csv_data(test_file)

        for name in benchmark_data.drop(columns=['id', 'date']).columns:
            benchmark_data[name] = np.NaN
        train_data.sort_values(['date'], ascending=[1], inplace=True)
        benchmark_data.sort_values(['date'], ascending=[1], inplace=True)
        train_data['Flag'] = train_data.apply(lambda item: True, axis=1)
        benchmark_data['Flag'] = benchmark_data.apply(lambda item: False, axis=1)

        if (fill_Nan is None):  # Default
            train_data = get_time_attribute(train_data, 'date')
            benchmark_data = get_time_attribute(benchmark_data, 'date')
            power_data = train_data
        else:
            power_data = pd.concat([train_data, benchmark_data.drop(columns=['id'])], ignore_index=True, axis=0)
            power_data = get_time_attribute(power_data,'date')

        wind_datas = [get_csv_data(wind_file) for wind_file in wind_files]

        farm_features = ['wp' + str(idx + 1) for idx, item in enumerate(wind_datas)]
        wind_datas, history_feature = pre_Process_Wind_data(wind_datas, train_data, time_offset)

        drop_col = []
        wind_feature = list(wind_datas[0].drop(columns=['start', 'new_date'] + drop_col).columns)
        train_datas = None
        power_feature = list(power_data.drop(columns=farm_features).columns)
        test_datas = []
        if(fill_Nan is None):
            benchmark_data['time'] = np.concatenate([list(range(1,49)) for i in range(0,benchmark_data['date'].shape[0],48)],axis=0)
            benchmark_data['start'] = benchmark_data.apply(lambda item: item['date'] - timedelta(hours=item['time']),axis=1)
        for idx, feature in enumerate(tqdm.tqdm(farm_features)):
            x_list = power_feature + [farm_features[idx]]
            test_farm = test_data[['date', farm_features[idx]]].copy()
            test_farm['Farm'] = idx + 1
            test_datas.append( test_farm.rename(columns={farm_features[idx]: "power"}) )
            temp_farm  = power_data[x_list].copy()
            farm_data = temp_farm.merge(wind_datas[idx], left_on='date',right_on='new_date')  ## temp_farm.shift(periods=1)['power']
            if fill_Nan is None:
                test_farm_data  = benchmark_data[x_list+['start']].merge(wind_datas[idx], left_on=['date', 'start'], right_on=['new_date', 'start'])
                temp_farm = pd.concat([farm_data,test_farm_data],axis=0)
            else:
                temp_farm = farm_data
            temp_farm['Farm'] = idx + 1
            temp_farm = temp_farm.rename(columns={farm_features[idx]: "power"})
            train_datas = pd.concat([train_datas, temp_farm], axis=0,ignore_index=True) if train_datas is not None else temp_farm
        train_datas['catagory'] = train_datas['dist'].apply(lambda x: ceil(x / 12))
        train_datas.sort_values(['date', 'Farm', 'dist'], ascending=[1, 1, 0], inplace=True)
        #if fill_Nan is not None:
        #    train_datas = train_datas.fillna(method='ffill')
        train_datas = normalize_Data(train_datas, list(
            wind_datas[0].drop(columns=['start', 'new_date', 'wd_cut_left'] + history_feature).columns))
        train_datas = train_datas.reset_index(drop=True)
        test_datas = pd.concat(test_datas,axis=0)
        ret = { 'train_datas': train_datas, 'benchmark_data': benchmark_data,
               'test_data': test_datas }
        ret['farm_feature'], ret['wind_feature'] = farm_features, wind_feature
        ret['power_feature'] = ['Farm'] + ['year', 'month', 'hour']  # + ['wp_hn']; ##['Farm','week']+ ['previous']
        # plot_orginal_Data(train_datas[(train_datas['Farm']==1) & (train_datas['dist']<=12)],test_data);
        # data = train_datas[(train_datas['Farm'] == 1) & (train_datas['catagory'] == 1)];
        # plt.plot(data['power'], 'b-o');# plt.plot(data['ws'], 'r-+');# plt.show();
        return ret
    except Exception as err:
        print(err)
        return None
    # 绘图
    # x_farmms = train_data.drop(columns=['date']).columns
    # wind_features = list(wind_datas[0].drop(columns = ['date','hors']).columns)
    # plot_result(wind_datas[0],wind_features)


def plot_orginal_Data(train_datas, test_data):
    plt.figure(1);
    plt.plot(train_datas['power'], 'r-+');
    plt.plot(train_datas['ws'], 'b--o');
    # train_datas.plot(x='date', y='power', kind='line', title='power'); # style='o'
    # train_datas.plot(x='date', y='ws', kind='line',  title='power');
    plt.legend(['power', 'ws']);
    plt.show();
    plt.figure(2);
    test_data.plot( x= 'date', y= 'wp1', kind= 'line', title= 'power');  # style='o'
    # test_data.plot(x='date', y='ws', kind='line',  title='power');
    plt.show();


def get_test_data(test_file):
    return get_csv_data(test_file);


def get_predict_data(predict_file):
    train_data = pd.read_csv(predict_file, encoding='utf');
    return train_data;


def deal_abnormal_value(data, name, methods='linear'):
    default_items = data[(data[name].isna()) | (data[name].isnull()) | (data[name] == '')];
    # 判断是否有空值;
    if (default_items.shape[0] > 0):
        data[name] = data[name].apply(
            lambda x: np.NaN if (data[name].isna()) | (data[name].isnull()) | (data[name] == '') else x);
    if (data[name].dtypes == object):
        data[name] = pd.to_numeric(data[name], errors='coerce')
    cutoff = data[name].mean() + 3 * data[name].std();
    data[name] = data[name].apply(lambda x: np.NaN if x > cutoff else x);
    temp_interpolate = data[name];
    temp_interpolate = temp_interpolate.interpolate(method=methods, axis=0);
    data[name] = temp_interpolate;
    # data[name] = data[name].fillna(method=methods);
