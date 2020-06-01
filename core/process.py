import numpy as np
from datetime import timedelta
import pandas as pd
import threading
from sklearn.model_selection import train_test_split
import tqdm
from keras.utils import np_utils
import math
from sklearn import preprocessing
from matplotlib import pyplot as plt
from os.path import join

def Process(X_train,y_train):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train);



from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

def ISO_Forestclustering(X,outliers_fraction = 0.01,show=True):
    anomaly_algorithms = IsolationForest(contamination=outliers_fraction,
                                         random_state=42)
    y_pred = anomaly_algorithms.fit(X).predict(X)
    if show:
        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])
        plt.show()
    return y_pred

def KNN_Clustering(X,x_list,min_sample,n_clusters=100,show=True):
    if False:
        wcss = []
        for i in range(2, 200,5):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        # Visualizing the ELBOW method to get the optimal value of K
        plt.plot(range(2, 200,5), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('no of clusters')
        plt.ylabel('wcss')
        plt.show()
    kmeansmodel = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    y_kmeans = kmeansmodel.fit_predict(X)
    cluser_label = "Cluster 1"
    cluser_label2 = "Cluster 2"
    X['cluster'] = 1
    for i in range(n_clusters):
        if X.values[y_kmeans == i, 0].shape[0] <= min_sample:
            X.loc[X.index[y_kmeans == i],'cluster'] = 0
            plt.scatter( X.values[y_kmeans == i, 0], X.values[y_kmeans == i, 1], s=10, c='blue',label= cluser_label) #, label='Cluster {}'.format(1))  # c='red',
            cluser_label = None
        else:
            plt.scatter(X.values[y_kmeans == i, 0], X.values[y_kmeans == i, 1], s=10, c='green',label = cluser_label2) #, label='Cluster {}'.format(2))  # c
            cluser_label2 = None
        # plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c='blue', label='Cluster 2')
    plt.title('Clusters of Wind Component')
    plt.xlabel(x_list[0])
    plt.ylabel(x_list[1])
    plt.legend()
    save_file = join('core','image','cluster')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches = 0,bbox_inches='tight')
    plt.show()
    return X

def pre_process(train_data,y_list="TARGETVAR", n_clusters = 50,min_sample = 20):
    ### First找到异常点
    X_list = ['u', 'v']  # ,'V100','U100'
    zero_output = train_data.loc[train_data[y_list]==0,:]
    X = zero_output.loc[:,X_list].copy()
    clusting = "kmeans"
    if clusting == "ISO_Forestclustering":
        ISO_Forestclustering(X)
    if clusting == "DBSCAN":
        outlier_detection = DBSCAN(
            eps=.1,
            metric="euclidean",
        min_samples = 5,
                      n_jobs = -1)
        clusters = outlier_detection.fit_predict(X)
        from matplotlib import cm
        cmap = cm.get_cmap("Set1")
        X = zero_output.loc[:, X_list]
        X.plot.scatter(x=X_list[0], y =X_list[1], c = clusters, cmap = cmap,colorbar = False)
        plt.show()
    else:
        X = KNN_Clustering(X,X_list,n_clusters=n_clusters,min_sample = min_sample)
        zero_output.loc[:, 'cluster'] = X['cluster']
        train_data.loc[zero_output.index,'cluster'] =  X['cluster']
        Lag_offset = 1
        ### 时间依赖性  [1,2]
        zero_output['pre_TimeDiff'] = (zero_output['date'] - zero_output['date'].shift(periods=Lag_offset)).astype('timedelta64[h]')
        zero_output['next_TimeDiff'] = (zero_output['date'] - zero_output['date'].shift(periods=-Lag_offset)).astype('timedelta64[h]')

        noise_data = zero_output[ (zero_output['cluster']==0 ) ] #  & (zero_output['next_TimeDiff'] != -1) & (zero_output['pre_TimeDiff'] != 1)]
        print(noise_data.shape)

        train_data.loc[noise_data.index,y_list] = np.nan
        train_data = train_data.interpolate(method="cubic")
        return train_data





from sklearn.metrics import mean_squared_error

def Process2(res_data,sequence_length = 49,train_val_split = 0.99):
    train_data,benchmark_data = res_data['train_data'],res_data['benchmark_data'];
    wind_datas = res_data['wind_datas'];
    start_time = benchmark_data['date'].min();
    farm_features = ['wp' + str(idx + 1) for idx in range(len(wind_datas))];
    time_feature  = ['Farm', 'year', 'month', 'hour'];
    x_features = ['power'] + farm_features + time_feature;
    power_features = x_features;
    wind_features = list(wind_datas[0].drop(columns=['date', 'hors']).columns);
    x_features = x_features + wind_features;
    res_data['x_features'] =x_features;

    wind_train_Data,merge_datas= [],[]
    sequence_length = sequence_length - 1
    seq_interval = 48* 10
    train_split = (train_val_split, 1 - train_val_split);
    x_Datas, y_Datas = None, None
    res_Data = dict();
    for idx, wind_data in enumerate(wind_datas):
        wind_data = wind_data.rename(columns={"date": "wind_date"})
        wind_data['new_date'] = wind_data.apply(lambda item: item['wind_date'] + timedelta(hours=(int)(item['hors'])),axis=1);
        wind_data.sort_values(['wind_date','new_date'], ascending=[1, 1], inplace=True)
        temp_wind_data = wind_data.groupby(wind_data['new_date']).tail(1);
        wind_train_Data.append(temp_wind_data);
        train_val_data = train_data[idx][train_data[idx]['date'] < start_time];
        offset,iter= 1,0;
        while iter < (train_val_data.shape[0] - 2*sequence_length - offset):
            pre_time = train_val_data.loc[train_val_data.index.values[iter + sequence_length],'date'];
            predict_wind = wind_data[wind_data['wind_date'] == pre_time];
            previous_wind = temp_wind_data[iter:iter + sequence_length];
            merge_windData = pd.concat([previous_wind, predict_wind],axis=0);
            previous_data = train_val_data[ iter + offset:2 * sequence_length + iter + offset];
            train_matrix = np.concatenate([previous_data[power_features].values,merge_windData[wind_features].values],axis=1);
            X_data, Y_data = get_train_data(train_matrix, sequence_length + 1);

            test_X_data = train_val_data[ (train_val_data['date'] <= pre_time) & ( train_val_data['date'] >  pre_time - timedelta(hours = sequence_length) )][farm_features].values;
            test_time_X_data = train_val_data[ (train_val_data['date'] <= pre_time+ timedelta(hours=1)) & ( train_val_data['date'] >  pre_time - timedelta(hours = sequence_length-1) )][time_feature].values;
            test_wind_data = temp_wind_data[ (temp_wind_data['new_date']<= pre_time+ timedelta(hours=1)) & ( temp_wind_data['new_date'] >  pre_time - timedelta(hours = sequence_length-1) )][wind_features].values;
            test_X_data = np.concatenate([test_time_X_data,test_wind_data,test_X_data],axis=1);
            print("Test For X_Data RMSE :{}".format( mean_squared_error( test_X_data,X_data[0])));
            predict_data = train_val_data[ (train_val_data['date'] >= predict_wind.loc[predict_wind.index.values[0],'new_date']) & ( train_val_data['date'] <= predict_wind.loc[predict_wind.index.values[-1],'new_date']) ];
            Y_test_data = predict_data[farm_features[idx]].values;
            print("Test For Y_Data RMSE: {}".format( mean_squared_error(Y_test_data,Y_data)**0.5 ) );
            print("Test For X_Datas RMSE: {}".format(mean_squared_error(X_data[0][1:48,:], X_data[1][0:47,:]) ** 0.5));
            iter = iter + seq_interval;
            x_Datas = X_data if x_Datas is None else np.concatenate( (x_Datas,X_data),axis=0);
            y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0);
        wind_datas[idx] = wind_data;
        res_Data[ x_features[idx+1] ] = dict();
    if (pd.isnull(np.array(x_Datas, dtype=float)).sum() > 0):
        print("Input NAN {}".format(idx));
    if (pd.isnull(np.array(y_Datas, dtype=float)).sum() > 0):
        print("Input NAN {}".format(idx));
    (x_train, y_train), (x_test, y_test) = split_data(x_Datas, y_Datas, train_split);
    res_data['x_train'],res_data['y_train'],res_data['x_test'],res_data['y_test'] = x_train, y_train,x_test, y_test;
    res_data['wind_train_Data'] = wind_train_Data;
    res_data['wind_datas'] = wind_datas;
    return res_Data;

def Process3(res_data,sequence_length = 49,train_val_split = 0.99):
    train_data,benchmark_data = res_data['train_data'],res_data['benchmark_data'];
    wind_datas = res_data['wind_datas'];
    start_time = benchmark_data['date'].min();
    farm_features = ['wp' + str(idx + 1) for idx in range(len(wind_datas))];
    x_features = ['power'] + farm_features + ['Farm', 'year', 'month', 'hour'];
    power_features = x_features;
    wind_features = list(wind_datas[0].drop(columns=['date', 'hors']).columns);
    x_features = x_features + wind_features;
    res_data['x_features'] =x_features;

    wind_train_Data,merge_datas= [],[];
    sequence_length = sequence_length - 1;
    seq_interval = 480;
    train_split = (train_val_split, 1 - train_val_split);
    x_Datas, y_Datas = None, None;
    res_Data = dict();
    for idx, wind_data in enumerate(wind_datas):
        wind_data = wind_data.rename(columns={"date": "wind_date"})
        wind_data['new_date'] = wind_data.apply(lambda item: item['wind_date'] + timedelta(hours=(int)(item['hors'])),axis=1);
        wind_data.sort_values(['wind_date','new_date'], ascending=[1, 1], inplace=True)
        temp_wind_data = wind_data.groupby(wind_data['new_date']).tail(1);
        wind_train_Data.append(temp_wind_data);
        train_val_data = train_data[idx][train_data[idx]['date'] < start_time];
        offset,iter= 1,0;
        while iter < (train_val_data.shape[0] - 2*sequence_length - offset):
            pre_time = train_val_data.loc[train_val_data.index.values[iter + sequence_length],'date'];
            predict_wind = wind_data[wind_data['wind_date'] == pre_time];
            #previous_wind = temp_wind_data[iter:iter + sequence_length];
            #merge_windData = pd.concat([previous_wind, predict_wind],axis=0);
            previous_data = train_val_data[ iter + offset:sequence_length + iter + offset];
            predict_data =  train_val_data[ sequence_length + iter + offset:2 * sequence_length + iter + offset];
            Y_data = np.array( [predict_data[power_features].values[:,0]] );
            x_idx1 = list( range(8, previous_data[power_features].shape[1]) );
            x_idx2 = list(range(1,8));
            X_data = np.array( [ np.concatenate( [ previous_data[power_features].values[:,x_idx1],predict_wind[wind_features].values,previous_data[power_features].values[:,x_idx2]],axis=1)]);
            iter = iter + seq_interval;
            x_Datas = X_data if x_Datas is None else np.concatenate( (x_Datas,X_data),axis=0);
            y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0);
        wind_datas[idx] = wind_data;
        res_Data[ x_features[idx+1] ] = dict();

    (x_train, y_train), (x_test, y_test) = split_data(x_Datas, y_Datas, train_split);
    res_data['x_train'],res_data['y_train'],res_data['x_test'],res_data['y_test'] = x_train, y_train,x_test, y_test;
    res_data['wind_train_Data'] = wind_train_Data;
    res_data['wind_datas'] = wind_datas;
    return res_Data;

#获取二分类
def get_Process_Power(res_data,train_val_split = 0.99, thread_value = 0.05):
    # 二分类问题; 第一步不考虑时间; 只考虑一维特征;
    number_class = 2;
    wind_features = res_data['wind_feature'];
    farm_features,power_feature= res_data['farm_feature'],res_data['power_feature'];
    train_Data = res_data['train_datas'].copy()
    train_Data['power'] = train_Data['power'].apply(lambda x: 0 if x< thread_value else 1);
    groups_data = train_Data[~(train_Data['ws'].isnull())].groupby(train_Data['Farm']);
    res_Data = dict();
    add_features_exist = train_Data.columns.contains("ws.angle");
    add_features_ = ["ws.angle"] if add_features_exist else [];
    x_list = power_feature + wind_features + ['previous'] + add_features_;# 即考虑之前也考虑时间+风特征;
    for idx, group_item in groups_data:
        farm_id = farm_features[idx - 1];
        y_train, x_train = np.array(group_item['power'].values), np.array(group_item[x_list].values);
        # y_train = np_utils.to_categorical(y_train);
        res_Data[farm_id] = {};
        res_Data[farm_id]['x_train'], res_Data[farm_id]['x_test'], res_Data[farm_id]['y_train'], res_Data[farm_id][
            'y_test'] = \
            train_test_split(x_train, y_train, test_size=1 - train_val_split, random_state=0)
    return res_Data;

def get_wind_train_data(train_data,x_list,train_val_split):
    y_train, x_train = np.array(train_data['power'].values), np.array(train_data[x_list].values);
    res_Data = {};
    res_Data['x_train'], res_Data['x_test'], res_Data['y_train'], res_Data['y_test'] = \
            train_test_split(x_train, y_train, test_size=1 - train_val_split, random_state=0)
    return res_Data;

## 获取回归数据;
def Wind_Power_Process(res_data,sequence_length = 49,train_val_split = 0.99,seq_interval = 12,addfeature=None):
    try:
        wind_features,farm_feature = res_data['wind_feature'],res_data['farm_feature'];
        power_feature = res_data['power_feature'];
        train_Power_Data = res_data['train_datas'].copy()
        train_Power_Data.sort_values(['date','Farm','dist'], ascending=[1,1,0], inplace=True)
        res_Data = dict();
        groups_data = train_Power_Data.groupby(train_Power_Data['Farm']); #Train_Data
        sequence_length = sequence_length - 1;
        column_titles = ['wind_power_' + str(subidx + 1) + 'h' for subidx in range(0, 4)];
        y_features = ['power']
        if(addfeature is None):
            column_titles=[];
            y_features = ['catagory_power']
        x_features = power_feature + column_titles + ['power'];
        res_Data['y_features'] = y_features;
        res_data['x_features'] = x_features;
        for idx, group_item in tqdm.tqdm(groups_data):
            farm_id = farm_feature[idx - 1];
            train_data = None;
            if( addfeature is None):
                group_catagory = group_item.groupby(group_item['catagory']);
                for subidx, sub_catagory in group_catagory:
                    column_title = column_titles[subidx-1];
                    sub_catagory = sub_catagory.rename(columns={"ws.angle": column_title});
                    if( train_data is None):
                        train_data = sub_catagory;
                    else:
                        train_data = train_data.merge(sub_catagory[['date',column_title]], left_on='date', right_on='date', how='right');
            train_data.sort_values(['date'], ascending=[1], inplace=True)
            res_Data[farm_id] = {};
            res_Data[farm_id]['power_Data'] = train_data;
            train_data = train_data[train_data['Flag']==True];
            x_Datas, y_Datas = None, None; # 48h
            iter= 0;
            # time_lapse = 48;
            while iter < ( train_data.shape[0] - 2 * sequence_length):
                pred_start_time = train_data.loc[ train_data.index.values[iter + 2 * sequence_length],'date'] -timedelta(hours = 2 * sequence_length);
                start_time = train_data.loc[ train_data.index.values[iter],'date'];
                if( start_time == pred_start_time):
                    train_matrix = train_data[iter : 2 * sequence_length + iter ][x_features + y_features ].values;
                    X_data,Y_data = get_power_train_data(train_matrix,sequence_length+1);
                    x_Datas = X_data if x_Datas is None else np.concatenate((x_Datas, X_data), axis=0);
                    y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0);
                #else:
                #    print(start_time);
                iter = iter + seq_interval;
            x_train, x_test, y_train, y_test = train_test_split(x_Datas,y_Datas,test_size= 1- train_val_split,random_state=0);
            #res_Data[farm_id]['x_train'],res_Data[farm_id]['y_train'] = x_Datas,y_Datas
            res_Data[farm_id]['x_train'], res_Data[farm_id]['y_train'],res_Data[farm_id]['x_test'],res_Data[farm_id]['y_test'] = \
                x_train, y_train,x_test, y_test
        return res_Data;
    except Exception as e:
        print(e);
        return None;

def get_Process_Wind_Data(res_data,x_list,train_val_split = 0.99,thread_value = 0.00,total=None):
    train_Data = res_data['train_datas'].copy();
    farm_feature = res_data['farm_feature']
    train_Data = train_Data[(train_Data['power'] >= thread_value) & (train_Data['Flag']==True) ];
    groups_data = train_Data[~(train_Data['ws'].isnull())].groupby(train_Data['Farm']);
    res_Data = dict()
    for idx, group_item in groups_data:
        farm_id = farm_feature[idx - 1]
        y_train,x_train = np.array( group_item['power'].values), np.array( group_item[x_list].values);
        res_Data[farm_id] = {}
        res_Data[farm_id]['x_train'], res_Data[farm_id]['x_test'],  res_Data[farm_id]['y_train'],res_Data[farm_id]['y_test'] = \
            train_test_split(x_train,y_train,test_size= 1 - train_val_split, random_state=0)
    if(total is not None):
        farm_id ='total'; res_Data[farm_id]={}
        y_train, x_train = np.array(train_Data['power'].values), np.array(train_Data[x_list].values);
        res_Data[farm_id]['x_train'], res_Data[farm_id]['x_test'], res_Data[farm_id]['y_train'], res_Data[farm_id][
            'y_test'] = \
            train_test_split(x_train, y_train, test_size=1 - train_val_split, random_state=0)
    return res_Data;

def get_wind_predict(y_features,res_data,offset=36,sequence_length=48,train_val_split = 0.99,total=None):
    train_Power_Data = res_data['train_datas'].copy()
    wind_feature,farm_feature,power_feature = res_data['wind_feature'], res_data['farm_feature'],res_data['power_feature']
    seq_interval = 12
    res_Data = dict()
    x_features = power_feature + wind_feature # ['u','v','ws','wd']; # y_features = ['ws'];
    groups_data = train_Power_Data.groupby(train_Power_Data['Farm'])
    total_x_train, total_y_train = None, None
    for idx, group_item in groups_data:
        farm_id = farm_feature[idx-1]
        train_data = group_item[group_item['catagory'] == 1].copy()
        forcast_data = group_item.copy()
        forcast_data.sort_values(['start', 'dist', 'date'], ascending=[1, 1, 1], inplace=True)
        res_Data[farm_id] = {}
        x_Datas,y_Datas = None,None;
        for iter in tqdm.tqdm(range(0, ( train_data.shape[0] - sequence_length - offset), seq_interval)):
            pred_start_time = train_data.loc[train_data.index.values[iter + sequence_length + offset], 'date'] \
                              - timedelta(hours=(sequence_length + offset))
            start_time = train_data.loc[train_data.index.values[iter], 'date']
            if (start_time == pred_start_time):
                train_matrix = pd.concat([train_data[iter:(iter + offset)], forcast_data[
                        forcast_data['start'] == (start_time + timedelta(hours=offset-1))]], axis=0)
                X_data = np.array([train_matrix[x_features].values])
                Y_data = np.array([train_data[(iter+sequence_length):(iter+sequence_length+offset)][y_features[0]].values])
                x_Datas = X_data if x_Datas is None else np.concatenate((x_Datas, X_data), axis=0)
                y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0)
        if (total is not None):
            total_x_train = x_Datas if total_x_train is None else np.concatenate([total_x_train, x_Datas], axis=0);
            total_y_train = y_Datas if total_y_train is None else np.concatenate([total_y_train, y_Datas], axis=0);
        x_train, x_test, y_train, y_test = train_test_split(x_Datas, y_Datas, test_size = 1 - train_val_split,random_state=0)
        res_Data[farm_id]['x_train'], res_Data[farm_id]['y_train'], res_Data[farm_id]['x_test'], \
        res_Data[farm_id]['y_test'] = x_train, y_train, x_test, y_test
    if (total is not None):
        res_Data['total'] = {};
        total_x_train, total_x_test, total_y_train, total_y_test = train_test_split(total_x_train, total_y_train,
                                                                                    test_size=1 - train_val_split,
                                                                                    random_state=0);
        res_Data['total']['x_train'], res_Data['total']['y_train'], res_Data['total']['x_test'], \
        res_Data['total']['y_test'] = total_x_train, total_y_train, total_x_test, total_y_test
    return res_Data;

def update_wind_predict(y_features,res_data,model_data,offset=36,sequence_length=48,train_val_split = 0.99,Limit_Number=None,update_vector=None):
    train_Power_Data = res_data['train_datas'].copy();
    wind_feature,farm_feature,power_feature = res_data['wind_feature'], res_data['farm_feature'],res_data['power_feature'],
    seq_interval = 48
    res_Data = dict()
    bench_mark = None;
    x_features = power_feature + wind_feature; # y_features = ['ws']; #['u','v','ws','wd']
    groups_data = train_Power_Data.groupby(train_Power_Data['Farm'])
    for idx, group_item in groups_data:
        farm_id = farm_feature[idx-1]
        train_data = group_item[group_item['catagory'] == 1].copy();
        predict_data = group_item[group_item['Flag']==False].copy();
        predict_data.sort_values(['start', 'dist', 'date'], ascending=[1, 1, 1], inplace=True)
        res_Data[farm_id] = {}
        if (Limit_Number is None):
            Limit_Number = predict_data.shape[0];
        for iter in tqdm.tqdm(range(0, ( predict_data.shape[0] - sequence_length - offset), 4 * seq_interval)):
            start_time = predict_data.loc[predict_data.index.values[iter], 'date'] - timedelta(hours=offset);
            predict_index_loc = predict_data.index.values[iter];
            start_index_loc = train_data[train_data['date'] == start_time].index.values[0];
            pred_start_time = train_data.loc[start_index_loc, 'date'];
            if (start_time == pred_start_time):
                train_matrix = pd.concat([ train_data.loc[ start_index_loc:predict_index_loc], predict_data[
                    predict_data['start'] == ( start_time + timedelta(hours= offset-1))]], axis=0)
                x_predict = np.array([ train_matrix[x_features].values] )
                y_predict = model_data[farm_id]['model'].model.predict( x_predict );

                if(train_data[train_data['date']== (start_time + timedelta(hours=sequence_length))].shape[0]==0):
                    print("XX");
                start_index_loc = train_data[train_data['date']== (start_time + timedelta(hours=sequence_length))].index.values[0];
                end_index_loc = train_data[train_data['date']== (start_time + timedelta(hours=offset+sequence_length-1))].index.values[0];
                index_loc = train_data.loc[start_index_loc:end_index_loc].index.values;
                train_data.loc[index_loc, y_features] = y_predict[0][:];
                train_Power_Data.loc[index_loc, y_features] = y_predict[0][:];
    if y_features == "wd":
        train_Power_Data['wd_cut_left'] = train_Power_Data['wd'].apply(lambda x: x // 30 * 30);
    elif y_features == "ws":
        train_Power_Data['ws_2'] = train_Power_Data['ws'].apply(lambda x: x ** 2);
        train_Power_Data['ws_3'] = train_Power_Data['ws'].apply(lambda x: x ** 3);
    if(update_vector is not None):
        train_Power_Data['u'] = train_Power_Data.apply(lambda x: x['ws'] * math.sin(x['wd']/180*math.pi),axis=1);
        train_Power_Data['v'] = train_Power_Data.apply(lambda x: x['ws'] * math.cos(x['wd']/180*math.pi),axis=1);
    res_data['train_datas'] = train_Power_Data;

#分类问题;
def get_seqence_data_back(train_data,sequence_length,offset,seq_interval,x_features,y_features,offset_time,offset_hour,offset_wind=None,reverse=None,single=None,is_add_future=None):
    x_Datas, y_Datas = None, None
    for iter in tqdm.tqdm(range(0, (train_data.shape[0] - sequence_length - offset), seq_interval)):
        pred_start_time = train_data.loc[train_data.index.values[iter + sequence_length + offset], 'date'] - offset_time #timedelta(hours=(sequence_length + offset));
        start_time = train_data.loc[train_data.index.values[iter], 'date'];
        """
        if (start_time == pred_start_time):
            if (offset_wind):
                forcast_data = [];
                train_matrix = pd.concat([train_data[iter:(iter + sequence_length)], forcast_data[
                    forcast_data['start'] == (start_time -offset_hour  )]], axis=0); # timedelta(hours=1)
                train_matrix = train_matrix[x_features + y_features].values;
            else:
                train_matrix = train_data.loc[
                    train_data.index.values[iter: sequence_length + iter + offset], x_features + y_features].values;
            if (reverse is None):
                X_data, Y_data = get_power_train_data(train_matrix, sequence_length, offset, single=single);
            else:
                X_data, Y_data = get_power_train_data2(train_matrix, sequence_length, offset, single=single);
            if (is_add_future):  # Future For Analysis;
                new_matrix = np.array([train_data.loc[train_data.index.values[sequence_length + iter + offset:(
                        sequence_length + iter + offset + sequence_length)], 'power'].values]);
                X_data = np.concatenate([X_data, new_matrix.reshape(-1, new_matrix.shape[-1], 1)], axis=2);
            x_Datas = X_data if x_Datas is None else np.concatenate((x_Datas, X_data), axis=0);
            y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0);
        else:
            continue;
        """
    return x_Datas, y_Datas


#from numba import jit
#@jit(nopython=True)
def get_power_data(train_val_data,sequence_length,offset ):
    x_idxs_power = np.array([-2]);
    x_idxs_wind =  np.array(list(range(0, train_val_data.shape[1] - 1)));
    x_idxs_wind2 = np.array( list(range(0, train_val_data.shape[1] - 2)) + [-1 ])
    y_idxs = np.array([-1]);

    X_data = np.concatenate([train_val_data[0:sequence_length, x_idxs_wind],train_val_data[(offset + sequence_length):(
            offset + 2 * sequence_length), x_idxs_wind]], axis = 0)


   # X_data = train_val_data[0:sequence_length,x_idxs_wind ]
    X_data = train_val_data[0:(sequence_length+offset), x_idxs_wind]

    ### 修改2020/02/26 增加（power=>)
    X_data = np.concatenate( [train_val_data[0:(sequence_length), x_idxs_wind2],train_val_data[(sequence_length):(sequence_length+offset), x_idxs_wind]],axis=0)
    Y_data = train_val_data[sequence_length:  (sequence_length + offset), y_idxs[0]];
    """
    if( sequence_length > offset ):
        X_data = np.concatenate((train_val_data[offset:  (sequence_length + offset), x_idxs_wind],
                                          train_val_data[0: sequence_length, x_idxs_power]), axis=1);
        Y_data = train_val_data[sequence_length:  (sequence_length + offset), y_idxs[0]];
    else:
        X_data = np.concatenate((train_val_data[sequence_length:  (2 * sequence_length), x_idxs_wind],
                                          train_val_data[0: sequence_length, x_idxs_power]), axis=1);
        Y_data = train_val_data[sequence_length:  (sequence_length + offset), y_idxs[0]];
    """
    return X_data, Y_data;

#from numba import jit
#@jit(nopython=True)
def get_sub_sequence_data(train_matrix,idxs,sequence_length,offset):
    num_len = idxs.shape[0]
    x_Datas = np.empty(shape=( num_len, sequence_length+offset, train_matrix.shape[1] - 1), dtype=np.float64)
    y_Datas = np.empty(shape=( num_len, offset), dtype=np.float64);
    idx = 0
    for iter in idxs:
        train_val_data = train_matrix[iter: sequence_length + iter + offset]
        x_Datas[idx], y_Datas[idx] = get_power_data(train_val_data, sequence_length, offset)
        idx = idx + 1
    return x_Datas,y_Datas;

def get_seqence_data(train_data,sequence_length,offset,seq_interval,x_features,y_features,offset_time):
    idxs = [];
    for iter in range(0, (train_data.shape[0] - sequence_length - offset), seq_interval):
        pred_start_time = train_data.loc[train_data.index.values[iter + sequence_length + offset], 'date'] - offset_time
        start_time = train_data.loc[train_data.index.values[iter], 'date']
        if (start_time == pred_start_time):
            idxs.append(iter)
    idxs = np.array(idxs);
    train_matrix = train_data[ x_features + y_features ].values;
    x_Datas,y_Datas = get_sub_sequence_data(train_matrix,idxs,sequence_length,offset)
    return x_Datas, y_Datas


"""
# 直接按照分组获取36->48
def Multi_Wind_Power_Process(res_data,params,is_add_future=None,single = None, offset = 48,reverse=None):
    try:
        sequence_length,train_val_split = params['sequence_length'],params['train_set_fraction'];
        seq_interval,number_catagory,offset_wind = params['seq_interval'],params['number_catagory'],params['offset_wind'];
        addfeature,obejective,total = params['add_new_feature'], params["objective"],params['total'];
        total_x_train,total_y_train = None,None;
        if(obejective=="category" or obejective=="binary_catagory"):
            single = True;
        min_thread_value = params['min_thread_value'];
        benchmark_data, farm_feature = res_data['benchmark_data'], res_data['farm_feature']
        power_feature, wind_feature = res_data['power_feature'], res_data['wind_feature']
        #train_Power_Data = res_data['train_datas'][res_data['train_datas']].copy()
        train_Power_Data = res_data['train_datas']

        # |   #&  (~ train_Power_Data['wp_hn_1'].isnull())
        train_Power_Data = train_Power_Data.groupby('date').head(1).copy()
        #train_Power_Data = pd.DataFrame({'date': pd.unique(train_Power_Data['date'])}).merge(train_Power_Data,how='left')
        #train_Power_Data = train_Power_Data[  (train_Power_Data['catagory']==1) | ( ((train_Power_Data['Flag'] == False) & (~ train_Power_Data['wp_hf_1'].isnull()))) ].copy()
        #train_Power_Data = train_Power_Data[ (~ train_Power_Data['wp_hn_1'].isnull())].copy() # & (~ train_Power_Data['wp_hf_1'].isnull())
        train_Power_Data.sort_values(['date', 'Farm', 'dist'], ascending=[1, 1, 0], inplace=True)
        y_features = ['power'];
        if ( obejective == "category"):
            train_Power_Data['category_power'] = train_Power_Data['power'].apply(lambda x: round( (number_catagory-1) * x))
            y_features = ['category_power']
        elif( obejective=="binary_catagory"  ):
            train_Power_Data['category_power'] = train_Power_Data['power'].apply(lambda x: 0 if x < min_thread_value else 1);
            y_features = ['category_power']
        res_Data = dict()
        groups_data = train_Power_Data.groupby(train_Power_Data['Farm'])
        sequence_length = sequence_length - 1;
        column_titles = ['ws.angle'];
        if addfeature==False:
            column_titles=[]
        else:
            column_titles = ['ws.angle'] if(offset_wind) else column_titles;
        x_features = power_feature + column_titles + ['power']; #+ wind_feature
        print(x_features);
        res_data['x_features'] = x_features
        res_data['y_features'] = y_features;
        for idx, group_item in tqdm.tqdm(groups_data):
            farm_id = farm_feature[idx - 1]
            train_data,x_Datas, y_Datas  = None, None, None;
            res_Data[farm_id] = {};
            res_Data[farm_id]['power_Data'] = group_item.copy();
            x_Datas,y_Datas = None,None;
            for catagory, sub_items in group_item.groupby('catagory'): #[ (group_item['catagory']==1)]
                train_data = sub_items.copy();
                train_data.sort_values(['date'], ascending=[1], inplace=True)
                train_data = train_data.reset_index(drop=True);
                train_data = train_data[train_data['Flag'] == True];
                X_data, Y_data = get_seqence_data(train_data,sequence_length,offset,seq_interval,x_features,y_features,offset_time = timedelta(hours=(sequence_length + offset)))
                x_Datas = X_data if x_Datas is None else np.concatenate((x_Datas, X_data), axis=0);
                y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0);
            if (obejective == "category"):
                y_Datas = np_utils.to_categorical( y_Datas, num_classes= number_catagory);
            if (total):
                total_x_train = x_Datas if total_x_train is None else np.concatenate( [ total_x_train,x_Datas],axis=0 );
                total_y_train = y_Datas if total_y_train is None else np.concatenate([ total_y_train,y_Datas], axis=0 );
            x_train, x_test, y_train, y_test = train_test_split(x_Datas, y_Datas, test_size=1 - train_val_split,
                                                                    random_state=0);
            res_Data[farm_id]['x_train'], res_Data[farm_id]['y_train'], res_Data[farm_id]['x_test'], \
            res_Data[farm_id]['y_test'] = x_train, y_train, x_test, y_test
        if(total):
            res_Data['total']={};
            total_x_train, total_x_test, total_y_train, total_y_test = train_test_split(total_x_train, total_y_train, test_size=1 - train_val_split,
                                                                random_state=0);
            res_Data['total']['x_train'], res_Data['total']['y_train'], res_Data['total']['x_test'], \
            res_Data['total']['y_test'] = total_x_train, total_y_train, total_x_test, total_y_test
        return res_Data;
    except Exception as e:
        print(e);
        return None;
"""

def Multi_Wind_Power_Process(x_features,train_datas,params,is_add_future=None,single = None, offset = 48,reverse=None):
    try:
        sequence_length,train_val_split = params['sequence_length'],params['train_set_fraction'];
        seq_interval,number_catagory,offset_wind = params['seq_interval'],params['number_catagory'],params['offset_wind'];
        addfeature,obejective,total = params['add_new_feature'], params["objective"],params['total'];
        total_x_train,total_y_train = None,None;
        if(obejective=="category" or obejective=="binary_catagory"):
            single = True;
        min_thread_value = params['min_thread_value'];
        #benchmark_data = train_datas[train_datas['Flag']==False]
        #train_Power_Data = res_data['train_datas'][res_data['train_datas']].copy()
        #train_Power_Data = train_datas
        # |   #&  (~ train_Power_Data['wp_hn_1'].isnull())
        train_Power_Data = train_datas.copy()
        #train_Power_Data = pd.DataFrame({'date': pd.unique(train_Power_Data['date'])}).merge(train_Power_Data,how='left')
        #train_Power_Data = train_Power_Data[  (train_Power_Data['catagory']==1) | ( ((train_Power_Data['Flag'] == False) & (~ train_Power_Data['wp_hf_1'].isnull()))) ].copy()
        #train_Power_Data = train_Power_Data[ (~ train_Power_Data['wp_hn_1'].isnull())].copy() # & (~ train_Power_Data['wp_hf_1'].isnull())
        train_Power_Data.sort_values(['date', 'Farm', 'dist'], ascending=[1, 1, 0], inplace=True)
        y_features = ['power'];
        res_Data = dict()
        res_Data['x_features'] = x_features
        res_Data['y_features'] = y_features
        groups_data = train_Power_Data.groupby(train_Power_Data['Farm'])
        sequence_length = sequence_length - 1
        predict_datas = []
        for idx, group_item in tqdm.tqdm(groups_data):
            #train_data,x_Datas, y_Datas  = None, None, None;
            x_Datas,y_Datas = None,None;
            sub_group_item = group_item.groupby('date').head(1).reset_index(drop=True).copy()
            print(sub_group_item.shape)
            train_group_item = sub_group_item[sub_group_item['Flag'] == True]
            loc_values = sub_group_item[sub_group_item['Flag']==False].index.values
            predict_data = np.array([sub_group_item.loc[iter - sequence_length: iter-1, x_features].values for iter in loc_values[range(0,len(loc_values),48)]])
            predict_datas.append(predict_data)
            for catagory, sub_items in train_group_item.groupby('catagory'): #[ (group_item['catagory']==1)]
                train_data = sub_items.copy();
                train_data.sort_values(['date'], ascending=[1], inplace=True)
                train_data = train_data.reset_index(drop=True);
                train_data = train_data[train_data['Flag'] == True];
                X_data, Y_data = get_seqence_data(train_data,sequence_length,offset,seq_interval,x_features,y_features,offset_time = timedelta(hours=(sequence_length + offset)))
                x_Datas = X_data if x_Datas is None else np.concatenate((x_Datas, X_data), axis=0);
                y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0);
                total_x_train = x_Datas if total_x_train is None else np.concatenate( [ total_x_train,x_Datas],axis=0 );
                total_y_train = y_Datas if total_y_train is None else np.concatenate([ total_y_train,y_Datas], axis=0 );
            print("Group: ", idx, total_x_train.shape)
        total_x_train, total_x_test, total_y_train, total_y_test = train_test_split(total_x_train, total_y_train, test_size=1 - train_val_split,random_state=0);
        res_Data['x_train'], res_Data['y_train'], res_Data['x_test'], res_Data['y_test'] = total_x_train, total_y_train, total_x_test, total_y_test
        res_Data['predict_datas'] = predict_datas
        return res_Data;
    except Exception as e:
        print(e);
        return None;


"""
# 真实分组，36-》84
def Multi_Wind_Power_Process(x_features,train_datas,params):
    try:
        sequence_length,train_val_split = params['sequence_length'],params['train_set_fraction']
        #power_feature, wind_feature = res_data['power_feature'], res_data['wind_feature']
        res_Data = dict()
        sequence_length = sequence_length - 1
        seq_interval =  12  # 12
        offset = 48
        train_Power_Data = train_datas
        y_features = ['power']
        x_features = x_features
        groups_data = train_Power_Data.groupby(train_Power_Data['Farm'])
        res_Data['x_features'] = x_features
        res_Data['y_features'] = y_features
        res_Data['farm_feature'] = []
        x_Datas, y_Datas = None, None;
        for id, item in tqdm.tqdm(groups_data): # For Each Farm
            farm = "wp"+str(id)
            res_Data[farm] = {}
            res_Data['farm_feature'].append(farm)
            group_item = item.copy()
            group_item.sort_values(['start', 'new_date'], ascending=[1, 1], inplace=True)
            group_item.reset_index(drop=True,inplace=True)
            temp = group_item[group_item['catagory'] == 1].copy()
            start_loc = list(range(0, temp.shape[0]-sequence_length, seq_interval))
            end_loc = list(range(36, temp.shape[0], seq_interval))
            valid_start_loc,valid_end_loc = [],[]
            for idx in range(0,len(start_loc),1):
                end_idx = temp.index.values[end_loc[idx]]
                start_idx = temp.index.values[start_loc[idx]]
                predict_time = temp.loc[end_idx, 'date']
                start_time = temp.loc[start_idx, 'date'] + timedelta(hours=sequence_length)
                temp_item = group_item.loc[end_idx:(end_idx + offset - 1)]
                null_count = temp_item[temp_item['Flag']==True].shape[0] - 48
                if predict_time == start_time and null_count==0:
                    valid_start_loc.append(start_loc[idx])
                    valid_end_loc.append(end_loc[idx])
            predict_data = [ group_item.loc[ loc:(loc + offset-1), y_features].values for loc in temp.index.values[valid_end_loc]]
            train_predict_data = [group_item.loc[loc:(loc + offset - 1), x_features].values for loc in temp.index.values[valid_end_loc]]
            catagory_items = group_item.groupby('catagory')
            test_data = group_item[group_item['Flag']==False]
            test_loc = test_data.index.values[range(0, test_data.shape[0], 48)]
            test_matrix = [test_data.loc[ test_data.index.values[loc:(loc+offset)],x_features].values for loc in range(0, test_data.shape[0], 48) ]
            res_Data[farm]['test'] = []
            for catagory, sub_items in catagory_items:
                train_data = sub_items.copy()
                train_matrix = [ train_data.loc[train_data.index.values[loc:(loc + sequence_length)],x_features].values  for loc in valid_start_loc]
                start_matrix =  [train_data.loc[(test_loc[0]- 36 * (catagory + 1)):(test_loc[0]-1),x_features].values]
                predict_matrix = np.concatenate([start_matrix,[ train_data.loc[(idx- 36 * 4 ):(idx-1),x_features].values for idx in test_loc[1:]]],axis=0)
                predict_matrix = np.concatenate([predict_matrix,test_matrix],axis=1)
                X_data, Y_data = np.concatenate([train_matrix,train_predict_data],axis=1), np.array(predict_data)
                x_Datas = X_data if x_Datas is None else np.concatenate((x_Datas, X_data), axis=0)
                y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0)
                res_Data[farm]['test'].append(predict_matrix)
        y_Datas = y_Datas.reshape(-1,y_Datas.shape[1])
        x_train, x_test, y_train, y_test = train_test_split(x_Datas, y_Datas, test_size=1 - train_val_split,random_state=0)
        res_Data['x_train'], res_Data['y_train'], res_Data['x_test'],res_Data['y_test'] = \
                x_train, y_train, x_test, y_test
        return res_Data
    except Exception as e:
        print(e)
        return None
"""

def get_power_train_data(train_val_data,sequence_length,offset=48,single=None):
    X_data = []; Y_data = [];
    number_data = len(train_val_data);
    x_idxs_power= [-2];
    x_idxs_wind = list(range(0, train_val_data.shape[1]-2));
    y_idxs = [-1];
    if( single is None):
        if( sequence_length> offset):
            X_data.append(np.concatenate((train_val_data[ offset :  (sequence_length+offset), x_idxs_wind],
                                          train_val_data[ 0: sequence_length, x_idxs_power]), axis=1));
            Y_data.append(train_val_data[ sequence_length :  (sequence_length+offset) , y_idxs[0] ] );
        else:
            X_data.append(np.concatenate((train_val_data[sequence_length:  (2 * sequence_length), x_idxs_wind],
                                          train_val_data[0: sequence_length, x_idxs_power]), axis=1));
            Y_data.append(train_val_data[sequence_length:  (sequence_length + offset), y_idxs[0]]);
    else:
        for i in range( number_data - sequence_length):
            X_data.append(np.concatenate((train_val_data[ i + 1: (i + sequence_length + 1), x_idxs_wind],
                                          train_val_data[i : (i + sequence_length), x_idxs_power]
                                          ), axis=1));
            Y_data.append( train_val_data[ i + sequence_length, y_idxs]);
    # 训练数据 # 测试数据
    X_data = np.array(X_data);
    Y_data = np.array(Y_data);
    return X_data, Y_data;

def get_power_train_data2(train_val_data,sequence_length,offset=48,single=None):
    X_data = []; Y_data = [];
    number_data = len(train_val_data);
    x_idxs_power= [-2];
    x_idxs_wind = list(range(0, train_val_data.shape[1]-2));
    y_idxs = [-1];
    if( single is None):
        if( sequence_length> offset):
            X_data.append(np.concatenate((train_val_data[ 0 :  (sequence_length), x_idxs_wind],
                                          train_val_data[ offset: (sequence_length+offset), x_idxs_power]), axis=1));
            Y_data.append( train_val_data[ 0 :  (offset) , y_idxs[0] ]);
        else:
            X_data.append(np.concatenate((train_val_data[(offset - sequence_length):  (offset), x_idxs_wind],
                                          train_val_data[ offset: (offset+sequence_length), x_idxs_power]), axis=1));
            Y_data.append(train_val_data[0: offset, y_idxs[0]]);
    else:
        for i in range( number_data - sequence_length):
            X_data.append(np.concatenate((train_val_data[ i + 1: (i + sequence_length + 1), x_idxs_wind],
                                          train_val_data[i : (i + sequence_length), x_idxs_power]
                                          ), axis=1));
            Y_data.append( train_val_data[ i + sequence_length, y_idxs]);
    # 训练数据 # 测试数据
    X_data = np.array(X_data);
    Y_data = np.array(Y_data);
    return X_data, Y_data;


def Test_Multi_Wind_Power_Process(res_data,params):
    try:
        sequence_length,train_val_split = params['sequence_length'],params['train_set_fraction'];
        seq_interval,number_catagory = params['seq_interval'],params['number_catagory'];
        addfeature,obejective,total = params['add_new_feature'], params["objective"],params['total'];
        single = None; total_x_train,total_y_train = None,None;
        if(obejective=="category" or obejective=="binary_catagory"):
            single = True;
        offset = 48; min_thread_value = params['min_thread_value'];
        benchmark_data, farm_feature = res_data['benchmark_data'], res_data['farm_feature']
        power_feature, wind_feature = res_data['power_feature'], res_data['wind_feature']
        train_Power_Data = res_data['train_datas'].copy()
        train_Power_Data.sort_values(['date', 'Farm', 'dist'], ascending=[1, 1, 0], inplace=True)
        y_features = ['power'];
        if ( obejective == "category"):
            train_Power_Data['category_power'] = train_Power_Data['power'].apply(lambda x: round(number_catagory * x))
            y_features = ['category_power']
        elif( obejective=="binary_catagory"  ):
            train_Power_Data['category_power'] = train_Power_Data['power'].apply(lambda x: 0 if x < min_thread_value else 1);
            y_features = ['category_power']
        res_Data = dict()
        groups_data = train_Power_Data.groupby(train_Power_Data['Farm'])
        sequence_length = sequence_length - 1
        column_titles = ['wind_power_' + str(subidx + 1) + 'h' for subidx in range(0, 4)]
        if addfeature==False:
            column_titles=[]
        x_features = power_feature + wind_feature + column_titles + ['power'];
        res_data['x_features'] = x_features
        res_data['y_features'] = y_features;
        for idx, group_item in groups_data:
            farm_id = farm_feature[idx - 1]
            train_data,x_Datas, y_Datas  = None, None, None;
            if addfeature==False:
                train_data = group_item[group_item['catagory']==1].copy();
                forcast_data = group_item.copy();
                forcast_data.sort_values(['start', 'dist', 'date'], ascending=[1, 1, 1], inplace=True);
            else:
                group_catagory = group_item.groupby(group_item['catagory'])
                for subidx, sub_catagory in group_catagory:
                    column_title = column_titles[subidx - 1];
                    sub_catagory = sub_catagory.rename(columns={"ws.angle": column_title});
                    train_data = sub_catagory if train_data is None else \
                        train_data.merge(sub_catagory[['date', column_title]], left_on='date',right_on='date', how='right');
            train_data.sort_values(['date'], ascending=[1], inplace=True)
            train_data = train_data.reset_index(drop=True);
            res_Data[farm_id] = {};
            res_Data[farm_id]['power_Data'] = train_data;
            train_data = train_data[train_data['Flag'] == True];

            for iter in tqdm.tqdm(range(0,(train_data.shape[0] - sequence_length - offset),seq_interval)):
                start_time = forcast_data.loc[ forcast_data.index.values[iter+sequence_length],'date'];
                #loc = train_data[train_data['date']==start_time].index.values[0];
                #forcast_time = forcast_data[ forcast_data.index.values[iter+sequence_length + offset],'date'] - timedelta(hours= (sequence_length + offset)) ;
                #start_time = train_data.loc[ train_data.index.values[loc-sequence_length], 'date'];
                pred_start_time = train_data.loc[train_data.index.values[iter +  sequence_length + offset ], 'date']\
                                  - timedelta(hours= (sequence_length + offset));
                start_time = train_data.loc[train_data.index.values[iter], 'date'];
                if (start_time == pred_start_time):
                    train_matrix = pd.concat([train_data[iter:(iter+sequence_length) ],forcast_data[forcast_data['start'] == (start_time - timedelta(hours=1))]],axis=0);
                    train_matrix = train_matrix[x_features + y_features ].values;
                    X_data, Y_data = get_power_train_data(train_matrix, sequence_length,offset,single=single);
                    x_Datas = X_data if x_Datas is None else np.concatenate((x_Datas, X_data), axis=0);
                    y_Datas = Y_data if y_Datas is None else np.concatenate((y_Datas, Y_data), axis=0);
                else:
                    continue;
            if (obejective == "category"):
                y_Datas = np_utils.to_categorical( y_Datas, num_classes= number_catagory);
            if (total):
                total_x_train = x_Datas if not total_x_train else np.concatenate( [ x_Datas,total_x_train],axis=0);
                total_y_train = y_Datas if not total_y_train else np.concatenate([ y_Datas, total_y_train], axis=0);
            x_train, x_test, y_train, y_test = train_test_split(x_Datas, y_Datas, test_size=1 - train_val_split,
                                                                random_state=0);
            res_Data[farm_id]['x_train'], res_Data[farm_id]['y_train'], res_Data[farm_id]['x_test'], \
            res_Data[farm_id]['y_test'] = x_train, y_train, x_test, y_test
        if(total):
            res_Data['total']={};
            total_x_train, total_x_test, total_y_train, total_y_test = train_test_split(total_x_train, total_y_train, test_size=1 - train_val_split,
                                                                random_state=0);
            res_Data['total']['x_train'], res_Data['total']['y_train'], res_Data['total']['x_test'], \
            res_Data['total']['y_test'] = total_x_train, total_y_train, total_x_test, total_y_test
        return res_Data;
    except Exception as e:
        print(e);
        return None;


def get_train_data(train_val_data,sequence_length):
    X_data = []; Y_data = [];
    number_data = len(train_val_data);
    x_idxs = list(range(8, train_val_data.shape[1]));
    farm_idxs = list(range(1,8));
    y_idxs = [0];
    sequence_length = sequence_length - 1;
    for i in range(number_data - sequence_length):
        X_data.append(np.concatenate((train_val_data[ i + 1: (i + sequence_length + 1), x_idxs],
                                      train_val_data[i: (i + sequence_length), farm_idxs],
                                      #train_val_data[i : (i + sequence_length), y_idxs]
                                      ), axis=1));
        Y_data.append(train_val_data[i + sequence_length, y_idxs]);
    # 训练数据 # 测试数据
    X_data = np.array(X_data);
    Y_data = np.array(Y_data);
    return X_data, Y_data;

"""
def get_predict_data(x_features,benchmark_data,train_data,wind_train_Datas,wind_datas, sequence_length):
    start_time = benchmark_data['date'];
    sequence_length = sequence_length-1;
    temp_dict = {};
    for idx,item in enumerate(start_time):
        offset = idx % sequence_length;
        new_data = benchmark_data[benchmark_data['date'] == item];
        if (offset == 0):
            data_array = train_data[train_data['date'] < item];
            data_array = data_array[-sequence_length:];
            data_array = data_array.drop(columns=['date','Flag']).values; # 保证data_array农场数据;
        for farmidx, feacture in enumerate(x_features):
            if (offset == 0):
                temp_dict[feacture] = {};
                wind_data = wind_datas[farmidx];
                wind_train_Data = wind_train_Datas[farmidx];
                # temp_wind_data 未来的气象数据
                temp_wind_data = wind_data[wind_data['date']<item];
                temp_dict[feacture]['temp_wind_data'] = temp_wind_data[-sequence_length:];
                # wind_array 前47天数据;
                wind_array = wind_train_Data[wind_train_Data['new_date']<item][-sequence_length+1:];
                temp_dict[feacture]['wind_array'] = wind_array.drop(columns=['date', 'hors', 'new_date']).values;
            temp_wind_data = temp_dict[feacture]['temp_wind_data'];
            wind_array = temp_dict[feacture]['wind_array'];

            temp = temp_wind_data[temp_wind_data['new_date']== item ]; # 预测的气象数据
            temp = temp.drop(columns=['date', 'hors', 'new_date']).values;
            wind_array = np.concatenate((wind_array, temp), axis=0);
            temp_dict[feacture]['wind_array'] = wind_array;

            x_predict = np.concatenate((data_array[-sequence_length:], wind_array[-sequence_length:]), axis=1)[-sequence_length:];
            y_predict = 0;# model.predict(x_predict);
            new_data[feacture] = y_predict;
        # 更新 data_array = [];
        train_data = pd.concat([ train_data, new_data.drop(columns=['id']) ]);
        train_data.sort_values(['date'], ascending=[1], inplace=True);
        # train_data.loc[(train_data['date'] == 'item'), feacture] = y_predict;
        # train_data[train_data['date'] == item][feacture] = y_predict;
        # print(train_data[train_data['date'] == item]);
        #train_data[feacture] = np.where(train_data['date'] == item, 0);
        #print(train_data[train_data['date'] == item]);
        benchmark_data[benchmark_data['date'] == item] = new_data;
        # print(benchmark_data[idx]);
        #print(train_data[feacture]);
        temp_data_array = train_data[train_data['date'] == item].drop(columns=['date']).values;
        data_array = np.concatenate( (data_array,temp_data_array),axis=0);
"""

# get_power_train_data(y,12,offset=12,single=None);

