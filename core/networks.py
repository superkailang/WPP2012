import os
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, BatchNormalization,Bidirectional, Dropout,Flatten,Input
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import LearningRateScheduler,EarlyStopping, ModelCheckpoint,TensorBoard
import datetime as dt
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import numpy as np;
from sklearn import preprocessing
import re
from core.plot import plot_Predict,plot_Predict_time,plot_history
import pandas as pd;
import tqdm
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import time;
from keras.layers import Layer
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold,train_test_split

"""
def create_model(unit_node = 48):
    NN = NN_Net(params,input);
    return NN.model;
epoches = np.array([100,20,300]);
batchs = np.array([512,256,128,64]);
init = ['glorot_uniform', 'normal', 'uniform'];
param_grid = dict(epochs=epoches,batch_size=batchs)
models = KerasRegressor(build_fn = create_model);
#cross_val_score(model)
grid = GridSearchCV(estimator=models,param_grid=param_grid);
grid_result = grid.fit(X_TRAIN,Y_TRAIN);
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
grid_result.best_params_;
grid_result.cv_results_;
grid_result.
"""


class Reverse_Layer(Layer):
    def __init__(self,
                 **kwargs):
        super(Reverse_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.axis = len(input_shape)-1;
        super(Reverse_Layer,self).build(input_shape);

    def call(self, inputs):
        output = K.reverse(inputs,axes = self.axis );
        return output

weight_decay = 1e-4

def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001

def root_mean_squared_error(y_true, y_pred):
    target_shape = tf.stack([-1])
    tf.reshape(y_true,target_shape)
    t1 = K.sqrt(K.mean(K.square(tf.reshape(y_true,target_shape) - tf.reshape(y_pred,target_shape)), axis=-1) + 1e-8)
    return t1


def root_mean_category(y_true,y_pred):
    y_pred_arg, y_true_arg = K.argmax(y_pred,axis=1),K.argmax(y_true,axis=1);
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1) + 1e-8);

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

def rever_matrix(x):
    x2 = K.reverse(x, axes= len(x.shape)-1 )
    print( x2 )
    return x2


"""
# The following calls are equivalent.
#value_3 = my_func(np.array([[[2, 0, 0,0], [0, 0, 1,0 ],[0,0,0,1],[1,0,0,0]],[[4, 0, 0,0], [0, 0, 1,0 ],[0,0,0,1],[1,0,0,0]]], dtype=np.float))

y_pre = [[3,2,1],[4,3,2]]
y = [[1,2,3],[5,4,3]]
value_2 = tf.convert_to_tensor( y_pre, dtype=np.float)
value_1 = tf.convert_to_tensor( y, dtype=np.float)
#value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
d = root_mean_squared_error(value_2,value_1)
#d = rever_matrix(value_3)
#c = root_mean_category(value_2,value_3);
# Launch the graph in a session.
session = tf.InteractiveSession()
# Evaluate the tensor `c`.
#print("输入:")
#print(session.run(value_3))
print("反转输出:")
print(session.run(d))
print("rmse: ",mean_squared_error(y,y_pre)**0.5)
"""


class NN_Net:

    def __init__(self,params,input,reverse = None):
        self.params = params;
        self.batch_sizes = params['batch_size'];
        self.model = Sequential();
        self.stack_layers(params['layer'],params['NN_model'],reverse=reverse);
        self.len_train = len(input);
        #self.build_model(shape);

    def stack_layers(self,layers,model_param,reverse = None,kernel_initializer='normal'): #glorot_uniform
        for layer in layers:
            # create the layer
            if layer['type'] == 'input':
                input_dim = layer['input_dim'] if 'input_dim' in layer else None
                size = layer['size'] if 'size' in layer else None
                l = Dense(layer['size'], input_dim=input_dim, kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(0.001), name=layer.get('name'))
            elif layer['type'] == 'softplus_reg':
                l = Dense(layer['size'], activation='softplus', kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(0.001), name=layer.get('name'))
            elif layer['type'] == 'softmax':
                l = Dense(layer['size'], activation='softmax', kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(0.001), name=layer.get('name'))
            elif layer['type'] == 'tanh':
                l = Dense(layer['size'], activation='tanh', kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(0.001), name=layer.get('name'))
            elif layer['type'] == 'relu':
                l = Dense(layer['size'], activation='relu', kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(0.001), name=layer.get('name'))
            elif layer['type'] == 'selu':
                l = Dense(layer['size'], activation='selu', kernel_initializer=kernel_initializer,
                          kernel_regularizer=l2(0.001), name=layer.get('name'))
            elif layer['type'] == 'BatchNormalization':
                l = BatchNormalization(name=layer.get('name'))
            elif layer['type'] == 'dropout':
                l = Dropout(layer['rate']);
            elif layer['type'] == 'flatten':
                l = Flatten();
            elif layer['type'] == 'lstm':
                go_backwards = layer['go_backwards'] if 'go_backwards' in layer else False
                #reverse =  True if go_backwards else None
                input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
                input_dim = layer['input_dim'] if 'input_dim' in layer else None
                #net_input = Input(shape=(input_timesteps, input_dim), name='net_input')
                return_seq = layer['return_seq'] if 'return_seq' in layer else None
                l =  LSTM(layer['neurons'], recurrent_dropout=0.2,input_shape=(input_timesteps, input_dim),
                         kernel_initializer='lecun_normal', return_sequences=return_seq,go_backwards=go_backwards);
            elif layer['type'] == 'dense':
                l = Dense( layer['size'], activation =layer['activation']);
            else:
                raise ValueError("Invalid layer type '{}'".format(layer['type']))
            self.model.add(l);
        if( reverse is not None):
            self.model.add( Reverse_Layer(name='reverse'));
        for name,item in model_param.items():
            if( name == "optimizer"):
                if( item['type']=="sgd" ):
                    optimizer = optimizers.SGD(lr=item['lr'], momentum=0.9, nesterov=True, clipnorm=1.0);
                elif(item['type'] == "sgd"):
                    optimizer = "adm";
            elif( name =="metrics"):
                metrics = item['metrics']+ [root_mean_squared_error];
            elif(name=="loss"):
                loss = item['loss'] if item['loss'] is not None else root_mean_squared_error;
        self.model.compile(loss = loss, optimizer=optimizer, metrics=metrics);
        print(self.model.summary());

    def build_model(self,input_shape):
        # MLP 模型;
        model1 = Sequential()
        model1.add(LSTM(48,recurrent_dropout=0.2, input_shape= input_shape , kernel_initializer='lecun_normal', return_sequences=False))
        model1.add(Dropout(0.2))
        model1.add(Dense(100,activation='softmax'));
        model1.add(Dropout(0.5))
        # model1.add(Dense(20,activation='elu'))
        # model1.add(Dense(1,activation='linear'))
        #sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True, clipnorm=1.0);
        model1.compile(loss= root_mean_squared_error, optimizer='adam',metrics=['accuracy',root_mean_squared_error])
        self.model = model1;
        """
        self.model.add(Flatten( input_shape=(10,2)))
        self.model.add(Dense(50, kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(0.0001));
        self.model.add(Dense(20, activation='linear'))
        self.model.add(Dropout(0.0001));
        self.model.add(Dense(10, activation='linear'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam',metrics=['mse']);
        print(self.model.summary());
        """

    def train(self, x, y,x_test,y_test,name,load_models=False):
        epochs = self.params['epoch_number'];
        batch_size = self.params['batch_size'];
        save_dir = self.params['save_dir'];
        if (load_models):
            save_h5 = os.path.join(save_dir,'25092019-091837_wsPower3.h5');
            self.model.load_weights(save_h5)
            return
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        save_fname = os.path.join(save_dir, '%s_%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'),name))
        # set callback
        tb_cb = TensorBoard(log_dir= save_dir, histogram_freq=0);
        change_lr = LearningRateScheduler(scheduler);
        callbacks = [tb_cb,change_lr,
            EarlyStopping(monitor='loss', patience=20),
            ModelCheckpoint(filepath=save_fname, save_best_only=False, mode='auto', period=10),
        ];
        self.history = self.model.fit(x,y,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)
        self.model.save(save_fname)
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        scores = self.model.evaluate(x_test, y_test, verbose=0);
        predicted_data = self.model.predict(x_test);
        #plot_Predict(y_test, predicted_data,name);
        #print("Accuracy: %.2f%%" % (scores));
        #self.show_histoty(epochs,name);

    def load_model(self,name):
        save_dir = self.params['save_dir'];
        filelist = self.file_name( save_dir);
        find_rules = '.*'+ name + '.h5$';
        filenames = [re.findall(find_rules, file) for i, file in enumerate(filelist['name']) if
                     (file.endswith((name + '.h5')))];
        save_h5 = os.path.join(save_dir, filenames[0][0]);
        self.model= load_model(save_h5,custom_objects={'root_mean_squared_error': root_mean_squared_error});

    def evaluat(self,x_test,y_test,y,name):
        #y_test = self.scaler.transform(y_test);
        #y_test = (y_test - self.mean()) / (self.max - self.min);
        model_dir = self.params['save_dir']
        Jsonfile = model_dir + '/22052019-183011-e2000_wind_power.h5'
        model  = load_model(Jsonfile)
        model  = model.load_weights(Jsonfile)
        #scores = self.model.evaluate(x_test, y_test, verbose=0);
        #print("Accuracy: %.2f%%" % (scores));
        predicted_data = self.model.predict(x_test)
        print(predicted_data)
        plot_Predict(y_test,predicted_data)

    def show_histoty(self,epochs,name):
        history_dict = self.history.history
        #acc = history_dict['acc']
        #val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = np.arange(0, epochs)
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        plot_history(epochs,loss,val_loss,name)

    def predict(self,data,time,params,name,x_test,y_test):
        model_dir = self.params['save_dir'];
        filelist = self.file_name(model_dir);
        find_rules = '-(.+?)-.*.h5$';
        filenames = [ re.findall(find_rules,file) for i, file in enumerate(filelist['name']) if( file.endswith( ( name+'.h5')))];
        filename = name + '.h5';
        Jsonfile = model_dir+'/' + filename;
        to_file = model_dir+'/' + "NetModel.png";
        model = load_model(Jsonfile);
        model.load_weights(Jsonfile);
        # model.compile(loss='mse', optimizer='adam', metrics=['mse']);
        # print(self.model.summary());
        #plot_model(self.net, to_file='', show_shapes=True);
        sequence_length = params['sequence_length'];
        nsteps = len(data)-sequence_length;
        y_real=[]; y=[]; y2=[]
        y_2 = model.predict(x_test);
        y_max = params['max_features'][name];
        for i in range(nsteps):
            x_data,y_data = self.get_sequence_data(data,i,sequence_length);
            y_pred = model.predict(x_data);
            y.append(y_pred[0]*y_max);
            y2.append(y_pred[0][0]*y_max)
            y_real.append(y_data*y_max);
            data[ i + sequence_length,1]= y_pred;
        #plot_Predict_time(time['time'],y_real,y,name)
        data_frame = pd.DataFrame({'time': time['time'], 'predict': y2});
        #print(name + " 预测MAPE: " + self.mean_absolute_percentage_error(y_real, y));
        return data_frame;

    def predict2(self,data,time,params,name,x_test,y_test):
        model_dir = self.params['save_dir'];
        filelist = self.file_name(model_dir);
        find_rules = '-(.+?)-.*.h5$';
        filenames = [ re.findall(find_rules,file) for i, file in enumerate(filelist['name']) if( file.endswith( ( name+'.h5')))];
        filename ='23052019-093203_wind_power_checkpoint-40-0.01.h5';
        Jsonfile = model_dir+'/' + filename;
        to_file = model_dir+'/' + "NetModel.png";
        model = load_model(Jsonfile);
        model.load_weights(Jsonfile);
        #plot_model(model, to_file=to_file, show_shapes=True);
        y_max = params['max_features']['power'];
        sequence_length = params['sequence_length'];
        y_predict = model.predict(x_test);
        y_predict = y_predict* y_max;
        y_test = y_test *y_max;
        #plot_Predict_time(time['time'],y_test,y_predict,name)
        print(name + " 预测MAPE: " + self.mean_absolute_percentage_error(y_test, y_predict));

    def get_sequence_data(self,data_item, idx,sequence_length):
        X_data = [];
        Y_data = [];
        x_idxs = list(range(0, data_item.shape[1] - 1));
        y_idxs = [data_item.shape[1] - 1];
        X_data.append(np.concatenate((data_item[idx + 1:(idx + sequence_length), x_idxs],
                                          data_item[idx:(idx + sequence_length - 1), y_idxs]), axis=1));
            # X_data.append( data_item[ i:(i+sequence_length),[0,1]] );
        Y_data=(data_item[idx + sequence_length, y_idxs ]);
        # 训练数据 # 测试数据
        X_data = np.array(X_data);
        Y_data = np.array(Y_data);
        return X_data, Y_data;

    def normalized(self,x):
        x = (self.x - self.min)/();

    def reshape_data(self,data):
        shape_dim = data.shape[1]*data.shape[2];
        data = np.reshape(data,(-1,shape_dim));
        return data;

    def file_name(self,file_dir):
        filenames = {
            'file': [],
            'name': []
        };
        for root, dirs, files in os.walk(file_dir):
            print(root)  # 当前目录路径
            # print(dirs)  # 当前路径下所有子目录
            #print(files)  # 当前路径下所有非目录子文件
            for file in files:
                filenames['file'].append(root + '\\' + file);
                filenames['name'].append(file);
            return filenames;

    def mean_absolute_percentage_error(self,y_true, y_pred):
        # y_true, y_pred = y_true, np.array(y_pred)
        sum = 0; n=0;
        for index, item in enumerate(y_true):
            if( abs(y_true[index])>0):
                sum = sum + abs(y_true[index] - y_pred[index]) / y_true[index];
                n = n+1;
        if(n>0):
            sum = sum[0] / n;
        return str(sum * 100);

def NN_predict_data(model_data,res_data,params,Limit_Number=None):
    sequence_length = params['sequence_length'];
    train_data, benchmark_data = res_data['train_data'], res_data['benchmark_data'];
    wind_datas, wind_train_Datas = res_data['wind_datas'], res_data['wind_train_Data'];

    x_features = ['Farm', 'year', 'month', 'hour'];
    farm_features = res_data['farm_feature'];

    if (Limit_Number is None):
        start_time = benchmark_data['date'];
    else:
        start_time = benchmark_data[:Limit_Number]['date'];
    sequence_length = sequence_length - 1;
    benchmark_data.is_copy = False;
    data_arrays = {};
    temp_dict = {};
    other_dict = {};

    for idx, item in enumerate(tqdm.tqdm(start_time)):
        offset = idx % sequence_length;
        new_data = benchmark_data[benchmark_data['date'] == item].copy();
        for farmidx, feacture in enumerate(farm_features):
            if (offset == 0):
                data_array = train_data[farmidx][train_data[farmidx]['date'] < item];
                data_array = data_array[-sequence_length:];
                other_dict[feacture] = data_array[x_features].values;
                data_array = data_array[ farm_features ].values;
                data_arrays[feacture] = data_array; #取历史数据;

                temp_dict[feacture] = {};
                wind_data = wind_datas[farmidx];
                wind_train_Data = wind_train_Datas[farmidx];
                # temp_wind_data 未来的气象数据
                temp_wind_data = wind_data[wind_data['wind_date'] < item];
                temp_dict[feacture]['temp_wind_data'] = temp_wind_data[-sequence_length:];
                # wind_array 前47天数据;
                wind_array = wind_train_Data[wind_train_Data['new_date'] < item][-sequence_length + 1:];
                temp_dict[feacture]['wind_array'] = wind_array.drop(columns=['wind_date', 'hors', 'new_date']).values;

            temp_wind_data = temp_dict[feacture]['temp_wind_data']; # 预测的气象数据
            wind_array = temp_dict[feacture]['wind_array'];  # 历史气象数据
            other_feature_array  = other_dict[feacture];

            temp = temp_wind_data[temp_wind_data['new_date'] == item];  # 预测的气象数据
            temp = temp.drop(columns=['wind_date', 'hors', 'new_date']).values;
            wind_array = np.concatenate((wind_array, temp), axis=0);
            temp_dict[feacture]['wind_array'] = wind_array;

            x_list = ['date','year', 'month', 'hour'] + [feacture];
            temp_farm = new_data[x_list].copy();
            temp_farm.loc[temp_farm.index._values,'Farm'] = farmidx + 1;
            temp_farm = temp_farm.rename(columns={feacture: "power"});
            new_data_temp =  temp_farm[x_features].values;
            other_feature_array = np.concatenate( (other_feature_array,new_data_temp),axis=0);
            other_dict[feacture] = other_feature_array;

            data_array = data_arrays[feacture];
            x_predict = np.concatenate(( other_feature_array[-sequence_length:,],wind_array[-sequence_length:],data_array[ - sequence_length:]), axis=1)[-sequence_length:];

            try:
                time_feature = ['Farm', 'year', 'month', 'hour'];
                wind_features = list(wind_datas[0].drop(columns=['wind_date', 'hors','new_date']).columns);
                pre_time = new_data.loc[new_data.index.values[0],'date'] - timedelta(hours=1);
                temp_newdata = new_data.copy();
                temp_newdata.loc[:,'Farm'] = farmidx + 1;
                test_X_data = train_data[farmidx][(train_data[farmidx]['date'] <= pre_time) & (
                        train_data[farmidx]['date'] > pre_time - timedelta(hours=sequence_length))][farm_features].values;
                test_time_X_data = train_data[farmidx][(train_data[farmidx]['date'] <= pre_time + timedelta(hours=1)) & (
                        train_data[farmidx]['date'] > pre_time - timedelta(hours=sequence_length - 1))][time_feature].values;
                test_time_X_data = np.concatenate( [test_time_X_data,temp_newdata[time_feature].values],axis=0);
                test_wind_data = wind_train_Datas[farmidx][(wind_train_Datas[farmidx]['new_date'] <= pre_time + timedelta(hours=1)) &
                                           (wind_train_Datas[farmidx]['new_date'] > pre_time - timedelta(hours=sequence_length - 1))][wind_features].values;
                test_X_data = np.concatenate([test_time_X_data, test_wind_data, test_X_data], axis=1);
                print("Test For X_Data RMSE :{}".format(mean_squared_error(test_X_data, x_predict )));
            except Exception as e:
                print(e);
            if (pd.isnull(np.array(x_predict, dtype=float)).sum() > 0):
                print("Input NAN {}".format(idx));

            y_predict = model_data[feacture]['model'].model.predict(  np.array([x_predict]));


            if (pd.isnull(np.array(y_predict, dtype=float)).sum() > 0):
                print("Input NAN {}".format(idx));

            new_data.loc[new_data.index._values[0],feacture] = y_predict[0][0];
            temp_farm.loc[temp_farm.index.values[0],'power'] = y_predict[0][0];

        benchmark_data.loc[benchmark_data['date'] == item] = new_data;
        # 更新DataArray;
        for farmidx, feacture in enumerate(farm_features):
            temp_data_array = new_data[ farm_features].values;
            data_arrays[feacture] = np.concatenate((data_arrays[feacture], temp_data_array), axis=0);
            temp_farm = new_data.drop(columns=['id','Flag']).copy();
            temp_farm.loc[temp_farm.index._values, 'Farm'] = farmidx + 1;
            temp_farm.loc[temp_farm.index.values[0], 'power'] = new_data[feacture].values;
            train_data[farmidx] = pd.concat([train_data[farmidx], temp_farm],sort=False);
            train_data[farmidx].sort_values(['date'], ascending=[1], inplace=True);
    test_data = benchmark_data.drop(columns=['id','year', 'month', 'hour']);
    return test_data;

def NN_predict_data_group(model_data,res_data,params,Limit_Number=None):
    try:
        sequence_length = params['sequence_length'] - 1;
        x_features = res_data['x_features']
        farm_features = res_data['farm_feature'];
        bench_mark = None;
        # measure process time
        t0 = time.clock()
        for farm_feature in farm_features:
            power_Data = model_data[farm_feature]['power_Data'];
            predict_datas = power_Data[power_Data['Flag'] == False];
            if (Limit_Number is None):
                Limit_Number = predict_datas.shape[0];
            for index, data_item in tqdm.tqdm(predict_datas[:Limit_Number].iterrows(),total=Limit_Number):
                index_loc = index;
                pred_start_time = power_Data.loc[index_loc-sequence_length,'date']+ timedelta(hours= sequence_length);
                start_time = data_item['date'];
                assert pred_start_time ==start_time;
                x_predict = np.concatenate([ power_Data[index_loc-sequence_length+1:index_loc+1][x_features[:-1]].values,
                                             power_Data[index_loc-sequence_length:index_loc][x_features[-1:]].values],axis=1);
                y_predict = model_data[farm_feature]['model'].model.predict(  np.array([x_predict]));
                predict_datas.loc[index,'power'] = y_predict[0][0];
                power_Data.loc[index_loc,'power'] = y_predict[0][0];
            predict_item = predict_datas[['date', 'power']].rename(columns={"power": farm_feature});
            bench_mark = bench_mark.merge(predict_item, left_on='date',right_on='date') \
                if bench_mark is not None else predict_item
        print("Time: ",time.clock()-t0);
        return bench_mark;
    except Exception as e:
        print(e);



def NN_multi_predict(train_datas,model_data, NN_model):
    try:
        bench_mark = None
        predict_datas = model_data['predict_datas']
        for idx,predict_data in enumerate(predict_datas):
            farm_feature = 'wp'+str(idx+1)
            farm_idx = int(farm_feature[-1])
            predict_item = None
            #for item in tqdm.tqdm(predict_data):
            sub_predict_items = train_datas[(train_datas['Flag']==False)&(train_datas['Farm']== farm_idx )].copy()
            y_val = NN_model.model.predict(predict_data)
            y_val = y_val.reshape(-1)
            print(y_val.shape)
            print(sub_predict_items.shape)
            print(sub_predict_items.shape)
            sub_predict_items['power'] = y_val
            predict_item = pd.concat([predict_item, sub_predict_items],axis=0) if predict_item is not None else sub_predict_items
            predict_item = predict_item.groupby('date').mean().reset_index()
            merge_item = predict_item[['date','power']].copy()
            merge_item.rename(columns={"power":farm_feature},inplace=True);
            bench_mark = bench_mark.merge(merge_item, left_on='date',right_on='date') if bench_mark is not None else merge_item
        return bench_mark;
    except Exception as e:
        print(e);


"""
def NN_multi_predict(train_datas, model_data, NN_model):
    try:
        #sequence_length = params['sequence_length'] - 1
        #number_category = params['number_catagory']
        #x_features = model_data['x_features']
        #y_feature = model_data['y_features']
        bench_mark = None
        for farm_feature in model_data['farm_feature']:
            group_item = model_data[farm_feature]['test']
            farm_idx = int(farm_feature[-1])
            predict_item = None
            for item in group_item:
                sub_predict_items = train_datas[(train_datas['Flag']==False)&(train_datas['Farm']== farm_idx )]
                y_val = NN_model.model.predict(item)
                y_val = y_val.reshape(-1)
                print(y_val.shape)
                print(sub_predict_items.shape)
                print(sub_predict_items.shape)
                sub_predict_items['power'] = y_val
                predict_item = pd.concat([predict_item, sub_predict_items],axis=0) if predict_item is not None else sub_predict_items
                break
            predict_item = predict_item.groupby('date').mean().reset_index()
            merge_item = predict_item[['date','power']].copy()
            merge_item.rename(columns={"power":farm_feature},inplace=True);
            bench_mark = bench_mark.merge(merge_item, left_on='date',right_on='date') if bench_mark is not None else merge_item
        return bench_mark;
    except Exception as e:
        print(e);
"""

## 多线程性能没有提高;
from multiprocessing.pool import ThreadPool as Pool

def get_iter_rows(predict_datas,power_Data,sequence_length,model,x_features,farm_feature,Limit_Number=None):
    if(Limit_Number is None):
        Limit_Number = predict_datas.shape[0];
    for index, data_item in predict_datas[:Limit_Number].iterrows(): #tqdm.tqdm(, total=Limit_Number):
            index_loc = index;
            pred_start_time = power_Data.loc[index_loc - sequence_length, 'date'] + timedelta(hours=sequence_length);
            start_time = data_item['date'];
            assert pred_start_time == start_time;
            x_predict = np.concatenate(
                [power_Data[index_loc - sequence_length + 1:index_loc + 1][x_features[:-1]].values,
                 power_Data[index_loc - sequence_length:index_loc][x_features[-1:]].values], axis=1);
            y_predict = model.predict(np.array([x_predict]));
            predict_datas.loc[index, 'power'] = y_predict[0][0];
            power_Data.loc[index_loc, 'power'] = y_predict[0][0];
    predict_item = predict_datas[['date', 'power']].rename(columns={"power": farm_feature});
    return predict_item;

def multi_predict(model_data,res_data,params,Limit_Number=None):
    try:
        sequence_length = params['sequence_length'] - 1;
        x_features = res_data['x_features']
        farm_features = res_data['farm_feature'];
        bench_mark = None;
        pool_size = len(farm_features);
        multiple_results=[];
        # measure process time
        t0 = time.clock()
        with Pool(processes= pool_size) as pool:
            for farm_feature in (farm_features):
                power_Data = model_data[farm_feature]['power_Data'];
                predict_datas = power_Data[power_Data['Flag'] == False];
                model = model_data[farm_feature]['model'].model;
                multiple_results.append(pool.apply_async(
                    get_iter_rows,(power_Data, predict_datas,sequence_length,model,x_features,farm_feature,Limit_Number)));
        for res in multiple_results:
            predict_item = res.get(); #timeout=1000
            bench_mark = bench_mark.merge(predict_item, left_on='date',right_on='date') \
            if bench_mark is not None else predict_item
        print("Time: ", time.clock() - t0);
        return bench_mark;
    except Exception as e:
        print(e);


def Ensemble_model(model_data,res_data, params, Limit_Number=None, offset=48,Ensemble_Model=None):
    try:
        sequence_length = params['sequence_length'] - 1
        number_category = params['number_catagory']
        if (params['objective'] == "category" or params['objective'] == "binary_catagory"):
            offset = 1
        x_features = res_data['x_features']
        y_feature = res_data['y_features']
        bench_mark = None
        for farm_feature, item in model_data.items():
            power_Data = model_data[farm_feature]['power_Data'].copy()
            predict_datas = power_Data[power_Data['Flag'] == False].copy()
            if (Limit_Number is None):
                Limit_Number = predict_datas.shape[0]
            for index in tqdm.tqdm(range(0, Limit_Number, offset)):
                data_item = predict_datas[index:(index + offset)]
                index_loc = data_item.index.values[0]
                pred_start_time = power_Data.loc[index_loc - sequence_length, 'date'] + timedelta(
                    hours=sequence_length);
                start_time = data_item.loc[data_item.index.values[0], 'date']
                assert pred_start_time == start_time;
                x_predict = np.concatenate(
                    [power_Data[(index_loc - sequence_length + offset):(index_loc + offset)][
                         x_features[:-1]].values,
                     power_Data[(index_loc - sequence_length):(index_loc)][x_features[-1:]].values], axis=1)
                if (Ensemble_Model is not None):
                    new_x_trains = None
                    for name in model_data.items():
                        new_x_train = model_data[name]['model'].model.predict( x_predict )
                        new_x_train = new_x_train.reshape(-1, new_x_train.shape[1], 1);
                        new_x_trains = np.concatenate([new_x_trains, new_x_train],axis=2) \
                            if new_x_trains is not None else new_x_train
                    y_predict = Ensemble_Model.model.model.predict(new_x_trains);
                else:
                    y_predict = model_data[farm_feature]['model'].model.predict(np.array([x_predict]));
                st_time = pred_start_time + timedelta(hours=offset)
                loc_idx = predict_datas[
                    (predict_datas['date'] < st_time) & (predict_datas['date'] >= pred_start_time)].index.values;

                if (params['objective'] == "category"):
                    y_value = np.argmax(y_predict[0][:]);
                    predict_datas.loc[index_loc, 'power'], predict_datas.loc[
                        index_loc, y_feature] = y_value / number_category, y_value;
                    power_Data.loc[index_loc, 'power'], power_Data.loc[
                        index_loc, y_feature] = y_value / number_category, y_value;
                else:
                    y_value = np.round(y_predict[0][:]) if (params['objective'] == "binary_catagory") else \
                        y_predict[0][:]
                    predict_datas.loc[loc_idx, 'power'] = y_value
                    power_Data.loc[loc_idx, 'power'] = y_value;
            predict_item = predict_datas[['date', 'power']].rename(columns={"power": farm_feature});
            bench_mark = bench_mark.merge(predict_item, left_on='date', right_on='date') \
                if bench_mark is not None else predict_item
        return bench_mark;
    except Exception as e:
        print(e);




