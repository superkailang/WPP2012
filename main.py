import os
from os.path import join
from collections import defaultdict
import numpy as np
from core.file import get_data

params = defaultdict(lambda: None)
cc_params = {
    "min_thread_value": 0.00,
    "Limit_Number":None,
    'n_farm':1,
    'thread_value': 0.1,
    'number_catagory':50,
    'sequence_length':37,
    'train_set_fraction': 0.90,
    'epoch_number':1,
    'seq_interval': 1,
    'batch_size': 1024,
    'save_dir':'WIND', # 保存文件目录;
    'objective': 'regression', #catagory,regression, category,binary_catagory
    'add_new_feature': False,
    'add_binary': False,
    'total': True,
    'offset_wind': False,
    'NN_model':{
        'optimizer': {'type': 'sgd', 'lr': 0.1},
        'metrics': {'metrics': ['mse'] },  # regression mse, categorical accuracy,
        'loss': {'loss': None } # regression None categorical_crossentropy,binary_crossentropy
    },
    'layer': [
        {"type": "lstm","neurons": 48,"input_timesteps": 84,"input_dim": 5, "return_seq":False,"go_backwards": False},
        {"type": "dropout","rate": 0.4},
        #{"type": "lstm","neurons": 72,"return_seq": False},
        #{"type": "dropout","rate": 0.4},
        {'type': 'BatchNormalization'},
        #{"type": "dropout", "rate": 0.4},
        {"type": "dense", "size": 48,'activation':'linear'},
    ],
    'wind_gbm':{ 'wp1':{ 'number_leave': 20}, 'wp2':{ 'number_leave': 50},
            'wp3': { 'number_leave': 20}, 'wp4':{ 'number_leave': 50},
            'wp5': { 'number_leave': 50}, 'wp6':{ 'number_leave': 50},
            'wp7': {'number_leave': 50}},
    'binary_gbm':{ 'wp1':{ 'number_leave': 100}, 'wp2':{ 'number_leave': 100},
            'wp3': { 'number_leave': 1000}, 'wp4':{ 'number_leave': 100},
            'wp5': { 'number_leave': 200}, 'wp6':{ 'number_leave': 100},
            'wp7': {'number_leave': 100}},
    'gbm_model':{
        'is_grid_search': {'lgm_params': {'num_leaves': [50, 80, 100, 200, 500,1000, 1500, 2000,2500,3000,4000]}},
        'n_estimators': 20000,
    },
    'file_path':"GEF2012-wind-forecasting", # Kaggle ../input;
}
params.update(cc_params)

def load_data(n_farm,file_path):
    path = os.getcwd()
    data_root_path = join(path, file_path)
    train_file = join(data_root_path, 'train.csv')
    benchmark_file = join(data_root_path, 'benchmark.csv')
    test_file = join(data_root_path, 'test.csv')
    wind_forcast_files = [join(data_root_path, 'windforecasts_wf' + str(i + 1) + '.csv') for i in range(n_farm)]
    res_Data = get_data(train_file, benchmark_file, wind_forcast_files,test_file,fill_Nan = None)
    return res_Data

if __name__ == '__main__':
    res_Data = load_data(params['n_farm'], params['file_path'])