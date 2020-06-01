#### Data Exploration From：https://github.com/greenlytics/gefcom2014-wind/blob/master/notebooks/data-exploration.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from os.path import join


import seaborn as sns
from matplotlib.ticker import MultipleLocator

sns.set(style="darkgrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

## 绘制功率时间曲线
def plot_FarmPower(train_data,test_data,Farm,save_file=None,filter=False):
    import seaborn as sns
    sns.set(style="darkgrid")
    tr_data, ts_data = train_data.copy(),test_data.copy()
    tr_data = tr_data.rename(columns={"power": "train"})
    ts_data = ts_data.rename(columns={"power": "test"})
    fig_size = (12, 5)
    if filter:
        fig_size = (12, 8)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    #plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #plt.margins(0, 0)
    title = "Farm={}".format(Farm)
    tr_data[(tr_data['Farm'] == Farm)].plot(x='date', y="train", kind='line', ax=ax, title= title)
    ts_data[ts_data['Farm'] ==  Farm].plot(x='date', y="test", kind='line', ax=ax, title = title)
    plt.ylabel("wind power")
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches = 0,bbox_inches='tight')
    plt.show()

## 绘制功率箱线图
def plot_Power_BoxPlot(Data,save_file=None):
    sns.set(style="darkgrid")
    ZoneDatas = Data[['Farm','date','power']].copy()
    ZoneMap =  None
    ZoneNum = 7
    plt.figure(figsize=(10, 8))
    for Farm in range(1,ZoneNum+1):
        Zone_label = "Farm {}".format(Farm)
        Zone_data = ZoneDatas[ZoneDatas['Farm'] == Farm].copy()
        Zone_data = Zone_data.rename(columns={"power": Zone_label })
        if ZoneMap is None:
            ZoneMap = Zone_data
        else:
            ZoneMap[Zone_label] = Zone_data[Zone_label].values

    data = pd.melt(ZoneMap.drop(columns=['date', 'Farm']),
                   var_name="Farm",
                   value_name='power')
    sns.boxplot(x="Farm", y="power", data=data)

    #ZoneMap.drop(columns=['date', 'Farm']).plot.box(figsize=(20,8))
    #sns.heatmap(data= ZoneMap.drop(columns=['date', 'Farm']).corr(), annot=True)
    plt.title("Wind power distribution at different Farm")
    plt.xlabel("Farm")
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

## 绘制功率分布
def plot_FarmPower_Hist(Data):
    """
    plt.figure(figsize=(6,6))
    grid = plt.GridSpec(4,4,wspace=0.05,hspace=0.05)
    wp_Farm_1 = Data[Data['Farm'] == 1]
    wp_Farm_2 = Data[Data['Farm'] == 2]
    main_ax = plt.subplot(grid[1:4, 0:3])
    plt.scatter(wp_Farm_1['power'],wp_Farm_2['power'], s=4, marker='o')
    y_hist = plt.subplot(grid[1:4, 3], xticklabels=[], sharey=main_ax)
    plt.hist(wp_Farm_2['power'], 60, orientation='horizontal')  # , color='grey'
    # y_hist.invert_xaxis(
    x_hist = plt.subplot(grid[0, 0:3], yticklabels=[], sharex=main_ax)
    plt.hist( wp_Farm_1['power'], 60, orientation='vertical')  # color='grey'
    plt.show()
    """
    wp_ratio = Data[Data['power'] == 0].shape[0] / Data.shape[0]
    print("wp: {}".format(wp_ratio))
    import seaborn as sns
    sns.set(style="darkgrid")
    #sns.distplot(Data['power'], bins=50, kde=True, color="coral");
    Data.hist('power',grid=True,bins=20,rwidth=0.9) #color='#0504aa'

    save_file = join("core","image","Hist_TARGETVAR")
    plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()
    # g=sns.heatmap(new_data[new_data['Flag']==False][top_corr_features].corr(),annot=True,cmap="RdYlGn")

##U10_V10 分布
def plot_Wind(Data,Farm,save_file=None):
    plt.figure(figsize=(6,6))
    grid = plt.GridSpec(4,4,wspace=0.05,hspace=0.05)
    title = "Farm={}".format(Farm)
    ULabel = 'u'
    VLabel = 'v'
    main_ax = plt.subplot(grid[1:4, 0:3])
    plt.scatter(Data[Data['power']!=0][ULabel],Data[Data['power']!=0][VLabel],s=4, marker='o')
    plt.scatter(Data[Data['power']==0][ULabel],Data[Data['power']==0][VLabel],s=4,c="darkorange", marker='o')
    plt.legend(["nonzero output","zero output"])
    #plt.scatter([0],[0],c='r',marker='+')
    plt.xlabel("Zonal Wind Velocity U") # zonal Wind
    plt.ylabel("Meridional Wind Velocity V")
    plt.title(title)

    y_hist = plt.subplot(grid[1:4, 3], xticklabels=[], sharey=main_ax)
    plt.hist(Data[Data['power']!=0][VLabel], 60, orientation='horizontal') #, color='grey'
    # y_hist.invert_xaxis()

    x_hist = plt.subplot(grid[0, 0:3], yticklabels=[], sharex=main_ax)
    plt.hist(Data[Data['power']!=0][ULabel], 60, orientation='vertical') #color='grey'
    # x_hist.invert_yaxis()
    #plt.xlim([-10,10])
    #plt.ylim([-10,10])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

##WS_Wd 分布
def plot_WindSpeed_WindDirection(Data,Farm,save_file=None):
    plt.figure(figsize=(6,6))
    grid = plt.GridSpec(4,4,wspace=0.05,hspace=0.05)
    title = "Farm={}".format(Farm)
    ULabel = 'wd'
    VLabel = 'ws'
    main_ax = plt.subplot(grid[1:4, 0:3])
    sns.kdeplot(Data[ULabel],Data[VLabel],shade=True)
    #plt.scatter(Data[Data['power']!=0][ULabel],Data[Data['power']!=0][VLabel],s=4, marker='o')
    #plt.scatter(Data[Data['power']==0][ULabel],Data[Data['power']==0][VLabel],s=4,c="darkorange", marker='o')
    #plt.legend(["nonzero output","zero output"])
    #plt.scatter([0],[0],c='r',marker='+')
    plt.xlabel("Wind Direction") # zonal Wind
    plt.ylabel("Wind Speed")
    plt.title(title)

    y_hist = plt.subplot(grid[1:4, 3], xticklabels=[], sharey=main_ax)
    plt.hist(Data[VLabel], 60, orientation='horizontal') #, color='grey'
    # y_hist.invert_xaxis()

    x_hist = plt.subplot(grid[0, 0:3], yticklabels=[], sharex=main_ax)
    plt.hist(Data[ULabel], 60, orientation='vertical') #color='grey'
    # x_hist.invert_yaxis()
    #plt.xlim([-10,10])
    #plt.ylim([-10,10])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

#WS_Power 分布
def plot_WS_Power(Data,ZoneNum,save_file=None):
    fig_size = (12, 8)
    ws_label = "ws"
    n_row = 2
    n_col = ZoneNum//n_row
    figure,ax = plt.subplots(n_row,n_col,figsize=fig_size) #sharex='all', sharey='all',
    for Farm in range(1,ZoneNum+1):
        row_idx = (Farm-1) // n_col
        col_idx =  (Farm - 1) % n_col
        ZoneData = Data[(Data['Farm'] == Farm)]
        #Data[(Data['Farm'] == Farm)].plot(x=ws_label, y="power", kind='line', ax=ax)
        ax[row_idx][col_idx].scatter(ZoneData[ws_label],ZoneData['power'],s=4, marker='o')
        title = "Farm {}".format(Farm)
        ax[row_idx][col_idx].set_title(title)
    figure.text(0.5, 0.04, "Wind Speed", ha='center')
    figure.text(0.08, 0.5, "Wind Power", va='center', rotation='vertical')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

## WS_Power 时序分布
def plot_WS_Power_Sequence(Data,Farm,save_file=None):
    ZoneData = Data[(Data['Farm'] == Farm)].copy()
    offset_time = ZoneData.loc[ZoneData.index[0],'date'] + timedelta(days=31)
    ZoneData = ZoneData[ZoneData['date']<=offset_time]
    ws_label = "ws"
    ZoneData[ws_label] = ZoneData[ws_label]/ ZoneData[ws_label].max()
    fig_size = (12, 5)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    title = "Farm {}".format(Farm)
    ZoneData.plot(x='date', y=ws_label, kind='line', ax=ax, title= title)
    ZoneData.plot(x='date', y="power", kind='line', ax=ax, title = title)
    plt.legend([ws_label,'wind power'])
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches = 0,bbox_inches='tight')
    plt.show()

## Lag 相关性比较
def Lag_Power(Data,Farm,save_file=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    ZoneData = Data[Data['Farm']==Farm].copy()
    for Lag_offset in range(1,240+1):
        Lag_label = 'Lag_{}'.format(Lag_offset)
        ZoneData[Lag_label] = ZoneData['power'].shift(periods=Lag_offset)
    Lag_labels = [ 'Lag_{}'.format(Lag_offset) for Lag_offset in range(1,240) ]
    corr = ZoneData[Lag_labels + ['power']].corr()
    corr_value = corr.loc[ ['power']+Lag_labels,'power'].values

    fig_size = (12, 5)
    fig, ax = plt.subplots(1, 2, figsize=fig_size)
    ax[1].plot(corr_value)
    ax[1].plot( [0,240],[0,0],'k')
    ax[1].plot( [0.0,240],[0.05,0.05],'--')
    ax[1].plot([0.0, 240], [-0.05, -0.05],'--')
    ax[1].set_xlim([0,240])
    xmajorLocator = MultipleLocator(48)
    ax[1].xaxis.set_major_locator(xmajorLocator)
    ax[1].set_xlabel("Lag (hours)")
    ax[1].set_ylabel("相关系数")
    ax[0].scatter(ZoneData['power'], ZoneData[Lag_labels[0]], s=4, marker='o')
    ax[0].set_xlabel("Power at time T")
    ax[0].set_ylabel("Power at time T+1")
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

### 风玫瑰图
def Wind_Rose_Map():
    import matplotlib.pyplot as plt
    from windrose import WindroseAxes
    import matplotlib.cm as cm
    # Create wind speed and direction variables
    ws = np.random.random(500) * 6
    wd = np.random.random(500) * 360
    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.show()

### Farm区域相关性
def Wind_Zones_Coeff(Data,ZoneNum,save_file=None):
    sns.set(style="darkgrid")
    ZoneDatas = Data[['Farm','date','power']].copy()
    ZoneMap =  None
    plt.figure(figsize=(10, 8))
    for Farm in range(1,ZoneNum+1):
        Zone_label = "Farm {}".format(Farm)
        Zone_data = ZoneDatas[ZoneDatas['Farm'] == Farm].copy()
        Zone_data = Zone_data.rename(columns={"power": Zone_label })
        if ZoneMap is None:
            ZoneMap = Zone_data
        else:
            ZoneMap[Zone_label] = Zone_data[Zone_label].values
    sns.heatmap(data=ZoneMap.drop(columns=['date', 'Farm']).corr(), annot=True)
    plt.title("Correlation between Different Farm")
    plt.xlabel("Farm")
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

### 风特征相关性
def Wind_Feature_Coeff(Data,save_file=None):
    plt.figure(figsize=(8, 6))
    x_list = ['year','month','week','hour','u','v','ws','wd','ws_2','ws_3'] # ['year','month','hour','u','v','ws','wd','ws_2','ws_3']
    ZoneData = Data[Data['Farm'] == 1]
    data_n_2 = ZoneData[x_list + ['power']].copy()
    if True:
        sns.heatmap(data=data_n_2[x_list + ['power'] ].corr(), annot=True)
        plt.title("Correlation between Different Features")
        plt.xlabel("Farm")
    else:
        data_n_2[x_list] = (data_n_2[x_list] - data_n_2[x_list] .mean()) / (data_n_2[x_list].max()-data_n_2[x_list].min())
        data = pd.melt(data_n_2, id_vars="power",
                       var_name="features",
                       value_name='value')
        sns.boxplot(x="features", y="value", data=data)
        #sns.violinplot(x="features", y="value", data=data, split=True, inner="quart")
        #sns.violinplot(x="features", y="value", hue="power", data=data,split=True, inner="quart")
        #plt.xticks(rotation=90)
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

def plot_quantile(y_pred_upper, y_pred_lower, y_test, num=240,alpha=1.0):
    # Reshape to 2D array for standardization
    y_pred_upper = np.reshape(y_pred_upper, (len(y_pred_upper), 1))
    y_pred_lower = np.reshape(y_pred_lower, (len(y_pred_lower), 1))
    # y_pred_upper = sc_load.inverse_transform(y_pred_upper)
    # y_pred_lower = sc_load.inverse_transform(y_pred_lower)
    # y_test = sc_load.inverse_transform(y_test)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(y_test[:num], label="Actual")
    ax.plot(y_pred_upper[:num], label="95%")
    ax.plot(y_pred_lower[:num], label="5%")
    ax.fill_between(np.arange(0, num), y_pred_upper[0:num].ravel(),
                    y_pred_lower[0:num].ravel(), color="c",label="90% PI", alpha=alpha) #
    ax.set_title("Quantile Regession")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Normalized Load (pu)")
    ax.legend()
    plt.show()

def plot_quantile2(Data2,Farm=1, num=120,offset = 48,alpha=1.0,save_file=None):
    Data = Data2[Data2['Farm']==Farm]
    quantile = np.array(range(1,51))
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    Time = np.array(range(num+offset)) #Data.loc[Data.index[:num + offset], 'date']
    y_test = Data['power'].values[0:num+offset]
    #ax.plot(y_test[:offset], label="Actual")
    ax.plot(Time,y_test, label="Actual")
    for quantile_idx in quantile:
        upper_label = "NN_wp_{}".format(quantile_idx)
        lower_label = "NN_wp_{}".format(100-quantile_idx)
        alpha =  0.020 * quantile_idx
        y_pred_upper = Data[upper_label].values[0:num+offset]
        y_pred_lower = Data[lower_label].values[0:num+offset]
        label = None
        if quantile_idx ==50:
            ax.plot(Time[offset:], y_pred_upper[offset:], label="50%")
        if quantile_idx == 5:
            #ax.plot(Time[offset:],y_pred_upper[offset:], label="95%")
            #ax.plot(Time[offset:],y_pred_lower[offset:], label="5%")
            label = "90% PI"
        if quantile_idx>=5:
            ax.fill_between(np.arange(offset, num + offset), y_pred_upper[offset:].ravel(),
                                y_pred_lower[offset:].ravel(), color='y', alpha=alpha, label=label)  # "#d2d8e8"
    ax.set_title("Quantile Regession Farm {}".format(Farm))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Normalized Load")
    ax.set_xlim([0,num+offset+1])
    ax.set_ylim([0,1])
    xmajorLocator = MultipleLocator(72)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.legend()
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches = 0,bbox_inches='tight')
    plt.show()

def postProcessWindData(task_id,addperiod=False):
    #save_quantile_file = './Data/Wind_quantile_{}.csv'.format(task_id)
    save_quantile_file = "./Data/NN_Data.csv"
    if addperiod:
        save_quantile_file = './Data/WindWithPeroid_quantile_{}.csv'.format(task_id)
    Data = pd.read_csv(save_quantile_file)
    Data['date'] = pd.to_datetime(Data['date'], format="%Y%m%d %H:%M:%S")
    train_data, test_data = Data[Data['Flag'] == True], Data[Data['Flag'] == False].copy()
    quantile_cols = np.array(range(1, 100)) / 100
    wp_labels = ['NN_wp_{}'.format(quantile_idx) for quantile_idx in np.array(range(1, 100))]
    test_data.loc[:, wp_labels][test_data.loc[:, wp_labels] < 0.00] = 0
    test_data.loc[:, wp_labels][test_data.loc[:, wp_labels] > 1.00] = 1.00
    if True:
        for quantile_idx in np.array(range(1, 100)):
            wp_label = 'NN_wp_{}'.format(quantile_idx)
            test_data.loc[:, wp_label][test_data.loc[:, wp_label] < 0.00] = 0
            test_data.loc[:, wp_label][test_data.loc[:, wp_label] > 1.00] = 1.00
    unsort_data = test_data.loc[:, wp_labels]  # n_row X n_col 数据
    test_data.loc[:, wp_labels] = np.sort( unsort_data )
    #sorted_Data = np.sort( unsort_data )
    #predict_Data = np.array([sorted_Data[:, [idx]] for idx in range(unsort_data.shape[1])])
    #unsort_predict_Data = np.array([unsort_data.values[:, [idx]] for idx in range(unsort_data.shape[1])])
    #L1 = get_BenchLoss(quantile_cols, predict_Data, test_data.loc[:, ['power']].values)
    #L2 = get_BenchLoss(quantile_cols, unsort_predict_Data, test_data.loc[:, ['power']].values)
    #print("quantile: {}, Loss: {}, after Loss:{}".format(task_id,L2,L1) )
    Data = pd.concat([train_data, test_data], axis=0)
    return Data


def plot_WindAbnormal(Data,Zone,save_file=None):
    title = "ZONEID={}".format(Zone)
    ULabel = 'u'
    VLabel = 'v'
    plt.scatter(Data[Data['power']!=0][ULabel],Data[Data['power']!=0][VLabel],s=4, marker='o')
    plt.scatter( Data[ (Data['power']==0) & (Data['cluster'] != 0) ][ULabel],Data[  (Data['power']==0) & (Data['cluster'] != 0) ][VLabel],s=4,c="darkorange", marker='o')
    plt.scatter( Data[Data['cluster'] == 0][ULabel], Data[Data['cluster'] == 0][VLabel], s=8, c="r", marker='+')
    plt.legend(["nonzero output","zero output","abnormal output",'zero wind'])
    #plt.scatter([0],[0],c='g',marker='+')
    plt.xlabel("Zonal Wind Velocity") # zonal Wind
    plt.ylabel("Meridional Wind Velocity")
    plt.title(title)
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

def get_Data():
    new_data = pd.read_csv('new_data3.csv')
    #new_data = pd.read_csv('./Data/NN_Result.csv')
    format = "%Y-%m-%d %H:%M:%S"
    new_data['date'] = pd.to_datetime(new_data['date'], format=format)
    #new_data['start'] = pd.to_datetime(new_data['start'], format=format)
    return new_data

def test_Plot():
    mean =[0,0]
    cov = [[1,1],[1,4]]
    x,y = np.random.multivariate_normal(mean,cov,3000).T
    plt.figure(figsize=(6,6))
    grid = plt.GridSpec(4,4,wspace=0.5,hspace=0.5)

    main_ax = plt.subplot(grid[1:4,0:3])
    plt.plot(x,y,'ok',markersize=3,alpha=.2)

    y_hist = plt.subplot( grid[1:4,3],xticklabels=[],sharey=main_ax)
    plt.hist(y,60,orientation='horizontal',color='grey')
    #y_hist.invert_xaxis()

    x_hist = plt.subplot(grid[0,0:3], yticklabels=[], sharex=main_ax)
    plt.hist(x, 60, orientation='vertical', color='grey')
    #x_hist.invert_yaxis()
    plt.show()

def plot_Data(test_data,Farm):
    temp_err = test_data[test_data['Farm']==Farm].copy()
    plt.figure(figsize=(8, 4))
    num = 480
    farm_name = 'power'
    #plt.plot( temp_err['ws.power'].values[:num], 'r')
    #plt.plot( temp_err[farm_name].values[:num], 'b--')
    #sns.pointplot(  temp_err['ws.power'].values[:num], 'r', markersize=2)
    #sns.pointplot( temp_err[farm_name].values[:num], 'b', markersize=2)
    d1 = temp_err.loc[temp_err.index[:num]]
    d1 = d1.reset_index(drop=True).reset_index()
    sns.lineplot(x="index", y='wp', data=d1,color = "#ff0000")
    ax = sns.lineplot(x="index", y='power', data=d1,dashes=True, color="#0000ff")
    ax.lines[0].set_linestyle("--")
    ax.lines[0].set_linewidth(2)
    plt.legend(['Predict Power', 'Real Power'])
    plt.xlabel('time')
    plt.ylabel('Wind Power')
    plt.title("Farm {}".format(Farm))
    save_file = join("core","image","Farm_Predict_Data_{}".format(Farm))
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches = 0,bbox_inches='tight')
    plt.show()

from scipy import stats
def cumulativePlot(samples,save_file=None):
    #fig = plt.figure(figsize=(8, 6))
    res = stats.cumfreq(samples, numbins=25)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,res.cumcount.size)
    plt.bar(x, res.cumcount/res.cumcount[-1], width=res.binsize)
    plt.title('Cumulative histogram')
    plt.xlim([x.min(), x.max()])
    plt.xlabel("Wind Speed m/s")
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()


from core.process import get_seqence_data,pre_process

Data = get_Data()
ZoneNum = 7

ZoneId = 1
Data.loc[:,"cluster"] = 1
Data.loc[Data['Farm'] == ZoneId] = pre_process(Data[Data['Farm'] == ZoneId], y_list="power", n_clusters=100,min_sample=40)
save_fig = join("core", "image", 'Farm{}_Abnormal_Wind_U'.format(ZoneId))
plot_WindAbnormal(Data[Data['Farm']==ZoneId],ZoneId,save_fig)

exit()

#plot_Wind(Data[Data['Farm']==ZoneId],ZoneId,save_fig)


#cumulativePlot(Data['ws'],save_file=join("core", "image", 'Cumulative_Plot'))
#exit()



def judge(x, max_value):
    return 0 if x < 0 else (max_value if x >= max_value else x);

Data['wp'] = Data['wp'].apply(lambda x: judge(x, 0.92))

for Farm in range(1,8):
    plot_Data(Data[Data['Flag']==False],Farm=Farm)

exit()

train_data,test_data = Data[Data['Flag']==True],Data[Data['Flag']==False]
print(Data.shape)

### 绘制相关系数
Wind_Feature_Coeff(Data,save_file=join("core", "image", 'Wind_Box_Plot_Zone_{}'.format(1)))
exit()

plot_WindSpeed_WindDirection(Data['Farm']==1,Farm=1)
exit()

### 绘制区域相关系数
Wind_Zones_Coeff(Data,ZoneNum,save_file=join("core", "image", 'Zone_Corr'))

## todo 绘制所有农场箱线图
plot_Power_BoxPlot(Data,save_file=join("core", "image", 'Power_Box_Plot_Zone'))


#todo 绘制农场风能数据 Hist
for ZoneId in range(1,ZoneNum+1):
    plot_FarmPower_Hist( Data[Data['Farm']==ZoneId] ) #
exit()


#g=sns.jointplot(x="u",y="v",kind="scatter",data=Data,color="darkorange",s=4, marker='o')
#g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
#sns.pairplot(train_data)
#g.set_axis_labels("$X$", "$Y$")
#plt.show()
#exit()

#todo 绘制农场风能数据
for ZoneId in range(1,ZoneNum+1):
    filter_month = False
    if filter_month:
        offset_time = test_data.loc[test_data.index[0],'date'] - timedelta(days=31)
        train_data = train_data[train_data['date']>=offset_time]
    save_fig = join("core","image",'Zone_Power_{}_{}'.format(ZoneId, "filter" if filter_month else "original" ))
    plot_FarmPower(train_data,test_data,ZoneId,save_file = save_fig,filter = None )



### 绘制Quantile
#for ZoneId in range(1,ZoneNum+1):
#    save_file=join("core", "image", 'Zone_{}_Task_{}_Wind_Forcasting'.format(ZoneId,15))
#plot_quantile2(Data.loc[Data['date'] >= offhour],Farm=ZoneId,num=480,save_file=save_file)



### 绘制箱线图





#sns.jointplot(x="U10",y="V10",data=Data,c=2)
#sns.pairplot(train_data)
#plt.show()
#exit()


#### 时间相关性
Lag_Power(Data,1,save_file =join("core", "image", 'Zone_{}_Corr_PowerSeq'.format(1)))
#exit()

#### 绘制时间序列
plot_WS_Power_Sequence(Data,1,save_file=join("core", "image", 'Zone_{}_Ws_{}_PowerSeq'.format(1,10)))

#### 绘制 全部数据
plot_WS_Power(Data,ZoneNum=7,save_file=join("core", "image", 'Zone_Ws_{}_Power'.format(10)))

## Wind_U10_V10 绘制 U10,V10 相关图

# exit()











# dataframe.plot(x='date', y=name, kind='line', ax=ax, title=name);


import seaborn as sns
from scipy.stats import kde

nbins = 100
x, y = data['ws'], data['power']
k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()

"""
data['power'] = np.exp(data['power']) - 1
for i in range(1,7):
    temp_name = 'wp_hn_'+str(i)
    data[temp_name] = np.exp(data[temp_name]) - 1
"""

ax = sns.kdeplot(x, y, shade=True)
plt.show()
def kdePlot(Data,x_label,y_label):
    ax = sns.kdeplot(Data[x_label], Data[y_label], cut=0.00001, cmap="Blues", shade=True, shade_lowest=False)
    plt.show()



#plot_orginal_Data(train_data,test_data)


def plot_importtance():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
    feature_imp = pd.DataFrame(clf.feature_importance, x_list, columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()



def Plot_result():
    temp_err = test_data[test_data['Farm'] == 1].copy()
    plt.figure(figsize=(8, 4))
    num = 480
    farm_name = 'wp1'
    plt.plot(temp_err['power'].values[:num], 'r+-', markersize=2)
    plt.plot(target_val_data[farm_name][:num], 'bo--', markersize=2)
    plt.legend(['Predict Power', 'Real Power'])
    plt.xlabel('time')
    plt.ylabel('Wind Power')
    plt.show()