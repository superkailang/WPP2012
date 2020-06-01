"""
*
  绘制图形;
*
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
# zhfont1 = fontManager.FontProperties(fname='simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fmt = mdates.DateFormatter('%Y-%m-%d')

def plot(data,xlabel,ylabels):
    nfigure = len(ylabels);
    for i,ylabel in enumerate(ylabels):
        plt.subplot(nfigure, 1, i+1);
        plt.plot(data[xlabel], data[ylabel]);
        plt.xlabel(xlabel);
        # X轴的间隔为天;
        plt.ylabel(ylabel);
        plt.legend([ylabel]);
        plt.gcf().autofmt_xdate()
    #plt.subplot(2, 1, 2);
    #plt.plot(data[xlabel], data[ylabel2], 'r-');
    #plt.legend([ylabel2]);
    #plt.xlabel(xlabel);
    #plt.ylabel(ylabel2);
    ## ax2.xaxis.set_major_formatter(fmt)
    #plt.gcf().autofmt_xdate()
    plt.show();

def plot2(data,xlabel,ylabel):
    plt.plot(data[xlabel], data[ylabel]);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.show();

def plotgroup(GroupByEqusid):
    i = 0;
    GroupSize = len(GroupByEqusid);
    for name, item in GroupByEqusid:
        print(item.head(10))
        print(item.describe())
        plt.subplot(GroupSize, 1, i + 1);
        i = i + 1;
        plot2(item, 'Data_Time', 'Value');
        plt.legend([name]);
    plt.show();

def plot_Predict(y1,y2,name):
    x = np.arange(len(y1))
    plt.plot(x,y1,'r-')
    plt.plot(x, y2, 'b-')
    plt.ylabel( u'test: ' + name)
    label = [u'real', u'predicted']
    plt.legend(label)
    plt.show()

def plot_Predict_time(time,y1,y2,name):
    #xs = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in time];
    x = np.arange(len(y1));
    plt.plot(x,y1,'r-');
    plt.plot(x, y2, 'b-');
    plt.ylabel(u'predict: ' + name)
    label = [u'real', u'predicted'];
    plt.legend(label);
    plt.show();

def plot_history(epochs,loss,val_loss,name):
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss_'+ name);
    plt.legend(['Training loss','Validation loss'])
    plt.show()
    #plt.clf()  # clear figure
    #plt.plot(epochs, acc, 'bo', label='Training acc')
    #plt.plot(epochs, val_acc, 'b', label='Validation acc')
    #plt.title('Training and validation accuracy')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.legend()
    plt.show()

def plot_importance(filename,x_importances, y_importances,ytitle,modelName):
    fig = plt.figure(figsize=(7, 6));
    y_pos = np.arange(len(x_importances))
    # 横向柱状图
    plt.barh(y_pos, y_importances, align='center')
    plt.yticks(y_pos, x_importances)
    plt.xlabel('Importances')
    plt.ylabel(ytitle);
    plt.xlim(0, 1)
    plt.title('Features Importances')
    fig.tight_layout(pad=2.4, w_pad=0.5, h_pad=0.5);
    plt.savefig(filename);
    plt.legend([modelName])
    plt.show()