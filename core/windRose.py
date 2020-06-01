import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from os.path import join
import pandas as pd


def Wind_Rose_Map(ws,wd,ax=None):
    import matplotlib.cm as cm
    # Create wind speed and direction variables
    if ax is None:
        ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='black')
    ax.set_legend()
    return ax


def Wind_Rose(ax,train_data,title = "",save_file=None):
    train_data['wd'] = train_data['wd'] * 360
    # fig, ax = plt.subplots(1, 2, figsize=fig_size)
    rose_axs = Wind_Rose_Map(train_data['ws'], train_data['wd'], ax=ax)
    rose_axs.set_xlabel(title)
    #ax2 = fig.add_subplot(122, projection="windrose")
    #rose_axs_right = Wind_Rose_Map(train_data['ws_100'], train_data['wd_100'], ax=ax2)
    #rose_axs_right.set_xlabel("Wind Speed Distribution")


def get_Data():
    new_data = pd.read_csv('new_data3.csv')
    format = "%Y-%m-%d %H:%M:%S"
    new_data['date'] = pd.to_datetime(new_data['date'], format=format)
    new_data['start'] = pd.to_datetime(new_data['start'], format=format)
    return new_data


Data = get_Data()
FarmNum=7



def Wind_Rose_AllFram():
    fig_size = (24, 13)
    fig = plt.figure(figsize=fig_size)
    save_file = join("core", "image", 'WindRose_Farm')
    title="Wind Speed Distribution in Laurel, NE"
    n_row = 2
    n_col = 4
    figure, ax = plt.subplots(n_row, n_col, figsize=fig_size)  # sharex='all', sharey='all',
    for FarmId in range(1,FarmNum+1):
        row_idx = (FarmId - 1) // n_col
        col_idx = (FarmId - 1) % n_col
        idxs = row_idx * n_col + col_idx
        ax = fig.add_subplot(2,4,idxs + 2, projection="windrose")
        Wind_Rose(ax,Data[Data['Farm']==FarmId],title="Farm {}".format(FarmId),save_file=save_file)
    fig.text(0.5, 0.94, title, ha='center')
    fig.text(0.08, 0.5, "Wind Power", va='center', rotation='vertical')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()
    exit()

fig_size = (14, 10)
fig = plt.figure(figsize=fig_size)
grid = plt.GridSpec(2, 8,wspace=0.05, hspace=0.05)
save_file = join("core", "image", 'WindRose_Farm')
title="Wind Speed Distribution in Laurel, NE"
n_row = 2
n_col = 4
offset = 1
pix_length = 2
for FarmId in range(1,FarmNum+1):
    if (offset > 6):
        offset = 0
    row_idx = ( FarmId  ) // n_col
    col_idx = (FarmId - 1) % n_col
    ax = plt.subplot(grid[1*row_idx:1*(row_idx+1), offset:offset+pix_length], projection="windrose")
    offset  = offset + pix_length
    Wind_Rose(ax,Data[Data['Farm']==FarmId],title="Farm {}".format(FarmId),save_file=save_file)
#fig.text(0.5, 0.84, title, ha='center')
#fig.text(0.08, 0.5, "Wind Power", va='center', rotation='vertical')
if save_file is not None:
    plt.savefig(save_file, dpi=300, pad_inches=0, bbox_inches='tight')
plt.show()
exit()



import plotly.express as px
df = px.data.wind()
print(df.head(1))
fig = px.bar_polar(df, r="frequency", theta="direction",
                   color="strength", template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()



import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Barpolar(
    r=[77.5, 72.5, 70.0, 45.0, 22.5, 42.5, 40.0, 62.5],
    name='11-14 m/s',
    marker_color='rgb(106,81,163)'
))
fig.add_trace(go.Barpolar(
    r=[57.5, 50.0, 45.0, 35.0, 20.0, 22.5, 37.5, 55.0],
    name='8-11 m/s',
    marker_color='rgb(158,154,200)'
))
fig.add_trace(go.Barpolar(
    r=[40.0, 30.0, 30.0, 35.0, 7.5, 7.5, 32.5, 40.0],
    name='5-8 m/s',
    marker_color='rgb(203,201,226)'
))
fig.add_trace(go.Barpolar(
    r=[20.0, 7.5, 15.0, 22.5, 2.5, 2.5, 12.5, 22.5],
    name='< 5 m/s',
    marker_color='rgb(242,240,247)'
))

fig.update_traces(text=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
fig.update_layout(
    title='Wind Speed Distribution in Laurel, NE',
    font_size=16,
    legend_font_size=16,
    polar_radialaxis_ticksuffix='%',
    polar_angularaxis_rotation=90,

)
fig.show()
