# WPP2012


### Background

The current global energy structure is transforming from traditional fossil energy to renewable energy. At present, one-third of the world's power generation comes from renewable energy, and wind energy is the main driving force for the transformation of renewable energy. In recent years, wind power generation has grown rapidly, with an average annual growth rate of 10% and accounting for about 24% of renewable energy. All wind turbines installed by the end of 2018 can cover nearly 6% of global electricity demand. It is expected that there will be more wind energy growth in the next few years.

Accurate wind energy forecasting is essential for smart grid operation. This projects studies machine learning algorithms for short-term wind power generation prediction. 


### DataSet Description


The [dataSet](https://github.com/superkailang/WPP2012/tree/master/Data) comes from Global Energy Forcasting Competition 2012 ([GEFCom2012](http://www.drhongtao.com/gefcom/2012))

It's recomend to get the description from [Kaggle Platform](https://www.kaggle.com/c/GEF2012-wind-forecasting/overview)

This is a wind power forecasting problem to predicting hourly power generation up to 48 hours ahead at 7 wind farms.

The period between 2009/7/1 and 2010/12/31 is a model identification and training period, while the remainder of the dataset, that is, from 2011/1/1 to 2012/6/28, is there for the evaluation. 

Note that to be consistent, only the meteorological forecasts for that period that would actually be available in practice are given. These two periods then repeats every 7 days until the end of the dataset. Inbetween periods with missing data, power observations are available for updating the models.

- "train.csv" contains the training data:
- the first column ("date") is a timestamp giving date and time of the hourly wind power measurements in following columns. For instance "2009070812" is for the 8th of July 2009 at 12:00;

- the following 4 columns ("u", "v", "ws" and "wd") are the weather forecasts variables,

- the file "benchmark.csv"  provides the forcast results 

### Recent Work

state of the Art work 

Methods | 48H ahead 
:-: | :-: 
ALL-CF | 0.14564 | 
GBM + K-Means + LR | 0.14567 |
KNN | 0.1472 |
SGCRF | 0.1488 |
LSBRT | 0.1518 | 
SDAE-m-m | 0.154 | 
S-GP-ENV  | 0.1604| 
GP + NN | 0.1752 |
Persistence | 0.361 | 

### Project Description

```python

```

There are majoly two part in our work.

- LightGBM used for regression analysis

This part is full described in the model section

```python

```

- LSTM regression


#### Hyprid LSTM Model for Wind Power Prediction
After K-means Processing 

```python
    Net_model = NN_Net(params, x_train)
    scores = Net_model.train(x_train, y_train, x_test, y_test, 'wsPower2', load_models=False)
```
