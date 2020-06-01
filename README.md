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
[ALL-CF](https://ieeexplore.ieee.org/abstract/document/7497013) [3]| 0.14564 | 
[GBM + K-Means + LR](https://www.sciencedirect.com/science/article/abs/pii/S0169207013000836) [2] | 0.14567 |
[KNN](https://www.sciencedirect.com/science/article/abs/pii/S0169207013000848) [1] | 0.1472 |
[SGCRF](https://ieeexplore.ieee.org/abstract/document/6760016/) [4] | 0.1488 |
[LSBRT](https://ieeexplore.ieee.org/abstract/document/7579134/) [5] | 0.1518 | 
[SDAE-m-m](https://ieeexplore.ieee.org/abstract/document/8240639) [6] | 0.154 | 
[S-GP-ENV](https://ieeexplore.ieee.org/abstract/document/7812215) [7] | 0.1604| 
[GP + NN](https://ieeexplore.ieee.org/abstract/document/6606922/)[8] | 0.1752 |
Persistence | 0.361 | 

### Project Description

There are majoly two part in our work.

- LightGBM used for regression analysis

This part is full described in the model section

- LSTM regression for 48h ahead

#### LightGBM wind power regression


#### Hyprid LSTM Model for Wind Power Prediction

##### K-means Processing 


##### Feature Engineering

```python
    Net_model = NN_Net(params, x_train)
    scores = Net_model.train(x_train, y_train, x_test, y_test, 'wsPower2', load_models=False)
```

#### Phsyical Constrains 


### Reference
1.	[Mangalova E, Agafonov E. Wind power forecasting using the k-nearest neighbors algorithm. INT J FORECASTING. 2014;30:402-406](https://www.sciencedirect.com/science/article/abs/pii/S0169207013000848).
2.	[Silva L. A feature engineering approach to wind power forecasting: GEFCom 2012. INT J FORECASTING. 2014;30:395-401](https://www.sciencedirect.com/science/article/abs/pii/S0169207013000836).
3.	[Fang S, Chiang H. A high-accuracy wind power forecasting model. IEEE T POWER SYST. 2016;32:1589-1590](https://ieeexplore.ieee.org/abstract/document/7497013).
4.	[Wytock M, Kolter JZ, "Large-scale probabilistic forecasting in energy systems using sparse gaussian conditional random fields," in 52nd IEEE Conference on Decision and Control, (IEEE, 2013), pp. 1019-1024](https://ieeexplore.ieee.org/abstract/document/6760016/).
5.	[Li G, Chiang H. Toward cost-oriented forecasting of wind power generation. IEEE T SMART GRID. 2016;9:2508-2517](https://ieeexplore.ieee.org/abstract/document/7579134/).
6.	[Yan J, Zhang H, Liu Y, Han S, Li L, Lu Z. Forecasting the high penetration of wind power on multiple scales using multi-to-multi mapping. IEEE T POWER SYST. 2018;33:3276-3284](https://ieeexplore.ieee.org/abstract/document/8240639).
7.	[Fang S, Chiang H. Improving supervised wind power forecasting models using extended numerical weather variables and unlabelled data. IET RENEW POWER GEN. 2016;10:1616-1624](https://ieeexplore.ieee.org/abstract/document/7812215).
8.	[Lee D, Baldick R. Short-term wind power ensemble prediction based on Gaussian processes and neural networks. IEEE T SMART GRID. 2013;5:501-510](https://ieeexplore.ieee.org/abstract/document/6606922/).

9.	[Landry M, Erlinger TP, Patschke D, Varrichio C. Probabilistic gradient boosting machines for GEFCom2014 wind forecasting. INT J FORECASTING. 2016;32:1061-1066.]()
