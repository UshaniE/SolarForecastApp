# Import basic modules
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
#import plotly.graph_objects as go

from datetime import datetime
from datetime import timedelta

# Import regression and error metrics modules
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Standard scaler for preprocessing
from sklearn.preprocessing import StandardScaler


st.header("Forecasting Solar Photovoltaics (PV) Power Generation Using Machine Learning")

st.write("Model : **Multiple Linear Regression**")

st.sidebar.image("Picture1.jpg", use_column_width=True)

with st.sidebar:
     st.subheader("ML Models")
     st.write("   - Multiple Linear Regression")
     st.write("   - Ridge Linear Regression")
     st.write("   - Elastic Net Linear Regression")
     st.write("   - Random Forest")
     st.write("   - Seasonal AutoRegressive Integrated Moving Average with eXogenous (SARIMAX)")
     st.subheader("Most Suitable Model")
     st.write("SARIMAX")
     st.subheader("Data Source")
     st.write('Buruthakanda Solar Park, Sri Lanka')
     st.write('Hourly data from 2011-07-21 to 2013-12-31')

#date = st.sidebar.selectbox(label = "Select a Date")
# Enter start date
#startdate = st.date_input(label = "Enter the start date for forecast",value=datetime(2013,9,1),
# min_value=datetime(2013,8,22), max_value=datetime(2013,12,24))
#st.write(start)

#enddate = st.date_input(label = "Enter the end date for forecast",value=datetime(2013,9,10),
# min_value=datetime(2013,9,1), max_value=datetime(2013,12,30))

# Functions
# Train test split  
def train_test(data, test_size = 0.15, scale = False, cols_to_transform=None, include_test_scale=False):

    df = data.copy()
    # get the index after which test set starts
    test_index = int(len(df)*(1-test_size))
    
    # StandardScaler fit on the entire dataset
    if scale and include_test_scale:
        scaler = StandardScaler()
        df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])
        
    X_train = df.drop('PowerAvg', axis = 1).iloc[:test_index]
    y_train = df.PowerAvg.iloc[:test_index]
    X_test = df.drop('PowerAvg', axis = 1).iloc[test_index:]
    y_test = df.PowerAvg.iloc[test_index:]
    
    # StandardScaler fit only on the training set
    if scale and not include_test_scale:
        scaler = StandardScaler()
        X_train[cols_to_transform] = scaler.fit_transform(X_train[cols_to_transform])
        X_test[cols_to_transform] = scaler.transform(X_test[cols_to_transform])
    
    return X_train, X_test, y_train, y_test

st.subheader('Observed and Forecasted Power Generation')

# Load data
df_mlr = pd.read_csv('Data_Model.csv', index_col = 'Date', infer_datetime_format=True)
#st.write(df_mlr)

# creating categorical columns for linear regression 
cat_cols = ['Year', 'Month', 'Day', 'Hour','MeanWD','MaxGustWD']
for col in cat_cols:
    df_mlr[col] = df_mlr[col].astype('category')

# Preparing dummy columns for use in sklearn's linear regression 
df_mlr= pd.get_dummies(df_mlr, drop_first = True)


cols_to_transform = ['MeanWS','MaxGustWS','Precipitation','GlobalSolarRadiation','TiltSolarIrradiance',
                     'DiffuseSolarIrradiance','OpenAirTempAvg','ModuleTempAvg']
X_train, X_test, y_train, y_test = train_test(df_mlr, test_size = 0.15, scale = True, cols_to_transform=cols_to_transform)

# load saved model
with open('MLR.pkl' , 'rb') as f:
    lr = pickle.load(f)


# plot the predictions for last 7 days of the test set
y_actualsel = y_test.loc['2013-12-23':'2014-01-01']
X_predsel = X_test.loc['2013-12-23':'2014-01-01']
y_actualplot = y_test.loc['2013-12-15':'2014-01-01']
y_predsel = lr.predict(X_predsel)

fig,axes = plt.subplots(figsize = (15,7))
axes.plot(y_actualplot.index, y_actualplot, label='Observed')
axes.plot(y_actualsel.index, y_predsel, color='r', label='Forecast')
    
# set labels, legends and show plot
axes.set_xlabel('Date')
axes.set_ylabel('Power Generation in kW')
axes.set_xticks(np.arange(0, len(y_actualplot.index), 24))
axes.tick_params(axis='x',labelrotation = 90)
axes.legend()  

st.write(fig)  

st.subheader('Model Performance Matrices')

# Performance evaluation
pred = lr.predict(X_test)

RMSE = round(np.sqrt(mean_squared_error(y_test, pred)),2)
R2 = round(r2_score(y_test, pred),2)
MAE = round(mean_absolute_error(y_test, pred),2)
  
col1,col2,col3 = st.columns(3)
col1.metric('RMSE',RMSE)
col2.metric('R2',R2)
col3.metric('MAE',MAE)

