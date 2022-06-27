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



# Functions
def show_predict_MLR():
    
    ok = st.button("Multiple Linear Regression")
    if ok:
    # Load data
        X_test = pd.read_csv('X_test.csv', index_col = 'Date', infer_datetime_format=True)
        y_test = pd.read_csv('y_test.csv', index_col = 'Date', infer_datetime_format=True)

# load saved model
        with open('MLR.pkl' , 'rb') as f:
            lr = pickle.load(f)

        st.subheader("Multiple Linear Regression")
#st.write("""##### Observed and Forecasted Power Generation""")

        st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Observed and Forecasted Power Generation"}</h1>', unsafe_allow_html=True)

    # plot the predictions for last 7 days of the test set
        y_actualsel = y_test.loc['2013-12-23':'2014-01-01']
        X_predsel = X_test.loc['2013-12-23':'2014-01-01']
        y_actualplot = y_test.loc['2013-12-15':'2014-01-01']
        y_predsel = lr.predict(X_predsel)

        fig,axes = plt.subplots(figsize = (15,7))
        axes.plot(y_actualplot.index, y_actualplot, label='Observed')
        axes.plot(y_actualsel.index, y_predsel, color='r',  label='Forecast:MLR')

    # set labels, legends and show plot
        axes.set_xlabel('Date')
        axes.set_ylabel('Power Generation in kW')
        axes.set_xticks(np.arange(0, len(y_actualplot.index), 24))
        axes.tick_params(axis='x',labelrotation = 90)
        axes.legend()  

        st.write(fig)  

    #st.write('Model Performance Matrices')
        st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Model Performance Matrices"}</h1>', unsafe_allow_html=True)

    # Performance evaluation
        pred = lr.predict(X_test)

        RMSE = round(np.sqrt(mean_squared_error(y_test, pred)),2)
        R2 = round(r2_score(y_test, pred),2)
        MAE = round(mean_absolute_error(y_test, pred),2)
  
        col1,col2,col3 = st.columns(3)
        col1.metric('RMSE',RMSE)
        col2.metric('R2',R2)
        col3.metric('MAE',MAE)