import streamlit as st
from predict_MLR import show_predict_MLR
from predict_RLR import show_predict_RLR

st.header("Forecasting Solar Photovoltaics (PV) Power Generation Using Machine Learning")

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

#st.selectbox("Select ML Model",("MLR","RLR"))

show_predict_MLR()
show_predict_RLR()