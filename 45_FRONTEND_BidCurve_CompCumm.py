
import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from TFM_PredCurve_Tools import bid_comp_cumm

sns.set_theme(style='darkgrid')

st.write('# UNIT BID CURVES AGGREGATED')

unit_path = './Data_Output/'

unit_type_list = ['COMBINED CYCLE', 'HYDRAULIC']

unit_type = st.selectbox('SELECT THE UNIT TECHNOLOGY', unit_type_list)

if unit_type == 'COMBINED CYCLE':
    df_units = pd.read_csv(unit_path + 'df_units_CC_112019_102020.csv',index_col=0) 
else:
    df_units = pd.read_csv(unit_path + 'df_units_HYD_112019_102020.csv',index_col=0)

df_units['Date'] = df_units['Date'].astype('datetime64[ns]')

year_list = df_units['Year'].unique()
year = st.selectbox('SELECT A YEAR', year_list)
    
month_list = df_units[df_units['Year']==year]['Month'].unique()
month = st.selectbox('SELECT A MONTH', month_list)
    
day_list = df_units[(df_units['Year']==year) & 
                    (df_units['Month']==month)]['Day'].unique()
day = st.selectbox('SELECT A DAY', day_list)
    
date = str(year) + '-' + str(month) + '-' + str(day)

hour_list = df_units[(df_units['Year']==year) & 
                     (df_units['Month']==month) & 
                     (df_units['Day']==day)]['Period'].unique()
hour = st.selectbox('SELECT AN HOUR', hour_list)


st.set_option('deprecation.showPyplotGlobalUse', False)

my_chart = bid_comp_cumm(df_units, date, hour)
st.pyplot(my_chart)
