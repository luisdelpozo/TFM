
import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from TFM_PredCurve_Tools import plot_24bids

sns.set_theme(style='darkgrid')

st.write('# UNIT DAY BID CURVES')

unit_path = './Data_Output/'

unit_type_list = ['COMBINED CYCLE', 'HYDRAULIC']

unit_list = ['PALOS1',
             'PALOS2',
             'PALOS3',
             'SAGUNTO1',
             'ARCOS1',
             'COLON4',
             'ESCATRON3',
             'ALGECIRAS3',
             'CASTELNOU']

unit_hyd_list = ['AGUAYO_GEN',
                 'TAJOENCANT',
                 'MORALETS',
                 'MUELA']

unit_type = st.selectbox('SELECT THE UNIT TECHNOLOGY', unit_type_list)

if unit_type == 'COMBINED CYCLE':
    unit = st.selectbox('SELECT A UNIT', unit_list)
else:
    unit = st.selectbox('SELECT A UNIT', unit_hyd_list)
    
df_unit = pd.read_csv(unit_path + unit + '_DataFrame.csv',index_col=0)
    
year_list = df_unit['Year'].unique()
year = st.selectbox('SELECT A YEAR', year_list)
    
month_list = df_unit[df_unit['Year']==year]['Month'].unique()
month = st.selectbox('SELECT A MONTH', month_list)
    
day_list = df_unit[(df_unit['Year']==year) & 
                    (df_unit['Month']==month)]['Day'].unique()
day = st.selectbox('SELECT A DAY', day_list)
    
date = str(year) + '-' + str(month) + '-' + str(day)

df_unit['Date'] = df_unit['Date'].astype('datetime64[ns]')

df_plot = df_unit[(df_unit['Date'] == date)]

st.set_option('deprecation.showPyplotGlobalUse', False)

my_chart = plot_24bids(df_plot, date)
st.pyplot(my_chart)
