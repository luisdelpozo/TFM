
import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from TFM_PredCurve_Tools import plot_bid_timeperiod

sns.set_theme(style='darkgrid')

st.write('# UNIT BID CURVES TIME PERIOD')

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

df_unit['Date'] = df_unit['Date'].astype('datetime64[ns]')

df_unit.rename(columns={"Period": "Hour"}, inplace = True)
df_unit['Date_Hour'] = pd.to_datetime(df_unit[['Year', 'Month', 'Day', 'Hour']])
df_unit_reduced = df_unit.set_index('Date_Hour') #New time index is created
#df_unit_reduced = df_unit_reduced[['Block', 'Hour', 'Energy_tot', 'Price', 'Marg_Price', 'NG_Price']]
df_unit_reduced = df_unit_reduced[['Block', 'Hour', 'Energy_tot', 'Price', 'Marg_Price']]

st.set_option('deprecation.showPyplotGlobalUse', False)

start_date = str(df_unit['Date'].min())[0:10]
end_date = str(df_unit['Date'].max())[0:10]

start_sel_block = df_unit['Block'].min()
end_sel_block = df_unit['Block'].max()

num_hours = df_unit.groupby(['Date','Hour'])['Block'].count().value_counts().sum()
serie_block_all = df_unit.groupby(['Block'])['Hour'].count()[df_unit.groupby(['Block'])['Hour'].count() == num_hours]
serie_block_all = serie_block_all.reset_index()
block_all = serie_block_all['Block'].min()

my_chart = plot_bid_timeperiod(df_unit_reduced, start_date, end_date, start_sel_block, end_sel_block, block_all)
st.pyplot(my_chart)
