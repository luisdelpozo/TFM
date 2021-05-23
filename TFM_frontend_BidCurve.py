
import streamlit as st
import altair as alt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime, timedelta

from TFM_PredCurve_Tools import data_date_hour_info, convert_to_df_curve, days_25h_23h
from TFM_PredCurve_Tools import data_report, data_report_total, missing_dates, bid_hour_summary
from TFM_PredCurve_Tools import plot_bid_curve, plot_marginal_price, plot_bid_margprice, plot_bid_margprice_day
from TFM_PredCurve_Tools import plot_bid_curve_day, plot_marginal_price_day, myplot, plot_24bids
from TFM_PredCurve_Tools import plot_bid_timeperiod, plot_bid_timeperiod_line
from TFM_PredCurve_Tools import plot_energy_timeperiod, plot_energy_timeperiod_line


st.write('# UNIT BID COMPARISON')

@st.cache
def get_cars():
    time.sleep(3)
    return data.cars()

@st.cache
def get_subset(origin, df):
    time.sleep(5)
    return df[df['Origin']== origin]


unit = st.selectbox('SELECT A UNIT', ['PALOS1', 'PALOS2'])
year = st.selectbox('SELECT A YEAR', [2019, 2020])
month = st.selectbox('SELECT A MONTH', [i for i in range(1, 13)])
day = st.selectbox('SELECT A DAY', [i for i in range(1, 32)])
hour = st.selectbox('SELECT AN HOUR', [i for i in range(1, 25)])

date = str(year) + '-' + str(month) + '-' + str(day)

#df_curve = st.file_uploader("/home/dsc/Repos/TFM/PALOS1_DataFrame.csv", type="csv")

df_curve = pd.read_csv('/home/dsc/Repos/TFM/PALOS1_DataFrame.csv',index_col=0)
df_curve['Date'] = df_curve['Date'].astype('datetime64[ns]')

#plot_bid_curve(df_curve, date, int(hour))

df_plot = df_curve[(df_curve['Date'] == date) & (df_curve['Period'] == hour)]

my_chart = alt.Chart(df_plot).mark_point().encode(x='Energy_tot', y='Price', color='Block').interactive()

st.altair_chart(my_chart)

#st.write('gracias por visitarme, %s' % name)
