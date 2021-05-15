import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime, timedelta


#Function created for showing for a certain day and hour of df_curve

def data_date_hour_info(data, date, hour):
    return data[(data['Date']==date) & (data['Period']==hour)]


def convert_to_df_curve(data):
    df_curve = data[['Pot_max', 'Year', 'Month', 'Day', 'Period', 'Block', 'Price', 'Energy']]

    #Including date and week day per each day.
    #Including date
    df_curve['Date']= pd.to_datetime(df_curve[['Year', 'Month', 'Day']])

    #Including weekdays as it is an important information
    df_curve['Weekday'] = df_curve['Date'].apply(lambda x: x.weekday())
    weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
    df_curve['Weekday'] = df_curve['Weekday'].apply(lambda x: weekDays[x])

    #Including the total Energy for all blocks in a certain hour (to be able to plot the bid curve)
    df_curve['Energy_tot'] = df_curve.groupby(['Date','Period'])['Energy'].cumsum()

    #Sorting the dataframe by date, hour and block 
    df_curve = df_curve.sort_values(['Date','Period','Block']).reset_index(drop=True)
    
    return df_curve


#Searching the days with 23h and 25h within the original data period. 
#It is known that in Spain times changes from 2am to 3am every last Sunday of March and 
#from 3am to 2am every last Sunday of October

def days_25h_23h(data):

    start_year = data['Year'].min()
    end_year = data['Year'].max()

    days_23h = []
    days_25h = []

    for yx in range(start_year,end_year+1):
    
        m_min = data['Month'][data['Year']==yx].min()
        m_max = data['Month'][data['Year']==yx].max()
    
        if 10 in range(m_min,m_max+1):
            October_date = datetime(yx,10,31)
            offset_October = (October_date.weekday() - 6)%7
            last_October_sunday = October_date - timedelta(days=offset_October)
            days_25h += [last_October_sunday]
 
        if 3 in range(m_min,m_max+1):
            March_date = datetime(yx,3,31)
            offset_March = (March_date.weekday() - 6)%7
            last_March_sunday = March_date - timedelta(days=offset_March)
            days_23h += [last_March_sunday]

    return days_25h,days_23h


def plot_bid_curve(dataframe, date, hour):
    df_plot = dataframe[(dataframe['Date'] == date) & (dataframe['Period'] == hour)]
    return plt.plot(pd.Series(0).append(df_plot['Energy_tot']), 
                    pd.Series(df_plot['Price'].iloc[0]).append(df_plot['Price']), 
                    drawstyle='steps', 
                    label='steps (=steps-pre)')


#Function to plot marginal price for a chosen dataframe, date and period (hour)

def plot_marginal_price(dataframe, date, hour):
    df_plot = dataframe[(dataframe['Date'] == date) & (dataframe['Period'] == hour)]
    return plt.plot(pd.Series(0).append(df_plot['Energy_tot']), 
                    pd.Series(df_plot['Marg_Price'].iloc[0]).append(df_plot['Marg_Price']), 
                    drawstyle='steps', 
                    label='steps (=steps-pre)')


#Function to plot bid curves for a chosen dataframe and date

def plot_bid_curve_day(dataframe, date):
    df_plot = dataframe[(dataframe['Date'] == date)]
    df_plot['Energy_tot_date'] = df_plot['Energy'].cumsum()
    return plt.plot(pd.Series(0).append(df_plot['Energy_tot_date']), 
                    pd.Series(df_plot['Price'].iloc[0]).append(df_plot['Price']), 
                    drawstyle='steps', 
                    label='steps (=steps-pre)')


#Function to plot marginal price for a chosen dataframe and date

def plot_marginal_price_day(dataframe, date):
    df_plot = dataframe[(dataframe['Date'] == date)]
    df_plot['Energy_tot_date'] = df_plot['Energy'].cumsum()
    return plt.plot(pd.Series(0).append(df_plot['Energy_tot_date']), 
                    pd.Series(df_plot['Marg_Price'].iloc[0]).append(df_plot['Marg_Price']), 
                    drawstyle='steps', 
                    label='steps (=steps-pre)')


def df_structure_24h(year_start, month_start, day_start, year_end, month_end, day_end, block_max):

    '''This function creates a empty structure of days, hours and blocks between two dates
    without 23h & 25h days'''
    
    #Adding the days of between the staring and ending days
    
    start = datetime(year_start,month_start,day_start)
    end = datetime(year_end,month_end,day_end)

    date_list = [start + timedelta(days=d) for d in range((end - start).days + 1)] 

    structure = pd.DataFrame({'Date' : date_list})
    structure['Year'] = structure['Date'].apply(lambda x: x.year)
    structure['Month'] = structure['Date'].apply(lambda x: x.month)
    structure['Day'] = structure['Date'].apply(lambda x: x.day)
    structure['Period'] = 1
    structure['Block'] = 1

    #Adding 24 hours per each day
    structure_tot = pd.DataFrame()

    for hour in range(1,25):
        structure_new = structure.copy()
        structure_new['Period'] = hour
        structure_tot = pd.concat([structure_tot,structure_new])

    structure_tot = structure_tot.reset_index(drop=True)

    #Adding maximum number of blocks per each period (hour)

    structure_block = pd.DataFrame()

    for block in range(1,block_max+1):
        structure_new = structure_tot.copy()
        structure_new['Block'] = block
        structure_block = pd.concat([structure_block,structure_new])
    
    structure_block = structure_block.sort_values(['Date','Period','Block'])
    structure_block = structure_block.reset_index(drop=True)
    return structure_block



def df_structure(year_start, month_start, day_start, year_end, month_end, day_end, block_max):

    '''This function creates a empty structure of days, hours and blocks between two dates'''
    
    #Adding the days between the staring and ending days
    
    start = datetime(year_start,month_start,day_start)
    end = datetime(year_end,month_end,day_end)

    date_list = [start + timedelta(days=d) for d in range((end - start).days + 1)] 

    structure = pd.DataFrame({'Date' : date_list})
    structure['Year'] = structure['Date'].apply(lambda x: x.year)
    structure['Month'] = structure['Date'].apply(lambda x: x.month)
    structure['Day'] = structure['Date'].apply(lambda x: x.day)
    structure['Period'] = 1
    structure['Block'] = 1

    #Adding 24 hours per each day
    structure_tot = pd.DataFrame()

    for hour in range(1,25):
        structure_new = structure.copy()
        structure_new['Period'] = hour
        structure_tot = pd.concat([structure_tot,structure_new])

    structure_tot = structure_tot.reset_index(drop=True)

    #Deleting hour 24 for the 4th Sunday of March and adding hour 25 for the 4th Sunday of October of each year
    y_min = structure_tot['Year'].min()
    y_max = structure_tot['Year'].max()

    for yx in range(y_min,y_max+1):
    
        m_min = structure_tot['Month'][structure_tot['Year']==yx].min()
        m_max = structure_tot['Month'][structure_tot['Year']==yx].max()
    
        if 10 in range(m_min,m_max+1):
            October_date = datetime(yx,10,31)
            offset_October = (October_date.weekday() - 6)%7
            last_October_sunday = October_date - timedelta(days=offset_October)
            structure_tot = structure_tot.append({'Date':datetime(yx,
                                                              last_October_sunday.month,
                                                              last_October_sunday.day),
                                              'Year': yx, 
                                              'Month': last_October_sunday.month, 
                                              'Day': last_October_sunday.day,
                                              'Period': 25,
                                              'Block': 1}, 
                                              ignore_index=True)    
 
        if 3 in range(m_min,m_max+1):
            March_date = datetime(yx,3,31)
            offset_March = (March_date.weekday() - 6)%7
            last_March_sunday = March_date - timedelta(days=offset_March)

            structure_tot = structure_tot.drop(structure_tot[
                                            (structure_tot['Year']== yx)& 
                                            (structure_tot['Month']== last_March_sunday.month)& 
                                            (structure_tot['Day']== last_March_sunday.day)&
                                            (structure_tot['Period']== 24)&
                                            (structure_tot['Block']== 1)].index)

    #Adding the the maximum number of blocks per each period (hour)

    structure_block = pd.DataFrame()

    for block in range(1,block_max+1):
        structure_new = structure_tot.copy()
        structure_new['Block'] = block
        structure_block = pd.concat([structure_block,structure_new])
    
    structure_block = structure_block.sort_values(['Date','Period','Block'])
    structure_block = structure_block.reset_index(drop=True)
    return structure_block

def area_pred_curve_summary(y, y_pred):
    
    y_TEST_area_tot = []
    y_area_tot_diff_TEST_abs = []
    y_area_tot_diff_TEST_rel = []

    y_TEST_area = []
    y_area_diff_TEST_abs = []
    y_area_diff_TEST_rel = []
    
    hours = len(y)
    
    for hour in range(0, hours):

        y_dE_TEST_area = (y[hour][:,0]*y[hour][:,1])
        y_dE_TEST_pred_area = (y_pred[hour][:,0]*y_pred[hour][:,1])
        area_diff_TEST_abs = abs(y_dE_TEST_area - y_dE_TEST_pred_area).sum() #Total Absolute Area Differences

        y_dE_TEST_area_tot = y_dE_TEST_area.sum() #Total area for the whole set period, for each hour 
        y_dE_TEST_pred_area_tot = y_dE_TEST_pred_area.sum()
        area_tot_diff_TEST_abs = (y_dE_TEST_area_tot - y_dE_TEST_pred_area_tot) #Total Differences of Total Areas

        #Relative area difference
        area_diff_TEST_rel = (area_diff_TEST_abs / y_dE_TEST_area_tot) * 100
        area_tot_diff_TEST_rel = (area_tot_diff_TEST_abs / y_dE_TEST_area_tot) * 100

        y_area_diff_TEST_abs.append(area_diff_TEST_abs)
        y_area_diff_TEST_rel.append(area_diff_TEST_rel)

        y_TEST_area_tot.append(y_dE_TEST_area_tot)
        y_area_tot_diff_TEST_abs.append(area_tot_diff_TEST_abs)
        y_area_tot_diff_TEST_rel.append(area_tot_diff_TEST_rel)

    df_y_TEST_area_summary = pd.DataFrame({'Area_tot(€)': y_TEST_area_tot,
                  'Area_diff_abs(€)': y_area_diff_TEST_abs, 
                  'Area_diff_rel(%)': y_area_diff_TEST_rel,
                  'Area_tot_diff_abs(€)': y_area_tot_diff_TEST_abs, 
                  'Area_tot_diff_rel(%)': y_area_tot_diff_TEST_rel})
    return df_y_TEST_area_summary



#List of df with the comparison of real data and prediction for every hour

def df_pred_summary(y, y_pred):

    df_pred_summary = []
    
    hours = len(y)

    for hour in range(0, hours):

        df_pred = pd.DataFrame(np.concatenate([y[hour],y_pred[hour]],axis=1), 
                                       columns= ['dE','Price','dE_pred','Price_pred'])
        df_pred['dE_diff'] = df_pred['dE'] - df_pred['dE_pred']
        df_pred['Price_diff'] = df_pred['Price'] - df_pred['Price_pred']
        df_pred['Area'] = df_pred['dE'] * df_pred['Price']
        df_pred['Area_pred'] = df_pred['dE_pred'] * df_pred['Price_pred']
        df_pred['Area_diff'] = df_pred['Area'] - df_pred['Area_pred']
        df_pred['Area_diff_rel(%)'] = df_pred['Area_diff']/df_pred['Area']*100
        df_pred['Period'] = hour + 1
        
        df_pred_summary.append(df_pred)
        
    # Creating a df with all the comparisons (to be able to retreive info easily)

    df_y_pred_summary = pd.DataFrame()

    for hour in range(0, len(df_pred_summary)):
        df_y_pred_summary = pd.concat([df_y_pred_summary,df_pred_summary[hour]])
        
    return df_y_pred_summary



def df_pred_summary_noPmax(df):
    
    df_noPmax = df[df['Price'] != df['Price'].max()]
    
    return df_noPmax



def data_report(df):
    first_day = datetime(df['Date'].min().year,
                         df['Date'].min().month,
                         df['Date'].min().day)

    last_day = datetime(df['Date'].max().year,
                        df['Date'].max().month,
                        df['Date'].max().day)

    num_days_real = (last_day - first_day).days + 1
    num_days = len(df['Date'].value_counts())
    diff_num_days = num_days_real - num_days #Number of missing days in original data
    diff_num_days_rel = diff_num_days / num_days_real * 100

    num_hours_real = num_days_real*24
    num_hours_orig_real = num_days*24
    #num_hours = df.groupby(['Date','Period'])['Block'].count().value_counts().sum()
    diff_num_hours = num_hours_real - num_hours_orig_real
    diff_num_hours_rel = diff_num_hours / num_hours_real * 100
    #diff_num_hours_orig = num_hours_orig_real - num_hours

    print('Num. of days: {}'.format(num_days_real))
    print('Num. of days with bid: {}'.format(num_days))
    print('Num. of missing days (abs/%): {} / {:.2f}%'.format(diff_num_days, diff_num_days_rel))
    print('Num. of hours: {}'.format(num_hours_real))    
    print('Num. of hours with bid: {}'.format(num_hours_orig_real))
    print('Num. of missing hours (abs/%): {} / {:.2f}%'.format(diff_num_hours, diff_num_hours_rel))
    #print('Num. of missing hours from non-missing days: {}'.format(diff_num_hours_orig))


def data_report_total(df, start, end):
    
    start_day = start
    end_day =end
    num_days_total = (end_day - start_day).days + 1
    
    first_day = datetime(df['Date'].min().year,
                         df['Date'].min().month,
                         df['Date'].min().day)
    last_day = datetime(df['Date'].max().year,
                        df['Date'].max().month,
                        df['Date'].max().day)
    num_days_real = (last_day - first_day).days + 1
    
    num_days_bid = len(df['Date'].value_counts())
    
    diff_num_days = num_days_total - num_days_bid #Number of missing days in original data
    diff_num_days_rel = diff_num_days / num_days_total * 100

    num_hours_total = num_days_total*24
    num_hours_real = num_days_real*24
    num_hours_bid = df.groupby(['Date','Period'])['Block'].count().value_counts().sum()
    diff_num_hours = num_hours_total - num_hours_bid
    diff_num_hours_rel = diff_num_hours / num_hours_total * 100
    #diff_num_hours_orig = num_hours_orig_real - num_hours

    print('Num. of total days: {}'.format(num_days_total))
    #print('Num. of observed days: {}'.format(num_days_real))
    print('Num. of days with bid: {}'.format(num_days_bid))
    print('Num. of missing days (abs/%): {} / {:.2f}%'.format(diff_num_days, diff_num_days_rel))
    print('Num. of total hours: {}'.format(num_hours_total))
    #print('Num. of observed hours: {}'.format(num_hours_real))    
    print('Num. of hours with bid: {}'.format(num_hours_bid))
    print('Num. of missing hours (abs/%): {} / {:.2f}%'.format(diff_num_hours, diff_num_hours_rel))
    #print('Num. of missing hours from non-missing days: {}'.format(diff_num_hours_orig))    
    

def missing_dates(d):
    date_set=set(d)
    missing_dates = [datetime.strftime(x, '%Y-%m-%d') for x in (d[0]+timedelta(x) for x in range((d[-1]-d[0]).days)) if x not in date_set]

    return missing_dates   