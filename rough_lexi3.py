#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 20:28:59 2021

@author: ratchanon
"""




import pandas as pd
import numpy as np
from tabulate import tabulate


#load file
df= pd.read_csv('/Users/ratchanon/Desktop/Python for master/data_column3.csv')
df.columns
df = df.drop('Unnamed: 0',axis=1)


df = df.round(decimals=4)


def income(x):
    df1 = df[df['INCOME'] >= x] 
    car = df1['CARBON']
    CHS = df1['CHSI']
    mncar = min(car)
    mxcar = max(car)
    mnchs = min(CHS)
    mxchs = max(CHS)
    return print('If income >=',f"{x:,}",'carbon is between',f"{mncar:,}",'and',f"{mxcar:,}",
                 '\n','and','CHSI is between',f"{mnchs:,}",'and',f"{mxchs:,}")


def carbon(x):
    df1 = df[df['CARBON'] >= x] 
    inc = df1['INCOME']
    CHS = df1['CHSI']
    mninc = min(inc)
    mxinc = max(inc)
    mnchs = min(CHS)
    mxchs = max(CHS)
    return print('If carbon >=',f"{x:,}",'income is between',f"{mninc:,}",'and',f"{mxinc:,}",
                 '\n','and','CHSI is between',f"{mnchs:,}",'and',f"{mxchs:,}")

def CHSI(x):
    df1 = df[df['CHSI'] >= x] 
    inc = df1['INCOME']
    CHS = df1['CARBON']
    mninc = min(inc)
    mxinc = max(inc)
    mnchs = min(CHS)
    mxchs = max(CHS)
    return print('If carbon >=',f"{x:,}",'income is between',f"{mninc:,}",'and',f"{mxinc:,}",
                 '\n','and','CHSI is between',f"{mnchs:,}",'and',f"{mxchs:,}")



min_in = min(df['INCOME'])
max_in = max(df['INCOME'])
min_car = min(df['CARBON'])
max_car = max(df["CARBON"])
min_chs = min(df['CHSI'])
max_chs = max(df['CHSI'])
print('The range of the objectives','\n'*2,
      'income  =',f"{min_in:,}",'and',f"{max_in:,}",'\n'*2,
      'carbon  =',f"{min_car:,}",'and', f"{max_car:,}",'\n'*2,
      'CHSI  =',f"{min_chs:,}",'and', f"{max_chs:,}",'\n')

#    ask_car = float(input('How much the carbon'))
#    ask_chs = float(input('How much the chsi'))

from tabulate import tabulate

from prophet import Prophet


index = df.index
number_of_rows = len(index)
unit_in = []
unit_car = []
unit_chs = []

while number_of_rows > 15:
    ask_in = float(input('How much would you like for the income? : '))
    print('')
    income(ask_in)
    ask_car = float(input('How much would you like for the carbon? : '))
    df = df[df['INCOME'] >= ask_in]
    df = df[df['CARBON'] >= ask_car]
    min_chs = min(df['CHSI'])
    max_chs = max(df['CHSI'])
    print(' ')
    print('CHSI is between',f"{min_chs:,}",'and',f"{max_chs:,}",'\n','Please choose the one number between the range')
    ask_chs = float(input('How much would you like for the chsi? : '))
    df = df[df['CHSI'] >= ask_chs]
    index = df.index
    number_of_rows = len(index)

    unit_in.append(ask_in)
    unit_car.append(ask_car)
    unit_chs.append(ask_chs)
        
    if number_of_rows <= 15:
        print('Please choose the final solution')
        print(tabulate(df, headers='keys', tablefmt='psql'))

            
    else:    
        print(' ','\n'*2)
        print('There are still much solutions to consider!!','\n'*2,'the new range is below','\n'*2)
        min_in = min(df['INCOME'])
        max_in = max(df['INCOME'])
        min_car = min(df['CARBON'])
        max_car = max(df["CARBON"])
        min_chs = min(df['CHSI'])
        max_chs = max(df['CHSI'])
        print('The range of the objectives','\n'*2,
      'income  =',f"{min_in:,}",'and',f"{max_in:,}",'\n'*2,
      'carbon  =',f"{min_car:,}",'and', f"{max_car:,}",'\n'*2,
      'CHSI  =',f"{min_chs:,}",'and', f"{max_chs:,}",'\n')
   
    
    

begin_date = '2019-10-16'


#Income suggestion
df1 = pd.DataFrame({
                   'ds':pd.date_range(begin_date, periods=len(unit_in)),
                   'y':unit_in})
m = Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
m.fit(df1)
future = m.make_future_dataframe(1)
future.tail()
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
fig1 = m.plot(forecast)

sug_in = forecast['yhat'].iloc[-1]
Final_table_in = df['INCOME']
index_sug_in = np.argmin(np.abs(np.array(Final_table_in)-sug_in))





#Carbon suggestion
df2 = pd.DataFrame({
                   'ds':pd.date_range(begin_date, periods=len(unit_car)),
                   'y':unit_car})
m2 = Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
m2.fit(df2)
future2 = m.make_future_dataframe(1)
future2.tail()
forecast2 = m2.predict(future)
forecast2[['ds','yhat','yhat_lower','yhat_upper']].tail()
fig2 = m2.plot(forecast2)

sug_car = forecast2['yhat'].iloc[-1]
Final_table_car = df['CARBON']
index_sug_car = np.argmin(np.abs(np.array(Final_table_car)-sug_car))


#CHS suggestion
df3 = pd.DataFrame({
                   'ds':pd.date_range(begin_date, periods=len(unit_chs)),
                   'y':unit_chs})
m3 = Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
m3.fit(df3)
future3 = m3.make_future_dataframe(1)
future3.tail()
forecast3 = m3.predict(future)
forecast3[['ds','yhat','yhat_lower','yhat_upper']].tail()
fig3 = m3.plot(forecast3)

sug_chs = forecast3['yhat'].iloc[-1]
Final_table_chs = df['CHSI']
index_sug_chs = np.argmin(np.abs(np.array(Final_table_chs)-sug_chs))

L1 = [index_sug_in,index_sug_car,index_sug_chs]
table_sug = df.iloc[L1]

#Remove the dupplicated value

table_sug= table_sug[~table_sug.index.duplicated()]
    
print('The suggestion solution is')
print(tabulate(table_sug, headers='keys', tablefmt='psql'))

