# %%%%
import pandas as pd
import numpy as np
from pandas_datareader import data


# %%%% 

'''SP500 Sector Return'''

spSector = pd.read_csv('sp500sectors-monthly.csv')
spSector['Date'] = pd.to_datetime(spSector['Date']).copy()
spSector.set_index('Date',inplace=True)
spSector.columns = spSector.columns.str.replace(' ','_')
column_name = spSector.columns

# estimate monthly return
SpSector_shifted = spSector.apply(lambda x : x.shift(1))[1:]
# add suffix to column name
SpSector_shifted = SpSector_shifted.add_suffix('_lag')

# merge two dataframe
spSectorFull = spSector.merge(SpSector_shifted,left_index = True, right_index = True)

# calculate return
for c in column_name:
    if c != 'Real_Estate':
        spSectorFull[c+'_ret'] = spSectorFull[c]/spSectorFull[c+'_lag'] - 1.
    else:
        tmp = spSectorFull[spSectorFull[c+'_lag'].isna()==False].copy()
        #ret = tmp[c]/tmp[c+'_lag'] - 1.
        spSectorFull[c+'_ret'] = tmp[c]/tmp[c+'_lag'] - 1.
    retm = np.mean(spSectorFull[c+'_ret'])
    spSectorFull[c+'_var'] = np.square(spSectorFull[c+'_ret']-retm) 
        
        
spSectorFull.to_csv('S&P500_sector_return.csv')

# %%%% function used to process other index data
def preprocess(dta,name):
    dta.columns = dta.columns.str.replace(' ','_')
    col = dta.columns[1]
    tmp = dta[dta[col].isna() == False].copy()
    # set Date to date format
    tmp['Date'] = pd.to_datetime(tmp['Date'])
    # Sort by Date
    tmp = tmp.sort_values(by = ['Date']).copy().reset_index().drop(['index'],axis = 1)
    # estimate monthly change
    tmp[name+'_lag'] = tmp[col].shift(1)
    tmp = tmp[1:].copy().reset_index().drop(['index'],axis = 1)
    tmp[name+'_ret'] = tmp[col]/tmp[name+'_lag'] - 1.
    # Estimate variance
    retm = np.mean(tmp[name+'_ret'])
    tmp[name+'_var'] = np.square(tmp[name+'_ret']-retm)
    return tmp
      
    
# %%%%

'''CPI'''

cpi = pd.read_csv('/Users/msstark/Desktop/Brandeis/Field Project/data/CPI.csv')
cpi.dropna(inplace = True)
cpi = cpi.reset_index().drop(['index'],axis = 1)
cpi.drop('CPI INDX.1',axis = 1,inplace = True)
CPI = preprocess(cpi,'cpi')

CPI.to_csv('CPI_return.csv')

# %%%%

'''PMI'''

pmi = pd.read_csv('/Users/msstark/Desktop/Brandeis/Field Project/data/PMI.csv')
PMI = preprocess(pmi,'pmi')

PMI.to_csv('PMI_return.csv')

# %%%%

'''DXY'''
dxy = pd.read_excel('/Users/msstark/Desktop/Brandeis/Field Project/data/DXY index-month.xlsx',header=2)
dxy = dxy.iloc[:,:2].copy()
DXY = preprocess(dxy,'dxy')

DXY.to_csv('DYX_change.csv')


# %%%%

'''BCOM'''

bcom = pd.read_csv('/Users/msstark/Desktop/Brandeis/Field Project/data/Bloomberg Commodity Historical Data.csv')
#bcom = bcom.sort_index(ascending=False).copy().reset_index().drop('index',axis = 1)
bcom = bcom.iloc[1:-3,:2].copy().reset_index().drop(['index'],axis = 1)
bcom['Date'] = cpi['Date']
BCOM = preprocess(bcom,'bcom')

BCOM.to_csv('BCOM_return.csv')

