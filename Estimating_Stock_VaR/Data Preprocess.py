# %%%%
import pandas as pd
import numpy as np
import re

# %%%% functions
## Fill missing values 
def fillmissing(x,col,index,benchmark):
    for i in range(index,len(x)):
        # find missing value
        if x.loc[i,col] == benchmark:
            # if first is missing, fill using the value next to it
            if i == index:
                x.loc[i,col] = x.loc[i+1,col]
            # if the last one is missing, fill using the value preceeds it
            elif i == len(x)-1:
                x.loc[i,col] = x.loc[i-1,col]
            # otherwise, fill using the average of the two not null values above and after
            else:        
                j = i-1
                k = i+1
                while x.loc[j,col] == benchmark:
                    j -= 1
                while x.loc[k,col] == benchmark:
                    k += 1
                x.loc[i,col] = np.mean([x.loc[j,col],x.loc[k,col]])
    return x

## Data Preprocess
def preprocess(x,name,Date,column,index,benchmark,q):
    # select the valid starting day
    x = x[x['Date'] > Date].copy()
    x = x.reset_index().copy()
    x = x.drop('index',axis = 1).copy()
    
    # fill na with benchmark we chose
    x[column] = x[column].fillna(benchmark).copy()
    # fill missing values
    x = fillmissing(x,column,index,benchmark).copy()
    
    # calculate daily return
    x['lag_'+column] = x[column].shift(1)
    x = x.iloc[1:,:].copy().reset_index()
    x = x.drop('index',axis = 1).copy()
    x['log_ret'] = np.log(x[column])-np.log(x['lag_'+column])
    retm = np.mean(x['log_ret'])
    x['retv'] = np.square(x['log_ret']-retm)*100 
    
    # estimate volatility
    x[name+'_20day_vol'] = np.sqrt(x['retv'].rolling(window=20,win_type="boxcar").mean())/10
    
    # estimate quantiles of the distribution of log-returns
    x[name+'_quant_ret'] = np.nan
    for r in range(len(x)-20):
        R_quant = np.quantile(x['log_ret'][r:r+20],q)
        x.loc[r+19,name+'_quant_ret'] = R_quant
    return x

# %%%% Main Dataset: csi300 
csi = pd.read_csv('/Users/msstark/Desktop/project/Shanghai Shenzhen CSI 300 Historical Data.csv')
# setting date format
csi['Date']  = csi['Date'].apply(lambda x: re.sub(r',',r'',x))
csi['Day']   = csi['Date'].apply(lambda x: x.split(' ')[1]).astype(int)
csi['Month'] = csi['Date'].apply(lambda x: x.split(' ')[0])
csi['Month'].unique()
csi['Month'] = csi['Month'].map({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})
csi['Year'] = csi['Date'].apply(lambda x: x.split(' ')[2]).astype(int)
csi['Date'] = csi['Year'].astype(str) +'-'+csi['Month'].astype(str)+'-'+csi['Day'].astype(str)
csi['Date'] = pd.to_datetime(csi['Date'], format='%Y-%m-%d')
csi = csi.rename(columns = {'Price':'Close'}).copy()

# convert object type to float
col = ['Close','Open','High','Low']
for c in col:
    csi[c] = csi[c].apply(lambda x: re.sub(r',',r'',x)).astype('float')
csi['log_dsprd'] = np.log(csi['High'] - csi['Low'])
csi.columns

# apply preprocess function
csi = preprocess(csi,'csi','2005-01-03','Close',0,0,0.025).copy()

# %%%% spot exchange rate
xr  = pd.read_csv('/Users/msstark/Desktop/project/DEXCHUS.csv')

# setting date format
xr['DATE']  = pd.to_datetime(xr['DATE'], format='%Y-%m-%d')
xr = xr.rename(columns = {'DATE':'Date','DEXCHUS':'exR'}).copy()


# we find there's '.' inside our dataset
# replace '.' with '0', which is also the benchmark we chose to fill the missing values
xr['exR'] = xr[['exR']].apply(lambda x: x.replace('.','0'))

# convert object type to float
xr['exR'] = xr['exR'].astype(float)

# apply preprocess function
xr = preprocess(xr,'exR','2005-01-03','exR',0,0,0.025).copy()

# merge onto the main dataset
csi = csi.merge(xr[['Date','exR_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# %%%% hsi
hsi = pd.read_csv('^HSI.csv')

# setting date format
hsi['Date']  = pd.to_datetime(hsi['Date'], format='%Y-%m-%d')

# apply preprocess function
hsi = preprocess(hsi,'hsi','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi = csi.merge(hsi[['Date','hsi_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# %%%% sse
sse = pd.read_csv('SSE Composite Index.csv')

# setting date format
sse['Date']  = pd.to_datetime(sse['Date'], format='%Y-%m-%d')

# apply preprocess function
sse = preprocess(sse,'sse','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi = csi.merge(sse[['Date','sse_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# %%%% commodities
# corn
corn = pd.read_csv('corn-prices-historical-chart-data.csv')
corn = corn.rename(columns = {'date':'Date',' value':'Close'})

# setting date format
corn['Date']  = pd.to_datetime(corn['Date'], format='%Y-%m-%d')

# apply preprocess function
corn = preprocess(corn,'corn','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi = csi.merge(corn[['Date','corn_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# soybean
soybean = pd.read_csv('soybean-prices-historical-chart-data.csv')
soybean = soybean.rename(columns = {'date':'Date',' value':'Close'})

# setting date format
soybean['Date']  = pd.to_datetime(soybean['Date'], format='%Y-%m-%d')

# apply preprocess function
soybean = preprocess(soybean,'soybean','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi = csi.merge(soybean[['Date','soybean_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# %%%% heating oil
heat = pd.read_csv('New_York_Harbor_No._2_Heating_Oil_Spot_Price_FOB.csv')
heat = heat.rename(columns = {'Day':'Date','New York Harbor No. 2 Heating Oil Spot Price FOB  Dollars per Gallon':'Close'})

# setting date format
heat['Date']  = heat['Date'].apply(lambda x: re.sub(r'\/',r'-',x))
heat['Date']  = pd.to_datetime(heat['Date'], format='%m-%d-%Y')
heat = heat.sort_values(by=['Date'],ascending=True).reset_index().copy()
heat = heat.drop('index',axis = 1).copy()

# apply preprocess function
heat = preprocess(heat,'heat','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi  = csi.merge(heat[['Date','heat_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# %%%% 10-year bond yield rate
bond = pd.read_csv('China 10-Year Bond Yield Historical Data.csv')
bond = bond.rename(columns = {'Price':'Close'})

# setting date format
bond['Date']  = bond['Date'].apply(lambda x: re.sub(r',',r'',x))
bond['Day']   = bond['Date'].apply(lambda x: x.split(' ')[1]).astype(int)
bond['Month'] = bond['Date'].apply(lambda x: x.split(' ')[0])
bond['Month'] = bond['Month'].map({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})
bond['Year']  = bond['Date'].apply(lambda x: x.split(' ')[2]).astype(int)
bond['Date']  = bond['Year'].astype(str) +'-'+bond['Month'].astype(str)+'-'+bond['Day'].astype(str)
bond['Date']  = pd.to_datetime(bond['Date'], format='%Y-%m-%d')
bond = bond.sort_values(by=['Date'],ascending=True).reset_index().copy()
bond = bond.drop('index',axis = 1).copy()

# apply preprocess function
bond = preprocess(bond,'bond','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi  = csi.merge(bond[['Date','bond_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# %%%% metal
# platinum
platinum = pd.read_csv('platinum-prices-historical-chart-data.csv')
platinum = platinum.rename(columns = {'date':'Date',' value':'Close'})

# setting date format
platinum['Date'] = pd.to_datetime(platinum['Date'], format='%Y-%m-%d')

# apply preprocess function
platinum = preprocess(platinum,'platinum','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi  = csi.merge(platinum[['Date','platinum_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()


# palladium
palladium = pd.read_csv('palladium-prices-historical-chart-data.csv')
palladium = palladium.rename(columns = {'date':'Date',' value':'Close'})

# setting date format
palladium['Date'] = pd.to_datetime(palladium['Date'], format='%Y-%m-%d')

# apply preprocess function
palladium = preprocess(palladium,'palladium','2005-01-03','Close',0,0,0.025).copy()

# merge onto the main dataset
csi  = csi.merge(palladium[['Date','palladium_quant_ret']],left_on = ['Date'],right_on = ['Date'],how = 'left').copy()

# %%%% Final Clean
csi.columns

# select useful features
data_col = ['Date', 'Day', 'Month', 'log_dsprd', 'log_ret', 'retv', 'csi_20day_vol', 
            'csi_quant_ret', 'exR_quant_ret', 'hsi_quant_ret','sse_quant_ret', 'corn_quant_ret', 
            'soybean_quant_ret','heat_quant_ret', 'bond_quant_ret', 'platinum_quant_ret',
            'palladium_quant_ret']
new_data = csi[data_col].copy()

# rename columns
data_col = list(map(lambda x: x.replace('quant_ret','rstar'), data_col))
new_data.columns = data_col


# save merged raw values
new_data.to_csv('csi_updated.csv')

# %%%% Fill missing values

# setting filling benchmark, leave the first 20 rows out because they are caused by rolling calculation
new_data.iloc[19:] = new_data.iloc[19:].fillna(np.inf).copy()
new_data.isna().sum()

# fill missing valus for each column
col_fill = ['exR_rstar','hsi_rstar', 'sse_rstar', 'corn_rstar','soybean_rstar', 
            'heat_rstar', 'bond_rstar','platinum_rstar','palladium_rstar']
for c in col_fill:
    y = fillmissing(new_data,c,19,np.inf).copy()


# save the full dataset
new_data = y.copy()
new_data.to_csv('csi_updated_full.csv')
