# %%%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
# %%%%

dta_airbnb = pd.read_csv('dta_airbnb_clean.csv').drop(['Unnamed: 0'],axis=1)

## data without missing values
dta_comp = dta_airbnb.dropna().reset_index() # 8608 rows * 43 columns
del dta_comp['index']
dta_comp.columns

## main variables used for predition --- 36
dta_reg = dta_comp.drop(columns = ['headline','cancel_policy_code','location', 'location_code', 
                                   'type_code','amenity','res_time_code']).copy()

## set dummies for categorical variables
dta_cat  = dta_reg[['cancel_policy', 'type','response_time','location_approx']].copy()
dta_cat  = dta_cat.rename(columns = {'cancel_policy':'cancel','response_time':'res_time','location_approx':'location'})

dta_reg2 = pd.get_dummies(dta_cat,drop_first = True)

dta_reg  = pd.concat([dta_reg,dta_reg2],axis = 1).drop(columns = ['cancel_policy', 'type','response_time','location_approx'])

## expand main variables
dta_reg.columns

# intercation
tmp_inter    = dta_reg.iloc[:,1:].copy()
inter        = PolynomialFeatures(interaction_only=True)
interact     = pd.DataFrame(inter.fit_transform(tmp_inter.to_numpy()))
inter_name   = inter.get_feature_names(tmp_inter.columns)
tmp_interact = interact.set_axis(inter_name, axis=1, inplace=False).iloc[:,1:2883] # drop bias and location*location term

# drop interactions within the same dummy variable
def remove_dup(x):
    column_title = []
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            col = x[i] + ' ' + x[j] 
            column_title.append(col)
    return column_title

cancel  = ['cancel_14 days before check in','cancel_30 days before check in','cancel_48 hours after booking','cancel_5 days before check in','cancel_no free cancel']
output1 = remove_dup(cancel)

type_   = ['type_camper','type_entire home','type_room','type_tent']
output2 = remove_dup(type_)

res     = ['res_time_within a day','res_time_within a few hours','res_time_within an hour']
output3 = remove_dup(res)

tmp_interact = tmp_interact.drop(columns = output1+output2+output3) 
dta_reg      = pd.concat([dta_reg.iloc[:,0],tmp_interact.copy()],axis = 1) # 2863

# polynomial and cubic
tmp = dta_reg[['clean_fee', 'service_fee', 'occupancy_fee', 'guest','bedroom', 'beds', 'baths',
               'rating','review_number', 'rating_cleanliness', 'rating_accuracy','rating_communication', 
               'rating_location', 'rating_check_in','rating_value', 'amenity_number','host_review']].copy()    
column_name = list(tmp.columns)

# square
square_name = [s + '_suqare' for s in column_name]
tmp_square  = pd.DataFrame(np.square(tmp.to_numpy()))
tmp_square  = tmp_square.set_axis(square_name, axis=1, inplace=False)
dta_reg     = pd.concat([dta_reg,tmp_square],axis = 1) # 2880

# cubic
cubic_name  = [s + '_cubic' for s in column_name]
tmp_cubic   = pd.DataFrame(np.power(tmp.to_numpy(),3))
tmp_cubic   = tmp_cubic.set_axis(cubic_name, axis=1, inplace=False)
dta_reg     = pd.concat([dta_reg,tmp_cubic],axis = 1) # 2897



## filter out test data and standardize
dta_reg['ML_group'] = np.random.randint(10,size = dta_reg.shape[0])
dta_reg['ML_group'] = (dta_reg['ML_group']<=8)*0 + (dta_reg['ML_group']>8)*1


dta_reg_std              = dta_reg.iloc[:,:-1][dta_reg['ML_group']==0].copy().reset_index()
del dta_reg_std['index']
dta_reg_std.iloc[:,0]    = dta_reg_std.iloc[:,0] - dta_reg_std.iloc[:,0].mean()
scaler                   = StandardScaler().fit(dta_reg_std.iloc[:,1:])
dta_reg_std.iloc[:,1:]   = pd.DataFrame(scaler.transform(dta_reg_std.iloc[:,1:]))

## Train---Test split
X_t    = dta_reg_std.iloc[:,1:].to_numpy()
Y_t    = dta_reg_std.iloc[:,0].to_list()
X_test = dta_reg.iloc[:,1:-1][dta_reg['ML_group']==1].copy().reset_index()
del X_test['index']
scaler = StandardScaler().fit(X_test)
X_test = pd.DataFrame(scaler.transform(X_test)).to_numpy()
Y_test = (dta_reg.iloc[:,0][dta_reg['ML_group']==1] - dta_reg.iloc[:,0][dta_reg['ML_group']==1].mean()).to_list()

## Train---Validation split
dta_reg_std['ML_group']  = np.random.randint(10,size = dta_reg_std.shape[0])

# %%%% OLS
ols_model = LinearRegression(fit_intercept = False).fit(X_t,Y_t)

# predict
ols_pred = ols_model.predict(X_test)
ols_mspe = np.mean(np.square(Y_test-ols_pred))

# %%%% Ridge Regression 
# choose lambda, minimize the 10-fold cross validated MSPE
alpha_range = [100, 150, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
alpha_range = [*range(1000, 3020, 50)]

emspe = []
for a in alpha_range: 
    mspe_list = []
    for r in range(10):        
        x_train = dta_reg_std[dta_reg_std.ML_group != r].iloc[:,1:-1].to_numpy()
        y_train = dta_reg_std[dta_reg_std.ML_group != r].iloc[:,0].to_list()
        x_val   = dta_reg_std[dta_reg_std.ML_group == r].iloc[:,1:-1].to_numpy()
        y_val   = dta_reg_std[dta_reg_std.ML_group == r].iloc[:,0].to_list()
        clf     = Ridge(alpha = a,fit_intercept=False).fit(x_train, y_train)
        y_pred  = clf.predict(x_val) 
        mspe    = np.mean(np.square(y_val-y_pred))
        mspe_list.append(mspe)
    emspe.append(np.mean(mspe_list))
plt.plot(alpha_range,emspe)    
plt.xlabel('lambda_ridge')
plt.ylabel('Mean Squared Error of prediction')

ridge_lambda = alpha_range[emspe.index(min(emspe))]

# predict using ridge
ridge_model = Ridge(alpha = ridge_lambda,fit_intercept=False).fit(X_t, Y_t)
ridge_pred  = ridge_model.predict(X_test)
ridge_mspe  = np.mean(np.square(Y_test - ridge_pred))


# %%%% LASSO Regression
# choose lambda, minimize the 10-fold cross validated MSPE
alpha_range = np.arange(0.05,2.05,0.05)
alpha_range = np.arange(0.5,0.7,0.01)

emspe = []
for a in alpha_range: 
    mspe_list = []
    for r in range(10):        
        x_train = dta_reg_std[dta_reg_std.ML_group != r].iloc[:,1:-1].to_numpy()
        y_train = dta_reg_std[dta_reg_std.ML_group != r].iloc[:,0].to_list()
        x_val   = dta_reg_std[dta_reg_std.ML_group == r].iloc[:,1:-1].to_numpy()
        y_val   = dta_reg_std[dta_reg_std.ML_group == r].iloc[:,0].to_list()
        clf     = Lasso(tol=0.001, max_iter = 10000,alpha=a,fit_intercept = False).fit(x_train, y_train)
        y_pred  = clf.predict(x_val) 
        mspe    = np.mean(np.square(y_val-y_pred))
        mspe_list.append(mspe)
    emspe.append(np.mean(mspe_list))
plt.plot(alpha_range,emspe)    
plt.xlabel('lambda_lasso')
plt.ylabel('Mean Squared Error of prediction')

lasso_lambda = alpha_range[emspe.index(min(emspe))]

# predict using lasso
lasso_model  = Lasso(tol=0.001, max_iter = 10000,alpha = lasso_lambda,fit_intercept = False).fit(X_t, Y_t)
lasso_pred   = lasso_model.predict(X_test)
lasso_mspe   = np.mean(np.square(Y_test - lasso_pred))


# %%%% PCA
range_p = [*range(1,600,10)]
range_p = [*range(1,21,1)]

emspe = []
for p in range_p:
    mspe_list = []
    pca = PCA(n_components = p)
    for i in range(10):
        x_train   = pca.fit_transform(dta_reg_std[dta_reg_std.ML_group != i].iloc[:,1:-1].to_numpy())
        x_val     = pca.fit_transform(dta_reg_std[dta_reg_std.ML_group == i].iloc[:,1:-1].to_numpy())
        y_train   = dta_reg_std[dta_reg_std.ML_group != i].iloc[:,0].to_list()
        y_val     = dta_reg_std[dta_reg_std.ML_group == i].iloc[:,0].to_list()
        pca_ols   = LinearRegression(fit_intercept = False).fit(x_train,y_train)
        y_pred    = pca_ols.predict(x_val)
        mspe      = np.mean(np.square(y_val-y_pred))
        mspe_list.append(mspe)
    emspe.append(np.mean(mspe_list))
plt.plot(range_p,emspe)    
plt.xlabel('number of components')
plt.ylabel('Mean Squared Error of prediction')   

pca_p     =  emspe.index(min(emspe))+1


pca_model      = PCA(n_components = pca_p)
X_t_reduced    = pca_model.fit_transform(X_t)
sum_variance   = sum(pca_model.explained_variance_ratio_)

X_test_reduced =  pca_model.fit_transform(X_test)

pca_ols_model  = LinearRegression(fit_intercept = False).fit(X_t_reduced,Y_t)
pca_pred       = pca_ols_model.predict(X_test_reduced)
pca_mspe       = np.mean(np.square(Y_test - pca_pred))
        
        
