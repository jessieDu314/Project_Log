# Fill Missing Values
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


# %%%%
def fillMissingPCA(tmp,m):
    tmp_hat      = tmp.fillna(0).copy()
    objective    = []
    tmp_hat_list = []
    for i in range(11):    
        pca = PCA(n_components=m)
        pca.fit(tmp_hat.iloc[:,0:-1])
        l = pca.components_
        z = pca.fit_transform(tmp_hat.iloc[:,0:-1])    
        pred = pd.DataFrame(np.matmul(z,l))
        obj = np.sum(np.square(tmp.iloc[:,0:-1][tmp_hat.index_na == 0].values - 
                               pred.iloc[tmp[tmp_hat.index_na    == 0].index].values))
        objective.append(obj)
        tmp_hat_list.append(tmp_hat.copy())
    
        for j in range(tmp_hat.shape[0]):
            for k in range(tmp_hat.shape[1]-1):
                if str(tmp.iloc[j][k]) == 'nan':                
                    tmp_hat.iloc[j,k] = pred.iloc[j][k].copy()
    return (objective,tmp_hat_list)

# %%%%
# choose k
def KNN_choosek(x_train,y_train,x_val,y_val):
    mse = []
    for k in range(1,21):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(x_train, y_train)    
        pred = knn.predict(x_val) 
        error = np.mean(np.square(y_val-pred))
        mse.append(error)
    return mse

def knn_regression(tmp,fill_num):
    x_to_predict  = tmp.iloc[:,3:-1][tmp.index_na == 1]
    x_all         = tmp.iloc[:,3:-1][tmp.index_na == 0]
    tmp_comp      = tmp[tmp.index_na == 0].copy()
    mse_list        = []
    k_neighbor_list = []
    y_pred_list     = []
    for n in range(fill_num):
        mse_min = np.inf
        for g in range(10):
            tmp_comp['ML_group']      = np.random.randint(10,size = tmp_comp.shape[0])
            tmp_comp['ML_group']      = (tmp_comp['ML_group']<=7)*0 + (tmp_comp['ML_group']>7)*1
            x_train = tmp_comp[tmp_comp.ML_group == 0].iloc[:,3:-1].to_numpy()
            x_val   = tmp_comp[tmp_comp.ML_group == 1].iloc[:,3:-1].to_numpy()
            y_train = tmp_comp[tmp_comp.ML_group == 0].iloc[:,0].to_list()
            y_val   = tmp_comp[tmp_comp.ML_group == 1].iloc[:,0].to_list()
            mse     = KNN_choosek(x_train,y_train,x_val,y_val)
            if min(mse) < mse_min:
                mse_min     = min(mse)
                mse_optimal = mse
        k_neighbor = mse_optimal.index(min(mse_optimal))+1
        k_neighbor_list.append(k_neighbor)
        mse_list.append(mse_optimal)
        y_all = tmp.iloc[:,n][tmp.index_na == 0]
        model = KNeighborsRegressor(n_neighbors=k_neighbor)
        model.fit(x_all,y_all)
        y_pred = round(pd.DataFrame(model.predict(x_to_predict)),2).copy()
        y_pred_list.append(y_pred)
    return (mse_list,k_neighbor_list,y_pred_list)

        

            
            
            
            
            
            
            
            
            
            
            
            


