# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:53:40 2022

@author: Xhensilda Allka
"""

import pandas as pd
import numpy as np
from numpy import *
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
from itertools import chain
from numpy import unravel_index
import os

dir_path = 'C:/Users/Usuario/Dropbox/PC/Desktop/UPC/Results-TPB/Ozone(2018)/FirstPhase/'
os.chdir(dir_path)
path = os.getcwd()


#%%
# estimated low-cost sensor data (reshape in 2-dimensional data, rows = 24 hours and columns = number of days)
sensor_data = pd.read_csv(path + "/Calibration/SVR-N212.csv", sep=',')
sensor_data.date = pd.to_datetime(sensor_data.date, format='%Y-%m-%d %H:%M:%S')
sensor_data.set_index('date', inplace = True)
data_sensor = pd.pivot_table(sensor_data, index=sensor_data.index.date, columns=sensor_data.index.hour, values='calibrated SVR O3').transpose()

# for col in range (0, data_sensor.shape[1]):
#     if np.isnan(data_sensor.iloc[1,col]):
#         data_sensor.loc[1,:] = (data_sensor.iloc[0,:]+data_sensor.iloc[2,:])/2

# data_sensor = data_sensor.sort_index()

# reference station data
data_ref = pd.read_csv(path + "/Database/N212_supervised.csv", sep=';')
data_ref.date = pd.to_datetime(data_ref.date, format='%Y-%m-%d %H:%M:%S')
data_ref.set_index('date', inplace = True)
ref_data = pd.pivot_table(data_ref, index = data_ref.index.date, columns = data_ref.index.hour, values='ref_o3').transpose()


#%%
# Remove the columns (days) which have nan data
data_ref = ref_data.dropna(axis='columns') 
data_capt = data_sensor.dropna(axis='columns') 

# Consider only the columns (days) that are in common in both reference stations and calibrated low-cost sensor data
index = data_ref.columns.intersection(data_capt.columns)
data_ref_com = data_ref[index]
data_capt_com = data_capt[index]

#%% Cross-validation procedure to obtain the hyperparameters for TPB-D and TPB-c

# Randomly split 80% train and 20% test
test_size  = 0.2
data_ref_com_shuf = data_ref_com.transpose().sample(frac = 1, random_state = 1)
data_capt_com_shuf = data_capt_com.transpose().sample(frac = 1, random_state = 1)

data_ref_com_shuf = data_ref_com_shuf.transpose()
data_capt_com_shuf = data_capt_com_shuf.transpose()


days_train = int(np.round(len(data_capt_com_shuf.columns)-len(data_capt_com_shuf.columns)*test_size))
testing_ref_all = data_ref_com_shuf.iloc[:,days_train: ]
testing_capt_all = data_capt_com_shuf.iloc[:,days_train:]

training_ref_all = data_ref_com_shuf.transpose().drop(testing_ref_all.transpose().index)
training_capt_all = data_capt_com_shuf.transpose().drop(testing_capt_all.transpose().index)

# Define the lists for the error during the cross-validation procedure for TPB-D and TPB-C
error_tot_TPBD = []
error_tot_TPBC = []

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(training_ref_all):
     training_ref, testing_ref = training_ref_all.iloc[train_index], training_ref_all.iloc[test_index]
     training_capt, testing_capt = training_capt_all.iloc[train_index], training_capt_all.iloc[test_index]

     training_ref = training_ref.transpose()
     testing_ref = testing_ref.transpose()
     training_capt = training_capt.transpose()
     testing_capt = testing_capt.transpose()

    # remove the average value for each column
     avg_o3 = np.mean(training_ref, axis = 1)

     training_ref_0 = (training_ref.transpose() - avg_o3).transpose()
     training_capt_0 = (training_capt.transpose() - avg_o3).transpose()

     testing_capt_0 = (testing_capt.transpose() - avg_o3).transpose()

     # svd in the matrix where the mean is zero
     U, sigma, VT = linalg.svd(training_ref_0)
     
     repetitions = len(testing_ref.columns)
     avg = np.transpose([avg_o3] * repetitions)
     
     # the dimension of subspace, kappa
     k_list = range(1, 25)
     random.seed(1)
     
     # regularization parameter
     lambd = random.sample(range(0, 20000), 200)	
     lambd.sort()
     error_landa = []
     
     TPBD_captValid_RMSE = []
     for k in k_list:  
         
         TPBC_captValid_RMSE = []
         for j in lambd:
             phi = U[:, :k].transpose() @training_ref_0 @ training_capt_0.transpose() @ U[:, :k]\
            @ np.linalg.inv(U[:, :k].transpose() @training_capt_0 @training_capt_0.transpose() @U[:, :k] + j*np.identity(k)) 
             TPBC_capt_valid_approx = avg + U[:,:k] @ phi @U[:, :k].transpose() @testing_capt_0
             
             approx_TPBC = pd.Series(TPBC_capt_valid_approx.values.ravel('F'))
             true_value = pd.Series(testing_ref.values.ravel('F'))
             TPBC_captValid_RMSE.append(np.sqrt(mean_squared_error(true_value, approx_TPBC)))
             
         # this is a 2 dimensional list that has k (len(k_list)) rows and j columns (length of lambda vector)    
         TPBD_capt_valid_approx = avg + U[:,:k] @U[:, :k].transpose() @testing_capt_0
         TPBD_captValid_RMSE.append(np.sqrt(mean_squared_error(testing_ref.values.ravel('F'), TPBD_capt_valid_approx.values.ravel('F'))))
         error_landa.append((TPBC_captValid_RMSE))
         flatten_list = list(chain.from_iterable(error_landa))
     error_tot_TPBD.append((flatten_list))
     error_tot_TPBC.append(TPBD_captValid_RMSE)
error_TPBC = pd.DataFrame(error_tot_TPBD)
error_TPBD = pd.DataFrame(error_tot_TPBC)

# Find the average error 
avg_error_TPBC = error_TPBC.mean(axis = 0)
avg_error_TPBD = error_TPBD.mean(axis = 0)

avg_rmse_TPBC = avg_error_TPBC.to_numpy().reshape(len(k_list), len(lambd))
avg_rmse_TPBD = avg_error_TPBD.to_numpy().reshape(len(k_list))

cross_error_TPBD = pd.DataFrame(avg_rmse_TPBD)
cross_error_TPBC = pd.DataFrame(avg_rmse_TPBC)

min_index_TPBC = unravel_index(avg_rmse_TPBC.argmin(), avg_rmse_TPBC.shape)
min_index_TPBD = unravel_index(avg_rmse_TPBD.argmin(), avg_rmse_TPBD.shape)

# Hyperparameters' value (where the error in the validation data set during CV is minimal)
min_lambda = lambd[min_index_TPBC[1]]
min_k_TPBC = k_list[min_index_TPBC[0]]
min_k_TPBD = k_list[min_index_TPBD[0]]

print(min_k_TPBD)
print(np.min(avg_rmse_TPBC), min_lambda, min_k_TPBC)

#%% Test the model in the testing dataset with the best hyperparameters obtained from CV
training_ref_all = training_ref_all.transpose()
training_capt_all = training_capt_all.transpose()

# Find the average day and remove it from each column (day)
avg_day = np.mean(training_ref_all, axis = 1)

training_ref_all_0 = (training_ref_all.transpose() - avg_day).transpose()
training_capt_all_0 = (training_capt_all.transpose() - avg_day).transpose()

testing_ref_all_0 = (testing_ref_all.transpose() - avg_day).transpose()
testing_capt_all_0 = (testing_capt_all.transpose() - avg_day).transpose()

# svd in the reference station data matrix where the mean is zero
U_all, sigma_all, VT_all = linalg.svd(training_ref_all_0)
     
repetitions_all = len(testing_ref_all_0.columns)

avg_all = np.transpose([avg_day] * repetitions_all)


k_TPBC = min_k_TPBC
k_TPBD = min_k_TPBD

phi = U_all[:, :k_TPBC].transpose() @training_ref_all_0 @ training_capt_all_0.transpose() @ U_all[:, :k_TPBC]\
        @ np.linalg.pinv(U_all[:, :k_TPBC].transpose() @training_capt_all_0 @training_capt_all_0.transpose() @U_all[:, :k_TPBC] + min_lambda*np.identity(k_TPBC)) 
TPBC_capt_test_approx = avg_all + U_all[:,:k_TPBC] @ phi @U_all[:, :k_TPBC].transpose() @testing_capt_all_0
TPBD_capt_test_approx = avg_all + U_all[:,:k_TPBD] @U_all[:, :k_TPBD].transpose() @testing_capt_all_0
