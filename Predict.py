# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:38:11 2022

@author: Enrico Soranzo
"""
# --------------------------------------------- README --------------------------------------------- #
# This script needs the files 'Data_All.csv', 'ANN1.h5', 'ANN2.h5' and 'ANN3.h5'.
# Make sure these files and the present script are in the same folder.
# 'Data_All.csv' is needed in order to scale the user input in the same way as the training data.
# 'ANN*.h5' are the neural networks trained by 'ANN*.py' and to be loaded by this script.
# For questions, please email the author at enrico.soranzo@boku.ac.at
# -------------------------------------------------------------------------------------------------- #
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import tensorflow as tf

# Set seed
tf.random.set_seed(17)
from numpy.random import seed
seed(17)
np.random.seed(17)

# Load neural networks
ANN1 = tf.keras.models.load_model('ANN1.h5')
ANN2 = tf.keras.models.load_model('ANN2.h5')
ANN3 = tf.keras.models.load_model('ANN3.h5')

# Input data
lam = float(input("Stress reduction factor $\lambda$ (-): "))
rho = float(input("Density (kg/m³): "))
coh = float(input("Cohesion (kPa): "))
phi = float(input("Friction angle (°): "))
E_50 = float(input("Secant modulus of elasticity (MPa): "))
E_oed = float(input("Oedometric modulus (MPa): "))
E_ur = float(input("Unload-reload modulus (MPa): "))
C = float(input("Overburden (m): "))
D = float(input("Tunnel diameter (m): "))
C_D = C/D

# Insert data in dataframe
X = pd.DataFrame()
X['lambda'] = [lam]
X['rho'] = [rho]
X['coh'] = [coh]
X['phi'] = [phi]
X['E_50'] = [E_50]
X['E_oed'] = [E_oed]
X['E_ur'] = [E_ur]
X['C'] = [C]
X['D'] = [D]
X['C_D'] = [C_D]

# Scale the data in the same way as the network training data
df = pd.read_csv('Data_All.csv')
X_scale = df[['lambda','rho','c', 'phi','E_50', 'E_oed', 'E_ur', 'C', 'D', 'C_D' ]]
y_scale = df[['s_max','i']]
X_scale_train, X_scale_test, y_scale_train, y_scale_test = train_test_split(X_scale, y_scale, test_size=0.3, random_state=17)
scaler = StandardScaler()
scaler.fit(X_scale_train) # Fit on dataset training data
X = scaler.transform(X) # Transform prediction data

# Print the results
print('----------------------------')
print('ANN 1')
print('----------------------------')
res_1 = ANN1.predict(X)
s_max_1 = res_1[0][0]
i_1 = res_1[0][1]
print('Max settlement: ' + str(round(s_max_1,1)) + ' mm')
print('Trough width: ' + str(round(i_1,1)) + ' m')
print('----------------------------')

print('ANN 2')
print('----------------------------')
res_2 = ANN2.predict(X)
s_max_2 = res_2[0][0]
print('Max settlement: ' + str(round(s_max_2,1)) + ' mm')
print('----------------------------')

print('ANN 3')
print('----------------------------')
res_3 = ANN3.predict(X)
i_3 = res_3[0][0]
print('Trough width: ' + str(round(i_3,1)) + ' m')
print('----------------------------')

