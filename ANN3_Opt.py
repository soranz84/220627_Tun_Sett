#!/usr/bin/env python
# coding: utf-8

# # ANN 2
# --------------------------------------------------------------------------------------------
# ## Import von Bibliotheken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ## Einlesen der Daten
df = pd.read_csv('Data_All.csv')

# ### Aufteilen in X und y Set
X = df[['lambda','rho','c', 'phi','E_50', 'E_oed', 'E_ur', 'C', 'D', 'C_D' ]]
y = df[['i']]
X_list=X.columns.values
X_list

# ### Sv max
Sv=y[['i']]
Sv_max=np.amax(Sv.values)
Sv_max

# ### i max
# i=y[['i']]
# i_max=np.amax(i.values)
# i_max

# ### Aufteilen in Trainings und Testset (Random number=17, 30% Testdaten)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# ### Normalisieren der Daten
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ### ANN mit tensorflow/keras
# #### import von tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
tf.random.set_seed(17)
from numpy.random import seed
seed(17)
np.random.seed(17)
tf.random.set_seed(17)

# #### Early Stop definieren
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50, restore_best_weights=True)


# #### Modelerstellung Funktion
def create_model (n_layers, n_nodes, activation_function, initializer, loss_function):
    model = Sequential()
    for i in range (n_layers):
            model.add(Dense(n_nodes, activation=activation_function, kernel_initializer=initializer, kernel_regularizer='l2'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=loss_function, metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

# #### Import und wrapping für scikit.learn
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,ShuffleSplit

model =  KerasRegressor(build_fn=create_model, validation_data=(X_test,y_test.values),
                        epochs=30000, callbacks=[early_stop], verbose=0)

# #### Grid definieren
activation_functions = ['gelu','relu'] 
initializers = [tf.keras.initializers.GlorotUniform(seed=17),tf.keras.initializers.GlorotNormal(seed=17), tf.keras.initializers.HeUniform(seed=17),tf.keras.initializers.HeNormal(seed=17)]
# loss_functions = ['mse']
n_layers = [1,2,3,4]
n_nodes = [10,20,30,40,50]

param_grid = dict(n_layers=n_layers, n_nodes=n_nodes, activation_function=activation_functions, 
                  initializer=initializers,
                  loss_function=['mse'])

 #### Gridsearch 
ss = ShuffleSplit(n_splits=4, test_size=0.25, train_size=0.75, random_state=17)

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv=ss,verbose=4)
grid.fit(X_train,y_train.values)
print(grid.best_score_)
print(grid.best_params_)
grid.cv_results_

pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']].to_csv('Grid_opt_ANN3.csv')