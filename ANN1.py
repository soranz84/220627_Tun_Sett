#!/usr/bin/env python
# coding: utf-8

# # ANN 2
# --------------------------------------------------------------------------------------------
# ## Import von Bibliotheken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set default plot style first
plt.rcParams.update(plt.rcParamsDefault)
# Change plot style
plt.rcParams["font.family"] = "helvetica"
plt.rcParams["font.serif"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.bf"] = "helvetica"
plt.rcParams["mathtext.it"] = "helvetica"
plt.rcParams["mathtext.rm"] = "helvetica"
plt.rcParams["mathtext.sf"] = "helvetica"
plt.rcParams["mathtext.tt"] = "helvetica"

# ## Einlesen der Daten
df = pd.read_csv('Data_All.csv')

# ### Aufteilen in X und y Set
X = df[['lambda','rho','c', 'phi','E_50', 'E_oed', 'E_ur', 'C', 'D', 'C_D' ]]
y = df[['s_max','i']]
X_list=X.columns.values
X_list


# ### Sv max
Sv=y[['s_max']]
Sv_max=np.amax(Sv.values)
Sv_max

# ### i max
i=y[['i']]
i_max=np.amax(i.values)
i_max

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



# #### Early Stop definieren
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50, restore_best_weights=True)


# #### Modelerstellung Funktion
def create_model (n_layers, n_nodes, activation_function, initializer, loss_function):
    model = Sequential()
    for i in range (n_layers):
            model.add(Dense(n_nodes, activation=activation_function, kernel_initializer=initializer, kernel_regularizer='l2'))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss=loss_function, metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

# #### Import und wrapping für scikit.learn
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,ShuffleSplit

model =  KerasRegressor(build_fn=create_model, validation_data=(X_test,y_test.values),
                        epochs=30000, callbacks=[early_stop], verbose=0)

# #### Grid definieren
activation_functions = ['sigmoid','gelu','relu'] 
initializers = [tf.keras.initializers.GlorotUniform(seed=17),tf.keras.initializers.GlorotNormal(seed=17), tf.keras.initializers.HeUniform(seed=17),tf.keras.initializers.HeNormal(seed=17)]
loss_functions = ['mse','mape','mae']
n_layers = [1,2,3,4,5]
n_nodes = [5,7,10,14,18,21,25,37,50,75,100,500,1000]

param_grid = dict(n_layers=n_layers, n_nodes=n_nodes, activation_function=activation_functions, 
                  initializer=initializers,
                  loss_function=['mse'])


# #### Gridsearch 
ss = ShuffleSplit(n_splits=4, test_size=0.25, train_size=0.75, random_state=17)

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv=ss,verbose=0)

#grid.fit(X_train,y_train)

#print(grid.best_score_)
#print(grid.best_params_)
#grid.cv_results_
#pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']].to_csv('Grid_opt_ANN1.csv')


# #### Model trainieren
visible = Input(shape=(10,))
hidden1 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(visible)
hidden2 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden1)
hidden3 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden2)
hidden4 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden3)
output = Dense(2, activation='relu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden4)
model = Model(inputs=visible, outputs=output)

model.compile(optimizer='adam',loss='mse',metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),epochs=30000,callbacks=[early_stop])

# Model save
model.save('ANN1.h5')  

df_loss=pd.DataFrame(model.history.history)
df_loss=df_loss.rename(columns={"loss": "training loss", "val_loss": "test loss"})
df_loss

sns.set_style('ticks')
plt.figure(dpi=300)

plt.plot(df_loss.index.values,df_loss[['training loss']], color='k', label='training loss')
plt.plot(df_loss.index.values,df_loss[['test loss']], color='firebrick', label='test loss')
plt.xlim(0,)
plt.ylim(0,)
plt.ylabel('MSE')
plt.xlabel('Training epochs')

#plt.legend().get_frame().set_edgecolor('k')
plt.legend(["Training","Test"],fancybox=False,edgecolor='0',loc='upper right',facecolor='white', framealpha=1)

#plt.title('ANN 2: Trainings-und Testloss nach Epochen ')

#plt.savefig('plot_004_ANN_epochen_loss.png', dpi=1000)
#plt.savefig('plot_004_ANN_epochen_loss.svg')
#plt.savefig('plot_004_ANN_epochen_loss.eps')

# #### Prognose
predictions = model.predict(X_test)

y_test_new_index = y_test.reset_index()

predictions_df = pd.DataFrame(predictions, columns = ['s_max_pred','i_pred'])

test = y_test_new_index.join(predictions_df)
test

# ## Plots + Evaluierung Sv_max
y_test_s = test[['s_max']]
y_pred_s = test[['s_max_pred']]

from sklearn.metrics import r2_score
from sklearn import metrics

print('R2:',r2_score(y_test_s.values, y_pred_s.values))
print('MAE:', metrics.mean_absolute_error(y_test_s, y_pred_s))
print('MAPE:',metrics.mean_absolute_percentage_error(y_test_s, y_pred_s))
print('MSE:', metrics.mean_squared_error(y_test_s, y_pred_s))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_s, y_pred_s)))

sns.set_style('ticks')
plt.figure(dpi=300)
Scatter_S=plt.scatter(y_test_s,y_pred_s, color='black',s=12)
plt.plot( [0,Sv_max],[0,Sv_max],color='firebrick', label='perfect fit' ,lw=1 )
plt.plot( [0,Sv_max],[0,Sv_max*1.2],color='grey', label='+20%', linestyle='--',lw=1 )
plt.plot( [0,Sv_max],[0,Sv_max/1.2],color='grey', label='-20%', linestyle='--',lw=1 )

plt.xlim(0, Sv_max)
plt.ylim(0, Sv_max)

plt.ylabel('$s_\mathrm{max}$ predicted (mm)')
plt.xlabel('$s_\mathrm{max}$ calculated (mm)')
plt.plot([], [], ' ', label='R²: '+str(round (r2_score(y_test_s.values, y_pred_s.values),3)))

plt.legend(fontsize='medium')
plt.legend().get_frame().set_edgecolor('k')
plt.legend(fancybox=False,edgecolor='0',loc='upper left',facecolor='white', framealpha=1)
#plt.title('ANN 2: maximale Setzung Test')

plt.gca().set_aspect('equal')

#plt.savefig('plot_005_ANN_S_test.png', dpi=1000)
#plt.savefig('plot_005_ANN_S_test.svg')
#plt.savefig('plot_005_ANN_S_test.eps')
plt.savefig('ANN1_S_test.pdf', bbox_inches='tight')
plt.show()

# ### Train Errors Sv
predictions_train = model.predict(X_train)
y_train_new_index = y_train.reset_index()
predictions_train = pd.DataFrame(predictions_train, columns = ['s_max_pred','i_pred'])
train = y_train_new_index.join(predictions_train)

#train
y_train_s = train[['s_max']]
y_train_pred_s = train[['s_max_pred']]

print('R2:',r2_score(y_train_s.values, y_train_pred_s.values))
print('MAE:', metrics.mean_absolute_error(y_train_s, y_train_pred_s))
print('MAPE:',metrics.mean_absolute_percentage_error(y_train_s, y_train_pred_s))
print('MSE:', metrics.mean_squared_error(y_train_s, y_train_pred_s))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train_s, y_train_pred_s)))

sns.set_style('ticks')
plt.figure(dpi=300)
Scatter_S=plt.scatter(y_train_s,y_train_pred_s, color='black', facecolors='none', s=12)
plt.plot( [0,Sv_max],[0,Sv_max],color='firebrick', label='perfect fit' ,lw=1 )
plt.plot( [0,Sv_max],[0,Sv_max*1.2],color='grey', label='+20%', linestyle='--',lw=1 )
plt.plot( [0,Sv_max],[0,Sv_max/1.2],color='grey', label='-20%', linestyle='--',lw=1 )

plt.xlim(0, Sv_max)
plt.ylim(0, Sv_max)
plt.ylabel('$s_\mathrm{max}$ predicted (mm)')
plt.xlabel('$s_\mathrm{max}$ calculated (mm)')
plt.plot([], [], ' ', label='R²: '+str(round (r2_score(y_train_s.values, y_train_pred_s.values),3)))
plt.legend(fontsize='medium')
plt.legend().get_frame().set_edgecolor('k')
plt.legend(fancybox=False,edgecolor='0',loc='upper left',facecolor='white', framealpha=1)
# plt.title('ANN 2: maximale Setzung Training')

plt.gca().set_aspect('equal')

#plt.savefig('plot_006_ANN_S_train.png', dpi=1000)
#plt.savefig('plot_006_ANN_S_train.svg')
#plt.savefig('plot_006_ANN_S_train.eps')
plt.savefig('ANN1_S_train.pdf', bbox_inches='tight')
plt.show()

# ## Plots + Evaluierung i
y_test_i= test[['i']]
y_pred_i= test[['i_pred']]

print('R2:',r2_score(y_test_i.values, y_pred_i.values))
print('MAE:', metrics.mean_absolute_error(y_test_i, y_pred_i))
print('MAPE:',metrics.mean_absolute_percentage_error(y_test_i, y_pred_i))
print('MSE:', metrics.mean_squared_error(y_test_i, y_pred_i))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_i, y_pred_i)))

sns.set_style('ticks')
plt.figure(dpi=300)
Scatter_S=plt.scatter(y_test_i,y_pred_i, color='black',marker='v',s=12)
plt.plot( [0,i_max],[0,i_max],color='firebrick', label='perfect fit',lw=1  )
plt.plot( [0,i_max],[0,i_max*1.2],color='grey', label='+20%', linestyle='--',lw=1 )
plt.plot( [0,i_max],[0,i_max/1.2],color='grey', label='-20%', linestyle='--',lw=1 )

plt.xlim(0, i_max)
plt.ylim(0, i_max)

plt.ylabel('$i$ predicted (m)')
plt.xlabel('$i$ calculated (m)')
plt.plot([], [], ' ', label='R²: '+str(round (r2_score(y_test_i.values, y_pred_i.values),3)))
plt.legend(fontsize='medium')
plt.legend().get_frame().set_edgecolor('k')
plt.legend(fancybox=False,edgecolor='0',loc='upper left',facecolor='white', framealpha=1)
# plt.title('ANN 2: Wendepunktabstand Test')

plt.gca().set_aspect('equal')

#plt.savefig('plot_007_ANN_i_test.png', dpi=1000)
#plt.savefig('plot_007_ANN_i_test.svg')
#plt.savefig('plot_007_ANN_i_test.eps')
plt.savefig('ANN1_i_test.pdf', bbox_inches='tight')
plt.show()


# ### Train errors i
y_train_i = train[['i']]
y_train_pred_i = train[['i_pred']]

print('R2:',r2_score(y_train_i.values, y_train_pred_i.values))
print('MAE:', metrics.mean_absolute_error(y_train_i, y_train_pred_i))
print('MAPE:',metrics.mean_absolute_percentage_error(y_train_i, y_train_pred_i))
print('MSE:', metrics.mean_squared_error(y_train_i, y_train_pred_i))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train_i, y_train_pred_i)))

sns.set_style('ticks')
plt.figure(dpi=300)
Scatter_S=plt.scatter(y_train_i,y_train_pred_i, color='black', facecolor='none',marker='v', s=12)
plt.plot( [0,i_max],[0,i_max],color='firebrick', label='perfect fit' ,lw=1 )
plt.plot( [0,i_max],[0,i_max*1.2],color='grey', label='+20%', linestyle='--',lw=1 )
plt.plot( [0,i_max],[0,i_max/1.2],color='grey', label='-20%', linestyle='--',lw=1 )

plt.xlim(0, i_max)
plt.ylim(0, i_max)

plt.ylabel('$i$ predicted (m)')
plt.xlabel('$i$ calculated (m)')
plt.plot([], [], ' ', label='R²: '+str(round (r2_score(y_train_i.values, y_train_pred_i.values),3)))
plt.legend(fontsize='medium')
plt.legend().get_frame().set_edgecolor('k')
plt.legend(fancybox=False,edgecolor='0',loc='upper left',facecolor='white', framealpha=1)
# plt.title('ANN 2: Wendepunktabstand Training')

plt.gca().set_aspect('equal')

#plt.savefig('plot_008_ANN_i_train.png', dpi=1000)
#plt.savefig('plot_008_ANN_i_train.svg')
#plt.savefig('plot_008_ANN_i_train.eps')
plt.savefig('ANN1_i_train.pdf', bbox_inches='tight')
plt.show()

# ## Feature Importance Shap
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
#from tensorflow.python.keras.initializers import GlorotUniformV2 as GlorotUniform
tf.random.set_seed(17)
from numpy.random import seed
seed(17)
np.random.seed(17)
tf.random.set_seed(17)

visible = Input(shape=(10,))
hidden1 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(visible)
hidden2 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden1)
hidden3 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden2)
hidden4 = Dense(40, activation='gelu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden3)
output = Dense(2, activation='relu',kernel_initializer=tf.keras.initializers.HeNormal(seed=17),kernel_regularizer='l2')(hidden4)
model = Model(inputs=visible, outputs=output)

model.compile(optimizer='adam',loss='mse',metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),epochs=30000,callbacks=[early_stop])


#model_S=Model(inputs=model.inputs, outputs=model.outputs[0])
import shap
#tf.compat.v1.disable_v2_behavior()

background = X_train

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test,check_additivity=False)
shap.summary_plot(shap_values[0], X_test, feature_names=X_list)

shap.initjs()
plt.figure(dpi=300)
shap.summary_plot(
    shap_values[0], 
    X_test,
    feature_names=X_list,
    max_display=50,
    plot_type='bar', show=False, color='tab:blue')
fig = plt.gcf()
ax = plt.gca()
plt.xlabel('Mean SHAP value')
# plt.title('ANN 2: durchschnittliche SHAP-Werte für Sv_max ')
plt.tight_layout()

#plt.savefig('plot_012_ANN_shap_s.png', dpi=1000)
#plt.savefig('plot_012_ANN_shap_s.svg')
#plt.savefig('plot_012_ANN_shap_s.eps')
plt.savefig('ANN1_S_SHAP.pdf', bbox_inches='tight')
plt.show()

shap.initjs()
plt.figure(dpi=300)
shap.summary_plot(
    shap_values[1], 
    X_test,
    feature_names=X_list,
    max_display=50,
    plot_type='bar', show=False, color='tab:blue')
fig = plt.gcf()
ax = plt.gca()
plt.xlabel('Mean SHAP value')
# plt.title('ANN 2: durchschnittliche SHAP-Werte für i ')
plt.tight_layout()

#plt.savefig('plot_013_ANN_shap_i.png', dpi=1000)
#plt.savefig('plot_013_ANN_shap_i.svg')
#plt.savefig('plot_013_ANN_shap_i.eps')
plt.savefig('ANN1_i_SHAP.pdf', bbox_inches='tight')
plt.show()




