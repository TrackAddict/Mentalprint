
# coding: utf-8

# In[1]:

#%matplotlib inline

#%install_ext https://raw.github.com/cpcloud/ipython-autotime/master/autotime.py
#%load_ext autotime


# In[2]:

#import pyprind

#from IPython.display import HTML


import numpy as np
from numpy import genfromtxt

import pandas as pd
from pandas import DataFrame
from pandas import Panel

import warnings

# import sklearn as skl
# #from sklearn.preprocessing import normalize
# from sklearn.cross_validation import KFold
# from sklearn.ensemble import RandomForestClassifier as RFC
# from sklearn.tree import DecisionTreeClassifier as DTC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder



import theano

import keras
from keras.models import *
from keras.layers import *
from keras.layers.recurrent import *
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD


# In[3]:

warnings.filterwarnings('ignore')


# In[4]:

Michael = pd.read_csv('Trials/Michael.csv', header=None)

Michael.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                         'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation', 'Label']

Brooke = pd.read_csv('Trials/Brooke.csv', header=None)

Brooke.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                  'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation']

Mark = pd.read_csv('Trials/Mark.csv', header=None)

Mark.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                         'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation', 'Label']

Mark.shape


# In[5]:

Mark.head()


# In[6]:

Test_x = pd.read_csv('Trials/results.csv', header=None)

Test_x.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                         'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation']

Test_x['Label'] = 1


# In[7]:

temp = Test_x.values

test_X_1 = temp[:, 0:9]

test_Y_1 = temp[:, 10]

#test_Y_1


# In[8]:

Test_x_2 = pd.read_csv('Trials/results2.csv', header=None)

Test_x_2.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                         'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation']

Test_x_2['Label'] = 2

temp = Test_x_2.values

test_X_2 = temp[:, 0:9]

test_Y_2 = temp[:, 10]

test_Y_2


# In[9]:

Test_x_3 = pd.read_csv('Trials/results3.csv', header=None)

Test_x_3.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                    'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation']

Test_x_3['Label'] = 3

temp = Test_x_3.values

test_X_3 = temp[:, 0:9]

test_Y_3 = temp[:, 10]

test_Y_3


# In[10]:

temp_Y = pd.Series(test_Y_1)

temp_Y = pd.get_dummies(temp_Y)

test_Y = temp_Y.values

temp2 = np.zeros((len(test_Y), 2))

test_Y_1 = np.concatenate((test_Y, temp2), axis=1)


# In[11]:

Brooke['Label'] = 3

Brooke.shape


# In[12]:

Michael = Michael[Michael['Label'] == 1]

Michael_prime = Michael[Michael['Attention'] > 0]

Michael = Michael_prime[Michael_prime['Meditation'] > 0]


Mark_prime = Mark[Mark['Attention'] > 0]

Mark = Mark_prime[Mark_prime['Meditation'] > 0]

Mark.shape


# In[13]:

Mark.head()


# In[14]:

frames = [Michael.dropna()[0:1800], Mark.dropna()[0:1800], Brooke.dropna()[0:1800]]

data = pd.concat(frames, copy=False, ignore_index=True)

data = data.drop_duplicates()

data = data.dropna()

#data = data.values


# In[15]:

data.shape


# In[16]:

data = data.dropna()

data.shape


# In[17]:

data = data.values


# In[18]:

X = data[:, 0:9]

Y = data[:, 10]

X.shape


# In[19]:

lstm_data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

lstm_data.shape


# In[20]:

lstm_X = lstm_data[:, :, 0:9]

lstm_test_X_1 = np.reshape(test_X_1, (121, 1, 9))

lstm_test_X_2 = np.reshape(test_X_2, (121, 1, 9))

lstm_test_X_3 = np.reshape(test_X_3, (121, 1, 9))


# In[21]:

Y = pd.Series(Y)

Y = pd.get_dummies(Y)

Y = Y.values


# In[22]:

test_X_1.shape


# In[23]:

mlp_model = Sequential()

mlp_model.add(Dense(81, input_dim=9, init='uniform'))
mlp_model.add(Activation('softmax'))
mlp_model.add(Dropout(0.5))

mlp_model.add(Dense(512, input_dim=9, init='uniform'))
mlp_model.add(Activation('softmax'))
mlp_model.add(Dropout(0.5))

mlp_model.add(Dense(512, input_dim=9, init='uniform'))
mlp_model.add(Activation('softmax'))
mlp_model.add(Dropout(0.5))


mlp_model.add(Dense(512, input_dim=9, init='uniform'))
mlp_model.add(Activation('softmax'))
mlp_model.add(Dropout(0.5))


mlp_model.add(Dense(3, init='uniform'))
mlp_model.add(Activation('softmax'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

mlp_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# mlp_model.summary()


# In[24]:

# results = []


# for i in np.arange(0, 5):
#     mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)
    
#     predicted = mlp_model.predict(test_X_1, batch_size=20)
    
#     temp = np.mean(predicted, axis=0)
    
#     results.append(temp)

# results = np.asarray(results)

# results


# In[25]:

mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)

predicted = mlp_model.predict(test_X_1, batch_size=20)

temp = np.mean(predicted, axis=0)

temp


# In[26]:

# results = []


# for i in np.arange(0, 5):
#     mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)
    
#     predicted = mlp_model.predict(test_X_2, batch_size=20)
    
#     temp = np.mean(predicted, axis=0)
    
#     results.append(temp)

# results = np.asarray(results)

# results


# In[27]:

mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)
    
predicted = mlp_model.predict(test_X_2, batch_size=20)

temp = np.mean(predicted, axis=0)

temp


# In[28]:

# results = []


# for i in np.arange(0, 5):
#     mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)
    
#     predicted = mlp_model.predict(test_X_3, batch_size=20)
    
#     temp = np.mean(predicted, axis=0)
    
#     results.append(temp)

# results = np.asarray(results)

# results


# In[29]:

mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)

predicted = mlp_model.predict(test_X_3, batch_size=20)

temp = np.mean(predicted, axis=0)

temp


# In[ ]:




# In[30]:

mlp_model = Sequential()

mlp_model.add(Dense(81, input_dim=9, init='uniform'))
mlp_model.add(Activation('relu'))
mlp_model.add(Dropout(0.5))


mlp_model.add(Dense(512, init='uniform'))
mlp_model.add(Activation('relu'))
mlp_model.add(Dropout(0.5))


mlp_model.add(Dense(512, init='uniform'))
mlp_model.add(Activation('relu'))


mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(512, init='uniform'))
mlp_model.add(Activation('relu'))
mlp_model.add(Dropout(0.5))


mlp_model.add(Dense(3, init='uniform'))
mlp_model.add(Activation('softmax'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

mlp_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

mlp_model.summary()


# In[31]:

results = []


for i in np.arange(0, 3):
    
    mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)
    
    predicted = mlp_model.predict(test_X_1, batch_size=20)
    
    temp = np.mean(predicted, axis=0)
    
    results.append(temp)

results = np.asarray(results)

np.mean(results, axis=0)


# In[32]:

results


# In[33]:

results = []


for i in np.arange(0, 3):
    
    mlp_model.fit(X, Y, batch_size=100, nb_epoch=100, verbose=1)
    
    predicted = mlp_model.predict(test_X_2, batch_size=20)
    
    temp = np.mean(predicted, axis=0)
    
    results.append(temp)

results = np.asarray(results)

np.mean(results, axis=0)


# In[ ]:




# In[28]:

lstm_model = Sequential()
lstm_model.add(LSTM(81, input_dim=9, return_sequences=True))  # returns a sequence of vectors of dimension 32
#lstm_model.add(Dropout(0.5))
lstm_model.add(LSTM(256, return_sequences=True))
#lstm_model.add(Dropout(0.5))
lstm_model.add(LSTM(512, return_sequences=True))
#lstm_model.add(Dropout(0.5))
lstm_model.add(LSTM(512, return_sequences=True))
#lstm_model.add(Dropout(0.5))
lstm_model.add(LSTM(256))
#lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(3, init='uniform', activation='softmax'))
lstm_model.summary()


# In[29]:

lstm_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[30]:

lstm_model.fit(lstm_X, Y, batch_size=100, nb_epoch=500, verbose=1)

#predicted


# In[ ]:

predicted = lstm_model.predict(lstm_test_X_1, batch_size=20)

#x = predicted[:, 1] - predicted[:, 2]

np.mean(predicted, axis=0)


# In[ ]:

predicted = lstm_model.predict(lstm_test_X_2, batch_size=20)

#x = predicted[:, 1] - predicted[:, 2]

np.mean(predicted, axis=0)


# In[ ]:

predicted = lstm_model.predict(lstm_test_X_3, batch_size=20)

#x = predicted[:, 1] - predicted[:, 2]

np.mean(predicted, axis=0)


# In[ ]:




# In[ ]:

cnn_data = np.reshape(data, (data.shape[0], data.shape[1], 1))

cnn_X = cnn_data[:, 0:9, :]
cnn_Y = cnn_data[:, 10, :]


# In[ ]:

cnn_X = np.reshape(X, X.shape + (1,))

cnn_X.shape


# In[ ]:

cnn_X.shape


# In[ ]:

cnn_model = Sequential()
#cnn_model.add(Convolution1D(81, 3, input_dim=9, input_length=1800, border_mode='same'))
#cnn_model.add(Convolution1D(81, 3, activation='relu', input_dim=9))
cnn_model.add(Convolution1D(81, 3, activation='relu', input_shape=(9, 1)))  
cnn_model.add(MaxPooling1D(pool_length=2))

cnn_model.add(Convolution1D(512, 3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_length=2))


cnn_model.add(Convolution1D(512, 3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_length=2))


cnn_model.add(Convolution1D(512, 3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_length=2))

#cnn_model.add(Flatten())

cnn_model.add(Dense(3, init='uniform', activation='softmax'))
cnn_model.summary()


# In[ ]:

cnn_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:

cnn_model.fit(cnn_X, cnn_Y, batch_size=1800, nb_epoch=500, verbose=1)

predicted = cnn_model.predict(test_X_1, batch_size=20)


# In[ ]:

# x = predicted[:, 1] - predicted[:, 2]

np.mean(predicted, axis=0)

