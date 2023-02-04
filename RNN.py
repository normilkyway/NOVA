import json
import random
from Main import Simulator
from Visualization import Visualization
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import matplotlib.pyplot as plt


'''
Created By Harry Yu
2/2/2023

RNN implementation - we will use IsolationForest as our classifier


'''

#load data (code re-use)
data = []
#cnt = str(input('Access id#: '))
cnt = '5' #hard coding 5
filename = 'data_' + cnt + '.txt'
with open(filename, 'r') as f:
    data = json.loads(f.read())
    # print(data)
    print('Successfully loaded ' + filename + '... ')
time = []
for i in range(len(data)):
    time.append(i+1)

np_data = np.array(data)
np_time = np.array(time)

datadict = {"time" : np_time, "data" : np_data}
datadf = pd.DataFrame(data = datadict)

#Classify each point using a simple statistical model
#this will be tweaked based on how severe of a difference an outlier should be
from sklearn.ensemble import IsolationForest

IsoClf = IsolationForest(random_state = 0)
IsoClf.fit(datadf) #fits the data
datadf["scores"] = IsoClf.predict(datadf)
datadfOutliers = datadf[datadf["scores"] == -1]
datadfNormal = datadf[datadf["scores"] == 1]

#plt.scatter(datadfNormal["time"], datadfNormal["data"], alpha = 0.1, color = "red")
#plt.scatter(datadfOutliers["time"], datadfOutliers["data"], alpha = 0.2 )
#plt.show()  


#We will now redefine the train set as our valid data, and our test set as the outliers
train = datadfNormal[['time', 'data']]
test = datadfOutliers[['time', 'data']]
#print(x_train.iloc[0].shape)  #1x3 shape right now

#normalize the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(train)
x_test = scaler.transform(test)

#LSTM expects a 3d tensor (data samples, time steps, features)
#reshaping..
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1]) #3 samples, time_step can be changed, I will tweak this at some point to test
print("xtrain shape:" ,x_train.shape)
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
print("xtest shape:", x_test.shape)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers


#autoencoding RNN --
#timesteps are iterated through, memory is preserved on sequential patterns
def autoencoder_model(X):
    inputs = Input(shape = (X.shape[1], X.shape[2])) #1, 3
    L1 = LSTM(16, activation = 'relu', return_sequences = True)(inputs) #() for sequential
    L2 = LSTM(4, activation = 'relu', return_sequences = False)(L1) #false to have 3d shape preserved after repeat vector
    L3 = RepeatVector(X.shape[1])(L2) #spreading out compressed data over the time step
    L4 = LSTM(4, activation = 'relu', return_sequences = True)(L3)
    L5 = LSTM(16, activation = 'relu', return_sequences = True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5) #time distributed to have the dense layer apply to each timestep instead of the last element of the tensor
    model = Model(inputs = inputs, outputs = output)
    return model

model = autoencoder_model(x_train) #initialize model
model.compile(optimizer = 'adam', loss = 'mae')
#model.summary()

#now train
nb_epochs = 60
batch_size = 10 #for memory usage
history = model.fit(x_train, x_train, epochs = nb_epochs, batch_size = batch_size, validation_split = 0.05).history 

#plot training losses
fig, ax = plt.subplots(figsize = (14, 6), dpi=80)
ax.plot(history['loss'], 'r', label = 'Train', linewidth = 2)
ax.plot(history['val_loss'], 'b', label = 'Validation', linewidth = 2)
ax.set_title('Model Loss')
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc = 'upper right')
plt.show()

#Plot the losses, figure out a threshhold, then label those points

#the loss distribution offers a basic look into the contamination of the data
#we will compare predicted vs actual xtrain
x_pred = model.predict(x_train) #using x_train, we can find the anomaly limit
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[2]) #x_pred shape (data samples, features)
x_pred = pd.DataFrame(x_pred, columns = train.columns)
x_pred.index = train.index

scored = pd.DataFrame(index= train.index)
xtrain = x_train.reshape(x_train.shape[0], x_train.shape[2]) #same shape for comparison
scored['Loss_mae'] = np.mean(np.abs(x_pred-xtrain), axis = 1) #the literall loss

sns.displot(scored['Loss_mae'], bins = 20, kde = True, color = 'blue')
plt.xlim([0.0, 0.2])
plt.show()

#Find the threshold with a subject to change calculation
threshold = scored['Loss_mae'].quantile(0.95) #so everything above this value is considered an outlier
print(threshold)


#the loss may be tiny due to the simplicity of the data
#a possible calculation to determine this automatically in the future involves using a certain percentile of data or standard deviation of error
#Now, we will calculate the loss on the test set.
x_pred = model.predict(x_test)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[2]) #cut out time_steps
x_pred = pd.DataFrame(x_pred, columns = test.columns)
x_pred.index = test.index

#we have normalized the predictions to the actual

#Visualization of the anomaly threshold
scored = pd.DataFrame(index = test.index)
xtest = x_test.reshape(x_test.shape[0], x_test.shape[2])
scored['loss_mae'] = np.mean(np.abs(x_pred-xtest), axis = 1)
scored['threshold'] = threshold
scored['anomaly'] = scored['loss_mae'] > scored['threshold'] #when the loss is more than it should be

#apply the same thing to the train set and then combine
x_pred_train = model.predict(x_train)
x_pred_train = x_pred_train.reshape(x_pred_train.shape[0], x_pred_train.shape[2])
x_pred_train = pd.DataFrame(x_pred_train, columns = train.columns)
x_pred_train.index = train.index

scored_train = pd.DataFrame(index = train.index)
scored_train['loss_mae'] = np.mean(np.abs(x_pred_train - xtrain)) #xtrain is reshaped from above
scored_train['threshold'] = threshold
scored_train['anomaly'] = scored_train['loss_mae'] > scored_train['threshold']
scored = pd.concat([scored, scored_train])

#scored.plot(logy = True, figsize = (16, 9), color = ['blue', 'red'])
#plt.show()

datadf['scores'] = scored['anomaly']
outliers = datadf[datadf["scores"] == 1]
plt.scatter(outliers['time'], outliers['data'], alpha = 0.1)
plt.show()