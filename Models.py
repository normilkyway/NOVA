import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

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

#two axis
np_data = np.array(data)
np_time = np.array(time)

datadict = {"time" : np_time, "data" : np_data}
datadf = pd.DataFrame(data = datadict)

#LOF novel dataset
with open('data_6.txt', 'r') as f:
    data2 = json.loads( f.read())
    # print(data)
    print('Successfully loaded novel')
time2 = []
for i in range(len(data2)):
    time2.append(i+1)

np_data2 = np.array(data2)
np_time2 = np.array(time2)

datadictnovel = {"time" : np_time2, "data" : np_data2}
datadfnovel = pd.DataFrame(data = datadictnovel)

#plt.scatter(datadictnovel["time"], datadictnovel["data"])
#plt.show()

# SEVERAL MODELS NOW

from sklearn.ensemble import IsolationForest

IsoClf = IsolationForest(random_state = 0)
IsoClf.fit(datadf) #fits the data
datadf["scores"] = IsoClf.predict(datadf)
datadfOutliers = datadf[datadf["scores"] == -1]
datadfNormal = datadf[datadf["scores"] == 1]

fig, axs = plt.subplots(3, 2)

axs[0,0].scatter(datadfNormal["time"], datadfNormal["data"], alpha = 0.1, color = "red")
axs[0,0].scatter(datadfOutliers["time"], datadfOutliers["data"], alpha = 0.2 )
axs[0,0].set_title("Isolation Forest")

#plt.scatter(datadfNormal["time"], datadfNormal["data"], alpha = 0.1, color = "red")
#plt.scatter(datadfOutliers["time"], datadfOutliers["data"], alpha = 0.2 )
#plt.show()


from sklearn.svm import OneClassSVM

datadict = {"time" : np_time, "data" : np_data}
datadf = pd.DataFrame(data = datadict)

OSVMclf = OneClassSVM(gamma = 'scale')
OSVMclf.fit(datadf)
datadf["scores"] = OSVMclf.predict(datadf)
datadfOutliers = datadf[datadf["scores"] == -1]
datadfNormal = datadf[datadf["scores"] == 1]

print("SVM Calculated")

axs[0,1].scatter(datadfNormal["time"], datadfNormal["data"], alpha = 0.1, color = "red")
axs[0,1].scatter(datadfOutliers["time"], datadfOutliers["data"], alpha = 0.2 )
axs[0,1].set_title("One Class SVM")


from sklearn.neighbors import LocalOutlierFactor  #THIS IS NOVELTY-DETECTION, SLIGHTLY DIFFERENT

datadf = pd.DataFrame(data = datadict)

LOFclf = LocalOutlierFactor(novelty = True, contamination = 0.05) #params are mostly auto
LOFclf.fit(datadf)
datadfnovel["scores"] = LOFclf.predict(datadfnovel)
datadfOutliers = datadfnovel[datadfnovel["scores"] == -1]
datadfNormal = datadfnovel[datadfnovel["scores"] == 1]

axs[1, 0].scatter(datadfNormal["time"], datadfNormal["data"], alpha = 0.1, color = "red")
axs[1, 0].scatter(datadfOutliers["time"], datadfOutliers["data"], alpha = 0.2, color = "blue" )
axs[1, 0].sharey(axs[0 , 0])
axs[1, 0].set_title("LocalOutlierFactor")


from sklearn.neighbors import KernelDensity

datadict = {"time" : np_time, "data" : np_data}
datadf = pd.DataFrame(data = datadict)

KDclf = KernelDensity() #params don't really need fiddling
KDclf.fit(datadf)
datadf["scores"] = KDclf.score_samples(datadf)

axs[1, 1].scatter(datadf["time"], datadf["scores"], alpha = 0.2) #here you can filter based on the probability
axs[1, 1].set_title("KernelDensity")



from sklearn.cluster import DBSCAN

datadf = pd.DataFrame(data = datadict)

DBSCANclf = DBSCAN(eps = 10, min_samples = 2)  #eps - maximum distance between samples to be considered same neighborhood(tweak), min_samples is the minimum samples to be considered neighborhood
datadf["cluster"] = DBSCANclf.fit(datadf).labels_ 

categories = np.unique(datadf['cluster'])
colors = np.linspace(0, 1, len(categories))
colordict = dict(zip(categories, colors))

datadf["Color"] = datadf['cluster'].apply(lambda x: colordict[x])
axs[2, 0].scatter(datadf["time"], datadf["data"], c = datadf["Color"])
axs[2, 0].set_title('DBSCAN eps = 10')

###############################################################################################################
#DBSCAN with different eps

datadf = pd.DataFrame(data = datadict)

DBSCANclf2 = DBSCAN(eps = 4, min_samples = 2)
datadf["cluster"] = DBSCANclf2.fit(datadf).labels_

categories = np.unique(datadf['cluster'])
colors = np.linspace(0, 1, len(categories))
colordict = dict(zip(categories, colors))

datadf["Color"] = datadf['cluster'].apply(lambda x: colordict[x])
axs[2, 1].scatter(datadf["time"], datadf["data"], c = datadf["Color"])
axs[2, 1].set_title('DBSCAN eps = 4')


plt.show()
#TENSOR FLOW - many options



