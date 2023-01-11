#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 1/2/2023
Program: NOVA Statistical Analysis (matplotlib)
    1. Regression Analysis: https://www.statology.org/scatterplot-with-regression-line-python/
        a. Linear polynomial fit
        b. Quadratic polynomial fit
        c. Cubic polynomial fit
    2. Mean Line Analysis
    3. Median Line Analysis
    4. 3D Plot
    5. Vs. all variables
    6. REIMPLEMENT Sklearn Models
        a. Sklearn --> DBSCAN
        b. Sklearn --> IsolationForest
            i. https://blog.paperspace.com/anomaly-detection-isolation-forest/
            ii. https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html 
    7. Outlier detection
        i. https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
    8. 2D Histogram
        i. https://www.google.com/search?q=hist2d+matplotlib&sxsrf=ALiCzsZPDiW9rd5wFbFMndaToAucS7QOQQ:1672704258701&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiftp35jKr8AhVUg2oFHYwvAWoQ_AUoAXoECAEQAw#imgrc=BfcelMz--Wwv8M
TODO: 
#6, #8, 
    
Helpful: 
MatPlotLib: https://matplotlib.org/cheatsheets/handout-beginner.pdf
MatPlotLib: Plotting Catalog
    i. https://matplotlib.org/stable/gallery/mplot3d/index.html 
    ii. https://schroederindustries.com/wp-content/uploads/Condition-Monitoring-Industry-4.0_2021.pdf
    iii. file:///Users/siddhantsingh/Downloads/1-s2.0-S221282711930126X-main.pdf
    iv. https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7
    
Discards:
Potentially Residual Plot
    i. https://www.statology.org/residual-plot-python/
    ii. https://www.statisticshowto.com/residual-plot/
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from Median_Fit_Calculator import MedianHeap 

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report

data = []
cnt = str(input('Access id#: '))
filename = 'data_' + cnt + '.txt'
with open(filename, 'r') as f:
    data = json.loads(f.read())
    # print(data)
    print('Successfully loaded ' + filename + '... ')
time = []
for i in range(len(data)):
    time.append(i+1)
'''
1. Regression line
2. Mean Line
3. Median Line
4. Statistically Significant Values
'''
np_data = np.array(data)
np_time = np.array(time)

'''Regression Line(s)'''
rl_1 = np.polyfit(np_time, np_data, 1) # linear polynomial fit
rl_2 = np.polyfit(np_time, np_data, 2) # quadratic polynomial
rl_3 = np.polyfit(np_time, np_data, 3) # cubic polynomial

'''Mean/Median Lines'''
running_sum = 0
mean = []
median = []
median_heap = MedianHeap()
for i in range(len(data)):
    running_sum += data[i]
    mean.append(running_sum/(i+1))
    median_heap.insert(data[i])
    median.append(median_heap.calculate_median())
np_mean = np.array(mean)
np_median = np.array(median)

'''Plotting'''
'''mean/median analysis seem to be predictors'''
def plot_ALL_fits():
    plt.title('Rotor-Speed over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotor-Speed (r)')
    plt.plot(np_time, np_data)

    # plotting regression
    plt.plot(rl_1[0]*np_time+rl_1[1], color='red', label='linear')
    plt.plot(rl_2[0]*np_time**2+rl_2[1]*np_time+rl_2[2], color='orange', label='quadratic')
    plt.plot(rl_3[0]*np_time**3+rl_3[1]*np_time**2+rl_3[2]*np_time+rl_3[3], color='green', label='cubic')

    # plotting mean/median
    plt.plot(np_mean, color='blue', label='mean')
    plt.plot(np_median, color='purple', label='median')

    plt.legend(loc='upper right', title='Fitted Functions')
    plt.show()

'''inconclusive'''
def plot_imshow_with_time():
    plt.figure(figsize=(10, 6))
    np_df = np.row_stack((np_time, np_data, np_mean, np_median))
    plt.imshow(np_df, extent=[0, len(data), 0.5, -0.5], aspect='auto')
    plt.show()

'''inconclusive'''
def plot_imshow_without_time():
    plt.figure(figsize=(10, 6))
    np_df = np.row_stack((np_data, np_mean, np_median))
    plt.imshow(np_df, extent=[0, len(data), 0.5, -0.5], aspect='auto')
    plt.show()
    

'''STRONG clustering pattern'''
def plot_median_data():
    plt.figure(figsize=(10, 6))
    plt.scatter(np_median, np_data)
    plt.show()
    
def plot_mean_data():
    plt.figure(figsize=(10, 6))
    plt.scatter(np_mean, np_data)
    plt.show()
    
def plot_histogram(binsz):
    plt.figure(figsize=(10, 6))
    plt.hist(np_data, binsz)
    plt.show()

'''Significant Potential: Time, etc.'''
def plot_median_data_histogram(binsz):
    plt.figure(figsize=(10, 6))
    plt.hist2d(np_data, np_median, binsz)
    plt.show()

def scatterplot_3D_mean_median_data():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np_data, np_median, np_mean, marker='D')
    plt.title('3D Mean, Median, Mode Data (' + filename + ')')
    ax.set_xlabel('np-data')
    ax.set_ylabel('np-median')
    ax.set_zlabel('np-mean')
    plt.show()

def box_plot():
    sns.boxplot(data=np_data)

# plot_median_data_histogram(100)
# scatterplot_3D_mean_median_data()

'''Outlier Detection with prebuilt machine learning models (sklearn)'''
'''re-implement'''
def sklearn_DBSCAN_outliers():
    outlier_detection = DBSCAN(min_samples=100, eps=3)
    np_data_transformed = np_data.copy().reshape(1, -1)
    
    clusters = outlier_detection.fit_predict(np_data_transformed)
    print(clusters)
    print(list(clusters).count(-1))

'''re-implement'''
def sklearn_isolation_forest():
    np_df = np.row_stack((np_data, np_mean, np_median))
    clf = IsolationForest(max_samples=len(data), random_state = 1, contamination= 'auto')
    preds = clf.fit_predict(np_df)
    print(preds)

'''KMeans Clustering Algorithm'''
def sklearn_KMeans():
    
#sklearn_DBSCAN_outliers()
#sklearn_isolation_forest()
