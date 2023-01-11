#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:53:48 2023

@author: siddhantsingh
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

class Visualization:
    def __init__(self, id):
        self.data = []
        self.id = id
        self.filename = 'data_' + str(id) + '.txt'
        with open(self.filename, 'r') as f:
            self.data = json.loads(f.read())
            # print(data)
            print('Successfully loaded ' + self.filename + '... ')
        self.time = []
        for i in range(len(self.data)):
            self.time.append(i+1)
        self.np_data = np.array(self.data)
        self.np_time = np.array(self.time)
        self.np_mean = []
        self.np_median = []
        self.init_mean_median_modalities()
        
        self.title_add = ' (' + self.filename + ')'
        
    '''Regression Line(s)'''
    def regression_line(self, degree):
        if degree < 1 or degree > 3: 
            pass
        return np.polyfit(self.np_time, self.np_data, degree)
    
    '''Mean/Median Lines'''
    def init_mean_median_modalities(self):
        running_sum = 0
        mean = []
        median = []
        median_heap = MedianHeap()
        for i in range(len(self.data)):
            running_sum += self.data[i]
            mean.append(running_sum/(i+1))
            median_heap.insert(self.data[i])
            median.append(median_heap.calculate_median())
        self.np_mean = np.array(mean)
        self.np_median = np.array(median)
        return (self.np_mean, self.np_median)

    '''Plotting'''
    '''mean/median analysis seem to be predictors'''
    def plot_ALL_fits(self, title, x_label, y_label):
        title += self.title_add
        rl_1 = self.regression_line(1)
        rl_2 = self.regression_line(2)
        rl_3 = self.regression_line(3)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(self.np_time, self.np_data)
    
        # plotting regression
        plt.plot(rl_1[0]*self.np_time+rl_1[1], color='red', label='linear')
        plt.plot(rl_2[0]*self.np_time**2+rl_2[1]*self.np_time+rl_2[2], color='orange', label='quadratic')
        plt.plot(rl_3[0]*self.np_time**3+rl_3[1]*self.np_time**2+rl_3[2]*self.np_time+rl_3[3], color='yellow', label='cubic')
    
        # plotting mean/median
        plt.plot(self.np_mean, color='green', label='mean')
        plt.plot(self.np_median, color='blue', label='median')
    
        plt.legend(loc='upper right', title='Fitted Functions')
        plt.show()

    '''inconclusive'''
    def plot_imshow_with_time(self, title, x_label, y_label):
        title += self.title_add
        plt.figure(figsize=(10, 6))
        np_df = np.row_stack((self.np_time, self.np_data, self.np_mean, self.np_median))
        plt.imshow(np_df, extent=[0, len(self.data), 0.5, -0.5], aspect='auto')
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.show()

    '''inconclusive'''
    def plot_imshow_without_time(self, title, x_label, y_label):
        title += self.title_add
        plt.figure(figsize=(10, 6))
        np_df = np.row_stack((self.np_data, self.np_mean, self.np_median))
        plt.imshow(np_df, extent=[0, len(self.data), 0.5, -0.5], aspect='auto')
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()
        
    
    '''STRONG clustering pattern'''
    def plot_median_data(self, title, x_label, y_label):
        title += self.title_add
        plt.figure(figsize=(10, 6))
        plt.scatter(self.np_median, self.np_data)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()
        
    def plot_mean_data(self, title, x_label, y_label):
        title += self.title_add
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.np_mean, self.np_data)
        plt.show()
        
    def plot_histogram(self, title, x_label, y_label, binsz):
        title += self.title_add
        plt.figure(figsize=(10, 6))
        plt.hist(self.np_data, binsz)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.show()
    
    '''Significant Potential: Time, etc.'''
    def plot_median_data_histogram(self, title, x_label, y_label, binsz):
        title += self.title_add
        plt.figure(figsize=(10, 6))
        plt.hist2d(self.np_data, self.np_median, binsz)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()
    
    def scatterplot_3D_mean_median_data(self, title, x_label, y_label, z_label):
        title += self.title_add
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.np_data, self.np_median, self.np_mean, marker='D')
        
        plt.title(title)#'3D Mean, Median, Mode Data (' + self.filename + ')')
        ax.set_xlabel(x_label)#'np-data')
        ax.set_ylabel(y_label)#'np-median')
        ax.set_zlabel(z_label)#'np-mean')
        plt.show()
    
    def box_plot(self, title, x_label, y_label):
        sns.boxplot(data=self.np_data, x=x_label, y=y_label)
    
    '''Outlier Detection with prebuilt machine learning models (sklearn)'''
    '''re-implement'''
    def sklearn_DBSCAN_outliers(self, title, x_label, y_label):
        outlier_detection = DBSCAN(min_samples=100, eps=3)
        np_data_transformed = self.np_data.copy().reshape(1, -1)
        
        clusters = outlier_detection.fit_predict(np_data_transformed)
        print(clusters)
        print(list(clusters).count(-1))
    
    '''re-implement'''
    def sklearn_isolation_forest(self, title, x_label, y_label):
        np_df = np.row_stack((self.np_data, self.np_mean, self.np_median))
        clf = IsolationForest(max_samples=len(self.data), random_state = 1, contamination= 'auto')
        preds = clf.fit_predict(np_df)
        print(preds)
    
    '''KMeans Clustering Algorithm'''
    def sklearn_KMeans():
        pass
    
# plot_median_data_histogram(100)
# scatterplot_3D_mean_median_data()  
#sklearn_DBSCAN_outliers()
#sklearn_isolation_forest()