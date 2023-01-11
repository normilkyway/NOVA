#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:59:29 2023

What is Iris?

@author: siddhantsingh
"""

from sklearn import datasets
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report


iris = datasets.load_iris()
X = scale(iris.data)
y = pd.DataFrame(iris.target)
var_names = iris.feature_names


