#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:51:50 2023

@author: siddhantsingh
"""
import json
import random
from Main import Simulator
from Visualization import Visualization
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class Main:
               # self, seed, mean, std_dev, err_rate, err_len, smin, smax
    def __init__(self, seed, mean, std_dev, err_rate, err_len, smin, smax, sz, id):
        self.seed = seed
        self.std_dev = std_dev
        self.err_rate = err_rate
        self.err_len = err_len
        self.smin = smin
        self.smax = smax
        self.mean = mean
        simulator = Simulator(seed=self.seed, \
                                mean=self.mean, \
                                std_dev=self.std_dev, \
                                err_rate=self.err_rate, \
                                err_len=self.err_len, \
                                smin=self.smin, \
                                smax=self.smax)
        self.data = []
        self.filename = ''
        self.id = id
        for i in range(sz):
            self.data.append(simulator.calc_next_val())
        self.__save_file__()
        
    def __init__(self):
        pass
    
    def __save_file__(self):
        self.filename = 'data_' + str(self.id) + '.txt'
        with open(self.filename, 'w') as f:
            f.write(json.dumps(self.data))
            print(self.filename + ' successfully loaded...')

    def save_df(self):
        vv = Visualization(self.id)
        (np_mean, np_median) = vv.init_mean_median_modalities()
        time = []
        for i in range(len(self.data)):
            time.append(i+1)
        np_time = np.array(time)
        ds = {'Time': np_time,'Data':self.data, 'Median':np_median, 'Mean':np_mean}
        df = pd.DataFrame(data=ds)
        
        print(df)

        df.to_csv('Data_'+str(len(self.data))+'_id_'+str(self.id))

    def __load_dataset__(self, id):
        v = Visualization(id)
        self.data = v.data
        self.filename = v.filename
        self.id = v.id

    def main():
        print('Initializing NOVA...')
      
        '''
        Important:
            1. Initialize "rotor_data"
                a. COMMENT OUT 'pass' __init__() function for CUSTOM values
                    i. 
                    START
                    rotor_data = Main(seed=random.randint(0, 1e99), mean=17.5, std_dev=5, err_rate=0.01, err_len=10, smin=10, smax=25, sz=10_000, id=6)
                    rotor_data.__save_file__()
                    END
                b. else DO NOT comment it out
            2. Initialize "visualization"
        '''
        #rotor_data = Main(seed=random.randint(0, 1e99), mean=17.5, std_dev=5, err_rate=0.01, err_len=10, smin=10, smax=25, sz=1_000_000, id=7)
        #rotor_data.__save_file__()
        
        rotor_data = Main()
        rotor_data.__load_dataset__(id=1)
        visualization = Visualization(rotor_data.id)
        
        rotor_data.save_df()
        
        '''Visualizations'''
        '''Graph #1'''
        visualization.plot_ALL_fits('NOVA: Rotor Speed Figure', 'Time (s)', 'Rotor Speed (hz)')
        '''Graph #2'''
        visualization.plot_median_data('NOVA: Rotor Speed vs. Time Subset Median', 'Time Subset Median (s)', 'Rotor Speed (hz)')
        '''Graph #3'''
        visualization.plot_imshow_with_time('NOVA: Rotor Speed vs. Time Subset Median vs. Time Subset Mean', 'Time (s)', 'Rotor Speed (hz)')
        '''Graph #4'''
        visualization.plot_imshow_without_time('NOVA: Rotor Speed vs. Time Subset Median vs. Time Subset Mean', 'Median (1/s)', 'Rotor Speed (hz)')
        '''Graph #5'''
        visualization.plot_median_data_histogram('NOVA: Rotor Speed vs. Time Subset Median', 'Time (s)', 'Rotor Speed (hz)', 100)
        visualization.plot_median_data_histogram('NOVA: Rotor Speed vs. Time Subset Median', 'Time (s)', 'Rotor Speed (hz)', 500)
        visualization.plot_median_data_histogram('NOVA: Rotor Speed vs. Time Subset Median', 'Time (s)', 'Rotor Speed (hz)', 20)
        '''Graph #6'''
        visualization.scatterplot_3D_mean_median_data('NOVA: Rotor Speed vs. Time Subset Median vs. Time Subset Mean', 'Rotor Speed', 'Time Subset Median', 'Time Subset Mean')
        
        print('NOVA complete...')  
        
Main.main()
            
        
        