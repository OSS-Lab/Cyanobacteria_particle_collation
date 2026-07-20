#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:09:49 2026

@author: u4139044
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#%%
root_path = '/Volumes/Shared279/OSSLAB_projectsShared/cyano_ironForaging_story/glass_data_buckles_plectonemes/'
folders = ['24-01-26_026-029/','24-01-26_030-033/','24-02-02_056-059/']
folders = [root_path + folder for folder in folders ]
buckle_only_lengths = []
plectoneme_lengths = []
for folder in folders:
    data_path = folder + 'buckle_only/'
    alldata = sorted(os.listdir(data_path))
    for filename in alldata:
        if (filename[-3:] == 'csv') & (filename[0]!='.'):
            thisfil = pd.read_csv(data_path +filename)
            buckle_only_lengths.append(thisfil['running average length'].sum())
    data_path = folder + 'plectonemes/'
    alldata = sorted(os.listdir(data_path))
    for filename in alldata:
        if (filename[-3:] == 'csv') & (filename[0]!='.'):
            thisfil = pd.read_csv(data_path +filename)
            plectoneme_lengths.append(thisfil['running average length'].sum())


# NOTE 8/1/26 running average length does a much better job than basic skeleton

#%%
root_path_pop = '/Volumes/Shared279/OSSLAB_projectsShared/cyano_ironForaging_story/pop_distr_lengths_fullsize/'
umpp = 1.61
pop_dataset = []
length_data = os.listdir(root_path_pop)
for thing in length_data:
    if (thing[-3:] == 'pkl') & (thing[0] != '.'):
        pop_dataset.append(pd.read_pickle( root_path_pop+ thing))

pop_dataset = pd.concat(pop_dataset)
pop_lengths = umpp*pop_dataset['length'].values
#%%
temp1 = len(pop_lengths)
temp2 = len(buckle_only_lengths)
temp3 = len(plectoneme_lengths)

plt.rc('font',size = 10,family = 'Arial')
fig, ax = plt.subplots(2,1, figsize=(3.5,3),layout = 'constrained')
ax[0].hist(pop_lengths,bins = np.arange(0,1500,50),color = [0, 0, 1, 0.5],edgecolor = 'b',label = f'population (N = {temp1})',density = True)
ax[0].set_ylabel(r'Prob. density [μm$^{-1}$]')
ax[0].vlines([120,320],0.00001,0.1,colors = ['r','g'],linestyles = ['--','--'],label = '_blah')
ax[0].set_xlim([0,1500])
ax[0].set_yscale('log')
ax[0].set_ylim([10**-4.8, 10**-1.8])
ax[0].set_xticklabels([])
ax[0].legend()


ax[1].hist(np.array(buckle_only_lengths)*umpp*2,bins = np.arange(0,1500,50),color = [1, 0, 0, 0.5],edgecolor = 'r',label = f'buckle only (N = {temp2})')
ax[1].hist(np.array(plectoneme_lengths)*umpp*2,bins = np.arange(0,1500,50),color = [0, 1, 0, 0.5],edgecolor = 'g',label = f'plectoneme (N = {temp3})')
ax[1].set_ylabel('Counts')
ax[1].vlines([120,320],0.1,1000,colors = ['r','g'],linestyles = ['--','--'],label = '_blah')
ax[1].set_xlabel('Filament length [μm]')
ax[1].set_yscale('log')
ax[1].legend()
ax[1].set_xlim([0,1500])
ax[1].set_ylim([0.7,100])
plt.savefig('/Users/u4139044/Documents/fig_lengthdistr.svg',bbox_inches = 'tight')