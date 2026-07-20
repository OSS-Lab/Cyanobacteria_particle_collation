#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:53:52 2025

@author: u4139044
"""

import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import signal
import trackpy.predict
import matplotlib as mpl



###### fig 1 E #######
#%% define function for making vdf

def make_vdf(data_path):
    f = pd.read_csv(data_path,header = 0,index_col = 0)
    f = f.rename(columns={'X': 'x', 'Y': 'y', 'Slice': 'frame'})
    t = tp.link_df(f, 7, memory=15) # Have manually checked: 7pix/frame is definitely a good upper bound
    t2 = tp.filter_stubs(t,10)
    ids = np.unique(t2['particle'])
    vdf = pd.DataFrame(columns = ['particle','x','y','vx','vy','frame','Area','speed'])
    data_toconcat = []
    for n_ in tqdm(ids):
        ptrack = t2.loc[t2['particle'] == n_].copy()
        r = ptrack[['x','y']].to_numpy()
        r = signal.savgol_filter(r,7,3,axis = 0)
        v = np.gradient(r, axis = 0)
        #update df with the speeds for ptrack and put into m/s
        ptrack[['vx','vy']] = v
        ptrack['speed'] = np.sqrt(v[:,0]**2 + v[:,1]**2)
        ptrack[['x','y']] = r #add smoothed positions to vdf
        # then add this ptrack to the master df
        ptrack = ptrack.astype(float)
        data_toconcat.append(ptrack)  
    vdf = pd.concat(data_toconcat,axis = 0,sort = True)
    vdf.index.names = ['']
    return vdf

#% make tracks and velocities for the 4 videos

path_root = '/Volumes/Shared279/OSSLAB_projectsShared/cyano_ironForaging_story/zenodo_data_shortcuts/fig1'
mmpp = 0.03 
umpp = 30
data_paths = ['2025-10-09_beakers/right.csv','2025-10-09_beakers/left.csv','2025-12-02_beakers/right.csv','2025-12-02_beakers/left.csv']

vdfs = []
for path in data_paths:
    f = pd.read_csv(path_root + path,header = 0,index_col = 0)
    f = f.rename(columns={'X': 'x', 'Y': 'y', 'Slice': 'frame'})    
    vdfs.append(make_vdf(path_root +path))
    
    
#% plot max cluster size over time

plt.figure(figsize = (2,1.6))
plt.rc('font',size =8,family = 'Arial')
plt.rc('axes',edgecolor='gray')
plt.tick_params(direction = 'inout',length = 3,color = 'gray')

for vdf in vdfs:
    maxvals = vdf.groupby(['frame'],as_index=False).max()
    temp = maxvals['Area']*mmpp**2
    plt.plot(np.arange(len(temp))/60,temp,linewidth = .8)
plt.xlabel('time [hrs]')
plt.ylabel(r'area of largest cluster [mm$^2$]')
plt.legend(['20251009_right','20251009_left','20251202_right','20251202_left'])
plt.ylim([0,4])
plt.xlim([0,2000/60])

#% plot histogram of speeds 
plt.figure(figsize = (2,1.4))
plt.rc('font',size =16,family = 'Arial')
plt.rc('axes',edgecolor='gray')
plt.tick_params(direction = 'in',length = 1,color = 'gray')
for i,vdf in enumerate(vdfs):
    plt.hist(vdf['speed']*umpp/60,bins = np.arange(0,4.1,0.4),histtype = 'step')  #in um/s
plt.xlabel('speed [um/s]')
plt.ylabel('counts')
plt.yscale('log')
plt.xlim([0,4])
#plt.savefig('fig1_speeds.png',dpi = 300,bbox_inches = 'tight')




##### fig 2b
#%% load in imagej results
path_root = '/Volumes/Shared279/OSSLAB_data/microscopy_data/Rebecca/'
vids = ['20250606_50umbeads_petri/','20250623_50umbeads_petri/','20251030_50umbeadsdish/dish3_30s_1/']

umpp = 1.61

#%%
styles = ['-','--']
cmap = mpl.colormaps['hot']
colors = cmap(np.linspace(0, 1, 6))
plt.figure(figsize = (2,2))

for i in [0,1]:
    results = pd.read_csv(path_root + vids[i]+'cluster_size_analysis_results.csv',header = 0,index_col = 0)
    for j, frame in enumerate([1,10,100,497]):
        thisframe = results[  (results['Slice']>=frame) & (results['Slice']<frame+2) ]
        vals,bins = np.histogram(thisframe['Area']*umpp**2,bins = umpp**2*np.concatenate((np.array([0]),np.exp(np.arange(7,11,0.5)))) ,density = True)
        plt.plot((bins[1:]+bins[:-1])/2, vals,linestyle = styles[i],color = colors[j])
       
plt.legend(['0 mins','10 mins','100 mins','500 mins'])   
plt.xlim([300*umpp**2,20000*umpp**2])
plt.xlabel('cluster area [um^2]')
plt.ylabel('P(area)')
plt.yscale('log')
plt.xscale('log')

#plt.savefig('/fig2.svg',bbox_inches = 'tight')


#### fig 4 E
#%% make tracks and velocities for the 3 videos

path_root = '/Volumes/Shared279/OSSLAB_data/macro_meso_imaging_cyano/macro_beaker_flask/'
mmpp = 0.03 
umpp = 30
data_paths = ['2026-01-16_long_short/long_results.csv','2026-01-16_long_short/short_results.csv','2025-10-21_beakers_M77/right_results.csv']

dfs = []
for path in data_paths:
    f = pd.read_csv(path_root + path,header = 0,index_col = 0)
    f = f.rename(columns={'X': 'x', 'Y': 'y', 'Slice': 'frame'})
    dfs.append(f)  

#% plot max cluster size over time

plt.figure(figsize = (2,1.6))
plt.rc('font',size =8,family = 'Arial')
plt.rc('axes',edgecolor='gray')
plt.tick_params(direction = 'inout',length = 3,color = 'gray')

for vdf in vdfs:
    maxvals = vdf.groupby(['frame'],as_index=False).max()
    temp = maxvals['Area']*mmpp**2
    plt.plot(np.arange(len(temp))/60,temp,linewidth = .8)
plt.xlabel('time [hrs]')
plt.ylabel(r'area of largest cluster [mm$^2$]')
plt.legend(['long','short','Pseudanabaena'])

plt.ylim([0,0.8])
plt.xlim([0,12])

#plt.savefig('fig4_clustersize.svg',bbox_inches = 'tight')

