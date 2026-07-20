#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:49:53 2026

@author: u4139044
"""

import trackpy as tp
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import trackpy.predict
import seaborn as sns
#%% set up
path_root = '/Volumes/RNP_T7/20251208_Pseudanabaena/objective_10x_dt_10s_2_results/'

#%%
f_all = []
for frame in range(400):
    # Centroid X, Centroid Y, End 1 X, End 1 Y, End 2 X, End 2 Y, Length
    f = pd.read_pickle(path_root+f'allfilaments_frame{frame:09}.pkl' )    
    f_all.append(f)
f = pd.concat(f_all)
#%% 
f_crop = f_all[0:120]
f_crop = pd.concat(f_crop)
t = tp.link_df(f_crop,30,memory = 20) 
#%% link with length binning
length_bins =   [0,  6,15,35,55,75,100,200,300,400]
length_fs = [ f_crop[( f_crop['length']>=length_bins[i]) & (f_crop['length']<=length_bins[i+1])] for i in range(len(length_bins)-1) ]
#maxdisps =        [20,30,40,50,50,60, 70,70,70]
maxdisps =        [20,20,30,40,40,50, 60,60,60]

t_all = [ tp.link_df(this_f,maxdisps[i],memory = 4)  for i, this_f in enumerate(length_fs[:7]) ]
#%
max_vals= [max(t['particle']) for t in t_all]
for i in range(1,len(t_all)):
    t_all[i]['particle'] += max_vals[i-1]+1
    max_vals= [max(t['particle']) for t in t_all]

t = pd.concat(t_all)
#%%
t2  = tp.filter_stubs(t,4)
ids = np.unique(t2['particle'])
#%%
framenum = 290
#im_path = '/Volumes/Shared279/OSSLAB_data/microscopy_data/Rebecca/20260116_shortfilament/timelapse_round2_1/long/'
#im = np.array(Image.open(im_path + f'img_position001_time{framenum:09}.tif')) #testframe000 f'{n:03}'

im = np.array(Image.open(f'/Volumes/Shared279/OSSLAB_data/microscopy_data/Rebecca/20251208_Pseudanabaena/objective_10x_dt_10s_1/Default/img_time{framenum:09}.tif'))

im = np.pad(im, 30)
threshold = 230
im_blank = np.zeros(np.shape(im))
#%%
plt.figure()
plt.imshow(im_blank, cmap = 'gray')#,vmin = threshold*0.9,vmax = threshold*1.2)
tp.plot_traj(t2)
plt.xlim([30,1392+30])
plt.ylim([1040+30,30])


#%% make velocities
howsmooth = 3
poly_order = 2
data_toconcat = []
for n_ in tqdm(ids):
    ptrack = t2.loc[t2['particle'] == n_].copy()
    r = ptrack[['x','y']].to_numpy()   
    #r = signal.savgol_filter(r, howsmooth, poly_order,axis = 0)
    v_x = np.gradient(r[:,0])
    v_y = np.gradient(r[:,1])
    v = np.sqrt(v_x**2+v_y**2) # this is in pix/frame
    # plt.figure();
    # plt.scatter(ptrack['frame'],r[:,0],c = 'gray',s = 15)
    # plt.plot(ptrack['frame'],pos_x_smooth)
    # plt.xlabel('t [frames]')
    # plt.ylabel('pos [pix]')
    
    # then update ptrack df
    ptrack['vx'] = v_x
    ptrack['vy'] = v_y
    ptrack['speed'] = v      
    # then add this ptrack to the master df
    data_toconcat.append(ptrack)
    
vdf = pd.concat(data_toconcat,axis = 0,sort = True)
#%%
vdf.to_pickle(path_root +'velocitydf_frames_0_120.pkl')

#%%

plt.figure()

particle_groups = vdf[['length','speed','particle']].groupby(['particle'],as_index=False).median() 
plt.scatter(particle_groups['length']*1.61   , particle_groups['speed']*1.61/60,marker = 'x')
plt.xlim([0,max(particle_groups['length'])*1.1*1.61])
plt.ylim([0,max(particle_groups['speed'])*1.1*1.61/60])
plt.xlabel('length [um]')
plt.ylabel('speed [um/s]')


#%%

long_path = '/Volumes/RNP_T7/20260116_shortfilament/long_results/'
short_path = '/Volumes/RNP_T7/20260116_shortfilament/short_results/'
pseud_path = '/Volumes/RNP_T7/20251208_Pseudanabaena/objective_10x_dt_10s_2_results/'


shortframes = [380,1200]
longframes = [280,900]
pseudframes = [0,230]



allvdfs = []
dt = 60 #s
umpp = 1.61#

for frame in longframes:
    vdf = pd.read_pickle(long_path + f'velocitydf_frames_{frame}_{frame+20}.pkl')
    vdf = vdf[['length','speed','particle']].groupby(['particle'],as_index=False).median() 
    vdf['length']*=umpp
    vdf['speed']*=umpp/dt
    vdf['dataset'] = ['long']*len(vdf)
    allvdfs.append(vdf[ (vdf['speed']>0) & (vdf['length']>10)])

for frame in shortframes:
    vdf = pd.read_pickle(short_path + f'velocitydf_frames_{frame}_{frame+20}.pkl')
    vdf = vdf[['length','speed','particle']].groupby(['particle'],as_index=False).median() 
    vdf['length']*=umpp
    vdf['speed']*=umpp/dt
    vdf['dataset'] = ['short']*len(vdf)
    allvdfs.append(vdf[ (vdf['speed']>0) & (vdf['length']>10)])


dt = 10 #s
umpp = 0.645#
for frame in pseudframes:
    vdf = pd.read_pickle(pseud_path + f'velocitydf_frames_{frame}_{frame+120}.pkl')
    vdf = vdf[['length','speed','particle']].groupby(['particle'],as_index=False).median() 
    vdf['length']*=umpp
    vdf['speed']*=umpp/dt
    vdf['dataset'] = ['pseudanabaena']*len(vdf)
    allvdfs.append(vdf[ (vdf['speed']>0) & (vdf['length']>10)])

#%
allvdfs_ = pd.concat(allvdfs)
g = sns.JointGrid(data = allvdfs_ ,x = 'length',y = 'speed',hue = 'dataset',marginal_ticks = True,xlim=[5,650],ylim=[0.005,3],height = 4.5)
g.plot_marginals(sns.boxplot, log_scale=True)
g.plot_joint(sns.scatterplot) #, color='black', edgecolor='black'
g.set_axis_labels(xlabel='length [μm]', ylabel='median speed [μm/s]',fontsize = 12)


#plt.savefig('fig4_length_distribution.svg',bbox_inches = 'tight')
#%%
means = allvdfs_.groupby(['dataset'],as_index = False).mean()
stds = allvdfs_.groupby(['dataset'],as_index = False).std()