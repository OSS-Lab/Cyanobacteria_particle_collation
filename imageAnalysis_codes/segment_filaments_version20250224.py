#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:30:20 2026

@author: u4139044
"""


import numpy as np
from PIL import Image
from scipy import signal
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from scipy import ndimage
import math
import pandas as pd
from tqdm import tqdm
#%% load image, set params
#save_path = '/Volumes/RNP_T7/20251208_Pseudanabaena/objective_10x_dt_10s_2_'
im_path = '/Volumes/RNP_T7/glass_data/pop_distr_lengths_fullsize/24-02-02_056-059_frame1806'

#'/Volumes/Shared279/OSSLAB_data/microscopy_data/Rebecca/20251208_Pseudanabaena/objective_10x_dt_10s_2/Default/'



framenum = 0

yfixsize = 13
xfixsize = 19 # fixel length is approx half of this value 
# ^^these two values are recorded in the title of the saved results

#neighbour_radius = 25 # 24/1/26: 25 for 20260116_shortfilament long and short filaments
neighbour_radius = 25

#% find threshold for image
# could automate this somehow?
threshold = 250 
#1300 for test images 
# 24/1/26: 230 for 20260116_shortfilament long and short filaments
# 350 for pseudanabaena vid (for paper)

#mask = np.array(Image.open(save_path + 'mask.tif'))/255

save_path_full = im_path + f'_thresh{threshold}_y{yfixsize}_x{xfixsize}_'
#%% plots for threshold check

avg_window = 5
avg_kernel = np.ones((avg_window,avg_window))/avg_window**2
im_tiff = Image.open(im_path+'.tif')# + f'img_time{framenum:09}.tif') #testframe000 f'{n:03}'
im = np.array(im_tiff,dtype = np.float64)
#im = im*mask
im = np.pad(im, 30)

avg = signal.correlate2d(im, avg_kernel, mode='same')
#check by plotting images
plt.figure()
plt.imshow(im,vmin = 0,vmax = threshold)
plt.figure()
plt.imshow(avg>threshold)

#%%
# make cross-structure consisting of two orthogonal transect lines at different angles
angles_from_x_axis = np.pi*np.arange(0,1,1/8)
radius = 10
start_x = (radius*np.cos(angles_from_x_axis)).astype(int)
end_x = -start_x
start_y = (radius*np.sin(angles_from_x_axis)).astype(int)
end_y = -start_y


radius = 3
start_x_2 = (radius*np.cos(angles_from_x_axis)).astype(int)
end_x_2 = -start_x_2
start_y_2 = (radius*np.sin(angles_from_x_axis)).astype(int)
end_y_2 = -start_y_2
#%%
for framenum in [0]:
    im_tiff = Image.open(im_path + '.tif') #+ f'img_time{framenum:09}.tif') #testframe000 f'{n:03}'
    
    im = np.array(im_tiff,dtype = np.float64)#*mask
    del im_tiff
    # pad with 30 pixels of blank either side to avoid edge issues
    im = np.pad(im, 30)
    print(f'opened frame {framenum}')
    #% threshold image to just pixels of interest
    
    # empirical threshold - this should give just the middles of the filaments I think
        
    avg = signal.correlate2d(im, avg_kernel, mode='same')
    
    #start_x, start_y points [0, pi] clockwise from x axis for (0,0) at top left 
    
    #% identify fixels
    # iterate over all locations where intensity > threshold 
    avg_working = avg.copy()
    im_working = im.copy()
    pixels_of_interest = np.where(avg_working>threshold)
    #print(len(what[0]))
    
    fixels = []
    
    #this sets the size of one fixel
    masksize = round(np.linalg.norm([yfixsize,xfixsize])*1.2)
    mask_1 = np.zeros((masksize,masksize))
    mask_1[round((masksize-1)/2)-int((yfixsize-1)/2)  :round((masksize-1)/2)+ int((yfixsize+1)/2) , round((masksize-1)/2)-  int((xfixsize-1)/2)  :round((masksize-1)/2)+ int((xfixsize+1)/2)] = 1
    
    while len(pixels_of_interest[0])>0:  
        holder = len(pixels_of_interest[0])
        i = 0
        # cut out a 10x10 region surrounding pixel of interest
        local_max_radius = 3
        crop = im_working[pixels_of_interest[0][i]-local_max_radius : pixels_of_interest[0][i]+local_max_radius, pixels_of_interest[1][i]-local_max_radius : pixels_of_interest[1][i]+local_max_radius] 
        # find local maximum within a 5 pix radius and guess that's the centre of filament
        y_centre, x_centre = np.unravel_index(np.argmax(crop), crop.shape)
        y_centre += pixels_of_interest[0][i]-local_max_radius
        x_centre += pixels_of_interest[1][i]-local_max_radius
        
        # make line profiles for the different angle lines
        lines = []
        line_sums = []
        for j in range(4):
            line1 = profile_line(im, (start_y[j]+y_centre, start_x[j]+x_centre), (end_y[j]+y_centre, end_x[j]+x_centre))
            line2 = profile_line(im, (start_y[j+4]+y_centre, start_x[j+4]+x_centre), (end_y[j+4]+y_centre, end_x[j+4]+x_centre))
            lines.append([line1, line2])
            line_sums.append([np.sum(line1),np.sum(line2)])
        line_diffs = np.array([line_sum[0]-line_sum[1] for line_sum in line_sums])
        line_diffs = np.concatenate((line_diffs, -line_diffs))
        # fixel orientation determined by max difference in sums along perp. lines
        temp = np.argmax(line_diffs)
        fix_angle = angles_from_x_axis[temp] # [0, pi] clockwise for (0,0) at top left 
        
        mask_2 = ndimage.rotate(mask_1,-fix_angle*180/np.pi, reshape=False, order = 0)
        this_mask = np.zeros((np.shape(im)))
        this_mask[y_centre-round((masksize-1)/2):y_centre+round((masksize+1)/2), x_centre-round((masksize-1)/2):x_centre+round((masksize+1)/2)  ] =  mask_2
        #this_mask_old = this_mask.copy()
        this_mask *= 10000
    
        fixels.append([y_centre, x_centre, fix_angle, temp])
        #plt.figure()
        #plt.imshow(avg_working)
        avg_working = avg_working - this_mask # set fixel area in 'working' image to zero to 'lock' from further identification
        im_working = im_working - this_mask
        #plt.figure()
        #plt.imshow(avg_working)
        pixels_of_interest = np.where(avg_working>threshold)
        if len(pixels_of_interest[0]) == holder: # sometimes it gets stuck at one pixel of interest
            # we just set that pixel to 'not interesting' and make it move on with its life
            avg_working[pixels_of_interest[0][0:3], pixels_of_interest[1][0:3]] -= 10000
            print('removed:',pixels_of_interest[0][0],pixels_of_interest[1][0])
            pixels_of_interest = np.where(avg_working>threshold)
            
#% 
    #% save fixels
    
    np.savetxt(save_path_full+'fixels.csv',fixels)#  f'results/fixels_y{yfixsize}_x{xfixsize}_frame{framenum:09}.csv',fixels)
    print('saved fixels')
    # saves as an array
    #what = np.loadtxt('/Volumes/RNP_T7/glass_data/buckling_vids_round2/24-01-26_026-029/testframe000_fixellist_11_17.csv')
    #% join fixels up
    
    # one row of fixels: fixel [ y_centre, x_centre, fix_angle [rad], index of angle in original array  ]
    #filament_assignment = np.zeros(len(fixels))
    fixels_arr = np.array(fixels)
    # first calculate all pairwise distances 
    all_pairwise_distances = cdist(fixels_arr[:,0:2],fixels_arr[:,0:2]) # N_fixels x 2 array
    # because it's distances between one set of points: gives symmetrical array with zeros on diag.
    
    #% testing distance range for possible neighbours
    # for test crop image, 30pix radius gives 7/350 fixels with no nearby neighbours (except itself)
    # temp = []
    # for i in range(len(fixels)):
    #     nearby_neighbours = np.where(all_pairwise_distances[i,:] < 30)[0]
    #     temp.append(len(nearby_neighbours))
    
    #%
    connectivity_info = np.zeros((len(fixels),len(fixels)))
    # For row i, 
    # contains 1 in column [j] if fixel [j] is a neighbour in the 'startxy' direction
    # i.e. in the 0(inclusive) to pi range clockwise of x axis
    # or a -1 if neighbour in the 'other' direction i.e. pi(inclusive) to 2pi 
    # or 0 if itself or someone else
    
    #%
    for i, fixel in enumerate(fixels):
        all_pairwise_distances[i,i] = 1000000000 # don't waste time looking at yourself!
        # just get the ones that are close enough to be of interest
        nearby_neighbours = np.where(all_pairwise_distances[i,:] < neighbour_radius )[0]
        for neighbour_index in nearby_neighbours:
            potential_neighbour = fixels[neighbour_index]
            neighbour_vector = np.array([potential_neighbour[0]-fixel[0],potential_neighbour[1]-fixel[1]  ]) 
            # [y,x] vector pointing FROM this fixel centre to potential neighbour centre
            #potential_neighbour_centre = (potential_neighbour,0.5)
            neighbour_distance = all_pairwise_distances[i,neighbour_index]      
            
            # I THINK WE NEED TO FINESSE THIS A BIT!!!!!
            
            # work out angle between fixel and potential neighbour      
            # is it in the half-plane surrounding end1 or end2?
            end1 = np.array([np.sin(fixel[2]),np.cos(fixel[2])]) 
            # this is [y,x] unit vector pointing fixel_direction from the x axis clockwise
            end2 = -end1 # and this is the opposite one
            
            
            temp = np.dot(neighbour_vector/neighbour_distance, end1)
            if temp > 0:
                closest_end = end1
                neighbour_polarity = 1
            else:
                closest_end = end2
                neighbour_polarity = -1
                
            temp = np.dot(neighbour_vector/neighbour_distance, closest_end) # equivalent to temp = abs(temp) ???
            neighbour_centre_angle = math.acos(temp) # result [0,pi] or actually [0,pi/2] if temp > 0
            relative_orientation_of_fixels = min( abs(fixel[2] - potential_neighbour[2]), np.pi - abs(fixel[2] - potential_neighbour[2]) )
            if (neighbour_centre_angle<np.pi/6) & (relative_orientation_of_fixels < np.pi/4):
                # centre lies within 'bowtie' AND fixel orientation within 45deg
                # then it's a neighbour
                connectivity_info[i,neighbour_index] = neighbour_polarity
    
            
    #% find NEAREST two neighbours and identify end fixels
    lone_fixels = []
    end_fixels = []
    connectivity_final = np.zeros((len(fixels),len(fixels)))
    for i in range(len(fixels)):
        thing1 = np.where(connectivity_info[i,:] == 1)[0]
        thing2 = np.where(connectivity_info[i,:] ==-1)[0]
        if (len(thing1) == 0) & (len(thing2) == 0):
            lone_fixels.append(i)
        elif (len(thing1) == 0) or (len(thing2) == 0):
            end_fixels.append(i)
        if len(thing1)>0:       
            distances1 = all_pairwise_distances[i,thing1]
            connectivity_final[i,thing1[np.argmin(distances1)]] = 1
        if len(thing2)>0:
            distances2 = all_pairwise_distances[i,thing2]        
            connectivity_final[i,thing2[np.argmin(distances2)]] = -1
    
        
    #% testing: plot end fixels 
     

    
    # plt.figure()
    # plt.imshow(im,cmap = 'gray')
    # for i in range(len(end_fixels)):
    #     fixel = fixels[end_fixels[i]]
    #     temp = fixel[3]
    #     x_centre = fixel[1]
    #     y_centre = fixel[0]
    #     plt.plot([start_x_2[temp]+x_centre,end_x_2[temp]+x_centre],[start_y_2[temp]+y_centre,end_y_2[temp]+y_centre], c = 'r')
    #     plt.scatter([start_x_2[temp]+x_centre,end_x_2[temp]+x_centre],[start_y_2[temp]+y_centre,end_y_2[temp]+y_centre], c = 'y', s= 2)
    
    
    #% assemble from new connectivity matrix
    allfilaments = []
    allfilament_idxs = []
    for end_fixel_idx in end_fixels:
        thisfilament = []
        thisfilament_idxs= []  #IDX TESTING
        thisfilament.append(fixels[end_fixel_idx]) # start it off with the end one   
        thisfilament_idxs.append(end_fixel_idx) #IDX TESTING
        #print('point1, just appended idx', end_fixel_idx)
        
        # then find the next one and add it to grow filament
        this_fixel_idx = np.nonzero(connectivity_final[end_fixel_idx,:] )[0][0] # there should be only one of these
        thisfilament.append(fixels[this_fixel_idx])
        thisfilament_idxs.append(this_fixel_idx)
        #print('point2, just appended idx', this_fixel_idx)
        
        #prev_fixel_idx = end_fixel_idx #store this to eliminate it in a moment
        
        temp = np.nonzero(connectivity_final[this_fixel_idx,:] )[0]  
        #print(temp)
        count = 0
        while len(temp)>1: # until we hit the next end pixel or we time out at 120 fixels
            # find the next one            
            other_neighbours = temp[temp != thisfilament_idxs[-2]]
            if len(other_neighbours)==1:
                this_fixel_idx = other_neighbours[0] #update this_fixel to the next in chain
            else: #work out which of the neighbours is most likely and update
                this_fixel = fixels_arr[this_fixel_idx,:2] 
                prev_fixel =  fixels_arr[thisfilament_idxs[-2]][:2] 
                neighbour_0 =  fixels_arr[other_neighbours[0]][:2]
                neighbour_1 = fixels_arr[other_neighbours[1]][:2]
                prev_to_this = this_fixel-prev_fixel # y,x vector
                this_to_neighbour_0 = neighbour_0 - this_fixel
                this_to_neighbour_1 = neighbour_1 - this_fixel
                # best neighbour is the one which has largest dot product of unit vectors
                # same direction: dot product = 1, opposite direction, dot product -1
                dot_0 = np.dot(prev_to_this/np.linalg.norm(prev_to_this),this_to_neighbour_0/np.linalg.norm(this_to_neighbour_0))
                dot_1 = np.dot(prev_to_this/np.linalg.norm(prev_to_this),this_to_neighbour_1/np.linalg.norm(this_to_neighbour_1))
                          
                this_fixel_idx = other_neighbours[np.argmax(np.array([dot_0,dot_1]))]
                
            #print('point3', thisfilament_idxs[-2])
            # and add it to grow filament
            thisfilament.append(fixels[this_fixel_idx])
            thisfilament_idxs.append(this_fixel_idx)
            temp = np.nonzero(connectivity_final[this_fixel_idx,:] )[0] 
            count+=1
            if count > 120:
                break
        allfilaments.append(np.array(thisfilament))
        allfilament_idxs.append(np.array(thisfilament_idxs))
    #%
    
    #% remove exact duplicates
    coms = np.zeros((len(allfilaments),2))
    basic_duplicates = []
    for i,filament in enumerate(allfilaments):
        temp = np.mean(filament[:,0:2],axis = 0)     
        if len(np.where((coms == temp).all(axis=1))[0])>0:        
            basic_duplicates.append(i)
        coms[i,:] = np.mean(filament[:,0:2],axis = 0)
    
    allfilaments_round1 =  [allfilaments[i] for i in range(len(allfilaments)) if i not in basic_duplicates ]
    allfilament_idxs_round1 =  [allfilament_idxs[i] for i in range(len(allfilament_idxs)) if i not in basic_duplicates ]
    
    #% find filaments that still share an end fixel
    fil_ends = np.zeros((len(allfilament_idxs_round1)*2,2))
    for i,filament in enumerate(allfilament_idxs_round1):
        fil_ends[i*2:i*2+2,0] = [filament[0],filament[-1]]
        fil_ends[i*2:i*2+2,1] = i
    
    sharing_pairs = [] 
    
    end_fixel_shared = []
    for i in range(len(fil_ends)):
        temp = np.where(fil_ends[:,0] == fil_ends[i,0])
        if len(temp[0])>1:
            sharing_pairs.append(list(fil_ends[temp[0],1].astype(int)))
            end_fixel_shared.append(fil_ends[i,0])
    sharing_pairs = pd.DataFrame({'pairs':sharing_pairs,'shared_end':end_fixel_shared})
    sharing_pairs_2 = sharing_pairs.drop_duplicates(subset = 'shared_end')
    # gives the indices of sets of filaments in the 
    # lists all_filament_idxs or all_filaments that share ends 
    #% group them
    if len(sharing_pairs_2) >0:    
        temp = []
        groups = []
        group_ids_for_df = np.zeros((len(sharing_pairs_2)))
        #for i,this_set in enumerate(sharing_pairs_2['pairs']):
        for i,this_set in enumerate(sharing_pairs_2['pairs']):
            if i == 0:
                temp += this_set
                this_group_id = 0
                groups += [this_group_id]*len(this_set)
                group_ids_for_df[i] = this_group_id           
            # if any of the elements in this set already have a group id    
            newlist = [temp.count(fil_id) for fil_id in this_set]  
            ids_already_grouped = np.nonzero(np.array(newlist))[0]
            if len(ids_already_grouped)>0: # if any ids in this_set already have group_id
                temp += this_set
                this_group_id = groups[temp.index(this_set[ids_already_grouped[0]])]
                #print(this_group_id)
                groups += [this_group_id]*len(this_set)
                # if any of them are already in a group then put that in
                group_ids_for_df[i] = this_group_id 
            else:
                temp += this_set
                this_group_id = max(groups)+int(1)
                groups += [this_group_id]*len(this_set)
                #print(this_group_id)
                group_ids_for_df[i] = this_group_id 
        sharing_pairs_2.loc[:,'group_id'] =   group_ids_for_df  
        
        filamentgroups = pd.DataFrame({'filament_id':temp,'group_id':groups})
        filamentgroups.drop_duplicates(subset = 'filament_id',inplace = True)
        #% count up filaments and ends of each group
        
        num_filaments = filamentgroups.groupby(['group_id']).size()
        num_ends = []
        for group_id in range(max(groups)+1):
            temp = []
            thisgroup = filamentgroups.loc[filamentgroups['group_id'] == group_id]['filament_id']
            for fil_id in thisgroup:
                temp+= [allfilament_idxs_round1[fil_id][0],allfilament_idxs_round1[fil_id][-1]]
            num_ends.append(len(np.unique(temp)))
        #num_ends = sharing_pairs_2.groupby(['group_id']).size()
        
        num_filaments_ends = np.zeros((len(num_ends),3))
        num_filaments_ends[:,0] = np.arange(len(num_ends))
        num_filaments_ends[:,1] = num_filaments
        num_filaments_ends[:,2] = num_ends
        
        del num_filaments, num_ends
        #%
        two_fil_groups = num_filaments_ends[np.where(num_filaments_ends[:,1] == 2)[0],0]
        sharing_fils_to_keep = []
        for group_id in two_fil_groups:
            thisgroup = filamentgroups.loc[filamentgroups['group_id'] == group_id]['filament_id'].values
            fil1 = allfilament_idxs_round1[thisgroup[0]]
            fil2 = allfilament_idxs_round1[thisgroup[1]]
            newlist = [list(fil1).count(fil_id) for fil_id in fil2] 
            if np.sum(newlist) == 1: 
                # if filaments only share one fixel (i.e. the end point) then we don't care
                sharing_fils_to_keep+=[thisgroup[0],thisgroup[1]] #keep both
            if np.sum(newlist)/len(newlist)>0.6: # if the two filaments have considerable overlap
                fil1 = allfilaments_round1[thisgroup[0]]
                fil2 = allfilaments_round1[thisgroup[1]]
                temp1 = fil1[1:,3] - fil1[:-1,3] # looks at changes in fixel orientation over filaments
                temp2 = fil2[1:,3] - fil2[:-1,3] # this could probably be improved...
                #keep the one with the lowest average value
                sharing_fils_to_keep.append( thisgroup[np.argmin([ np.sum(abs(temp1))/len(temp), np.sum(abs(temp2))/len(temp) ])]   )
                
        #%
        temp = filamentgroups['filament_id'].values
        sharing_fils_to_discard = [fil_id for i,fil_id in enumerate(temp) if fil_id not in sharing_fils_to_keep]
        
        allfilaments_final = [thisfil for i, thisfil in enumerate(allfilaments_round1) if i not in sharing_fils_to_discard ]
        allfilament_idxs_final = [thisfil for i, thisfil in enumerate(allfilament_idxs_round1) if i not in sharing_fils_to_discard ]
    else:
        allfilaments_final = allfilaments_round1
        allfilament_idxs_final = allfilament_idxs_round1
   
    
    for lone_fixel_idx in lone_fixels:
        fixel = fixels[lone_fixel_idx]
        temp = fixel[3]
        x_centre = fixel[1]
        y_centre = fixel[0]
        thisfilament = np.array([ [start_y_2[temp]+y_centre, start_x_2[temp]+x_centre, fixel[2],temp  ] , [end_y_2[temp]+y_centre, end_x_2[temp]+x_centre, fixel[2],temp  ] ])
        allfilaments_final.append(thisfilament)
        allfilament_idxs_final.append(lone_fixel_idx)
    #%
    y_vals = []
    x_vals = []
    lengths = []
    for filament in allfilaments_final:
        y_vals.append(filament[int(len(filament)/2),0])
        x_vals.append(filament[int(len(filament)/2),1])
        lengths.append(np.cumsum(np.sqrt( (filament[1:,0] - filament[:-1,0])**2 + (filament[1:,1] - filament[:-1,1])**2) )[-1])
        
    f = pd.DataFrame({'y':y_vals,'x':x_vals,'frame':[framenum]*len(y_vals),'points':allfilaments_final,'fixel_idxs':allfilament_idxs_final,'length':lengths})
    f.to_pickle(save_path_full + 'filaments.pkl')#+f'results/allfilaments_frame{framenum:09}.pkl')
    print(f'{len(allfilaments)} filaments found, {len(allfilaments_final)} kept')
    
    
   #%% final plot
#folder = '20260116_shortfilament/timelapse_round2_1/long/'
#20251208_Pseudanabaena/objective_10x_dt_10s_1_
#save_path = '/Volumes/RNP_T7/20260116_shortfilament/long_'

#im_path = '/Volumes/Shared279/OSSLAB_data/microscopy_data/Rebecca/'+ folder
#framenum = 291
#im_tiff = Image.open(im_path + f'img_position001_time{framenum:09}.tif') #testframe000 f'{n:03}'

im_path = '/Volumes/RNP_T7/glass_data/pop_distr_lengths_fullsize/24-02-02_056-059_frame42'
im_tiff= Image.open(im_path + '.tif')


im = np.array(im_tiff,dtype = np.float64)
#im = im*mask
im = np.pad(im, 30)
threshold = 230
#%
plt.figure()
plt.imshow(im, cmap = 'gray',vmax = np.max(im)*0.3  )#,vmin = threshold*0.9,vmax = threshold*1.5)
#allfilaments_final = pd.read_pickle(save_path+f'results/allfilaments_frame{framenum:09}.pkl')

allfilaments_final = pd.read_pickle(im_path + '_thresh250_y13_x19_filaments.pkl')
for i in range(len(allfilaments_final)): 
    filament = allfilaments_final.loc[i,'points']
    plt.plot(filament[:,1],filament[:,0],linewidth = 1.5)
    plt.scatter(filament[0,1],filament[0,0],s = 12,c = 'y',marker = 'x')
plt.title('13 and 19')


#%% testing: plot all fixels 
 
angles_from_x_axis = np.pi*np.arange(0,1,1/8)
radius = 3
start_x_2 = (radius*np.cos(angles_from_x_axis)).astype(int)
end_x_2 = -start_x_2
start_y_2 = (radius*np.sin(angles_from_x_axis)).astype(int)
end_y_2 = -start_y_2

plt.figure()
plt.imshow(im,cmap = 'gray',vmax = threshold * 1.2)
for fixel in fixels:
    temp = fixel[3]
    x_centre = fixel[1]
    y_centre = fixel[0]
    plt.plot([start_x_2[temp]+x_centre,end_x_2[temp]+x_centre],[start_y_2[temp]+y_centre,end_y_2[temp]+y_centre], c = 'r')
    plt.scatter([start_x_2[temp]+x_centre,end_x_2[temp]+x_centre],[start_y_2[temp]+y_centre,end_y_2[temp]+y_centre], c = 'y', s= 2,marker = 'x')



#%% final plot with already loaded stuff
plt.figure()
plt.imshow(im, cmap = 'gray',vmax = threshold * 1.2)
for filament in allfilaments_final: 
    plt.plot(filament[:,1],filament[:,0],linewidth = 4, linestyle = '--')
    plt.scatter(filament[0,1],filament[0,0],s = 12,c = 'y',marker = 'x')
