'''
This code standarizes the label datasets using the individual means and standard
deviations corresponding to each label. And shuffle the entire datasets and then
stores them in a new h5py database file containing the shuffled spectra, and
standarized labels together with the individual mean and standard deviations.
The mean and standard deviations are stored so that re-standarization can be performed
in the testing phase (file 4.test.py)
'''

import h5py
import numpy as np
import random

np.random.seed(123)

folder_path = ''
tobe_norm_dataset = 'training_dataset.hdf5'                  #hdf5 for the original un-normalized dataset coming from dumpfile
norm_dataset = 'n_s_training_dataset.hdf5'                   #hdf5 for normalized training dataset
moments = 'moments.hdf5'                                     #hdf5 for moments of labels to be used later on for denormalization

f = h5py.File(folder_path+tobe_norm_dataset,'r')                         #load and shuffle the examples
spectra = f['X']
temp = f['TEMP']
logg = f['LOGG']
vrot = f['VROT']
meta = f['META']

M = np.column_stack((spectra,temp,logg,vrot,meta))
np.random.shuffle(M)

num_example = spectra.shape[0]                                #number of spectra example
num_labels  = 4                                               #number of labels(TEMP, LOGG, VROT, META)
num_flux    = spectra.shape[1]                                #number of flux points

y_norm = np.zeros((num_example, num_labels))
norm = np.zeros((num_labels, 2))

for i in range(num_labels):                                   #calculating the moments for the labels
    mean = np.mean(M[:,10800+i])
    std  = np.std(M[:,10800+i])
    norm[i,:] = np.column_stack((mean,std))

for i in range(num_labels):
    for j in range(num_example):
        y_norm[j][i] = (M[j][10800+i]-norm[i][0])/ (norm[i][1]+1e-011)     #standarization procedure


X  = M[:,:10800]                                             #storinf the shuffled spectra
temp_norm = y_norm[:,0]                                      #storing the normalized labels
logg_norm = y_norm[:,1]
vrot_norm = y_norm[:,2]
meta_norm = y_norm[:,3]

F = h5py.File(folder_path+norm_dataset,'w')                  #creating the standarized shuffled datasets
F.create_dataset('X', (num_example,num_flux), data=X)
F.create_dataset('TEMP', (num_example,),  data=temp_norm)
F.create_dataset('LOGG', (num_example,),  data=logg_norm)
F.create_dataset('VROT', (num_example,),  data=vrot_norm)
F.create_dataset('META', (num_example,),  data=meta_norm)
F.create_dataset('moments', (num_labels,2), data=norm)
F.close()
