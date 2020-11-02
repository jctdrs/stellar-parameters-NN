'''
This code explores the created datasets just so to see the distribution of the
labels in the label space. To check the spread, mean, distribution etc of the data.
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig

folder_path = ''
train_dataset = 'training_dataset.hdf5'

F = h5py.File(folder_path+train_dataset,'r')

temp = F['TEMP']
logg = F['LOGG']
vrot = F['VROT']
meta = F['META']

y = np.column_stack((temp,logg,vrot,meta))
names = ['teff','logg','vrot','meta']

for i in range(y.shape[1]):
    plt.subplot(4,1,i+1)
    plt.hist(y[:,i], 300)  #check the frequency of each label value (distribution of the label space)
    plt.xlabel(names[i])
    plt.tight_layout()
savefig('Label Grid')    #create a summary of the entire dataset in a png file
