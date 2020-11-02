'''
This script will load the dataset from dumpfile and then convert it to hdf5
'''
import numpy as np
import h5py
import pickle

dataset_2 = 'training_dataset_2.hdf5'
dataset_3 = 'training_dataset_3.hdf5'
h5py_dataset = 'training_dataset2.hdf5'

F2 = h5py.File(dataset_2,'r')
spectra_2 = F2['X']
temp_2 = F2['TEMP']
logg_2 = F2['LOGG']
vrot_2 = F2['VROT']
meta_2 = F2['META']
indices = []

F3 = h5py.File(dataset_3,'r')
spectra_3 = F3['X']
temp_3 = F3['TEMP']
logg_3 = F3['LOGG']
vrot_3 = F3['VROT']
meta_3 = F3['META']

print spectra_2.shape[0]+spectra_3.shape[0]

for i in range(spectra_2.shape[0]):
    if meta_2[i]!=0:
        indices.append(i)

meta_2 = F2['META'][indices]
spectra_2 = F2['X'][indices]
temp_2 = F2['TEMP'][indices]
logg_2 = F2['LOGG'][indices]
vrot_2 = F2['VROT'][indices]
meta_2 = F2['META'][indices]

print spectra_2.shape[0]+spectra_3.shape[0]
def conc(data_2, data_3):
    data= []
    for i in range(data_2.shape[0]):
        data.append(list(data_2[:])[i])
    for i in range(data_3.shape[0]):
        data.append(list(data_3[:])[i])
    return data

temp = conc(temp_2,temp_3)
print 'h'
logg = conc(logg_2,logg_3)
print 'h'
vrot = conc(vrot_2,vrot_3)
print 'h'
meta = conc(meta_2,meta_3)
print 'h'

print len(meta), len(temp), len(logg), len(vrot)
spectra = np.row_stack((spectra_2,spectra_3))
print 'h'

num_example =  spectra_2.shape[0]+spectra_3.shape[0]   #number of example
num_labels  = 4                                        #number of labels
num_flux    = 10800                                    #number of flux points

F = h5py.File(h5py_dataset,'w')
F.create_dataset('X', (num_example,num_flux), data=spectra)
F.create_dataset('TEMP', (num_example,),  data=temp)
F.create_dataset('LOGG', (num_example,),  data=logg)
F.create_dataset('VROT', (num_example,),  data=vrot)
F.create_dataset('META', (num_example,),  data=meta)
F.close()
