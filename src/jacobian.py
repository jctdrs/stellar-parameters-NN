import numpy as np
from keras.models import load_model
import h5py
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import random
from pylab import savefig
import time


train_dataset = 'n_s_training_dataset.hdf5'
F = h5py.File(train_dataset, 'r')                       #loading the dataset and the labels
spectra = F['X']
teff    = F['TEMP']
norm    = F['moments']

tot_example  = spectra.shape[0]
num_train    = int(np.floor(spectra.shape[0]*0.6))
num_cv       = int(np.floor(spectra.shape[0]*0.2))


lamb  = np.arange(4450,4990,0.05)
width = 5
chosen_teff = []
line = []

model = load_model('model0.h5')

def denormalize(file_name):
    transfer_vector = []
    for i in range(len(file_name)):
        transfer_vector = np.append(transfer_vector, file_name[i]*(norm[0][1]+1e-011) + norm[0][0])
    return transfer_vector

def load_dataset():
    idx = []
    for i in range(num_train+num_cv, tot_example):
        if (np.int(de_teff[i])==np.int(de_teff[k])):
               idx = np.append(idx, i)
    idx = np.sort(idx)
    X = spectra[idx, :]
    return X

def jacobian(input, model):
    input = input.reshape(input.shape[0],input.shape[1],1)
    y_list = tf.unstack(model.output[0])
    J = [tf.gradients(y_list[0], model.input)]
    jacobian_func = [K.function([model.input, K.learning_phase()], j_) for j_ in J]
    jacobian = np.array([jf([input,False]) for jf in jacobian_func])[:,:,0,:,0]
    del y_list
    del input
    del J
    del jacobian_func
    return jacobian

de_teff = denormalize(teff)
teff_set = np.sort(list(set(de_teff)))

mu   = 4860
sig  = width
x    = np.linspace(4450,4990,10800)
weight = np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2.)))
'''

line = [0.00233396557245,0.0022772237642,0.00240188484735,0.00259884960193,
        0.00324806315493,0.00317119789852,0.00326037944146,0.00423455102453,
        0.00365769786212,0.00358023605192,0.0038481799939,0.0041840125888,
        0.00410242950056,0.00408633098992,0.00392770616736,0.00385010748706,
        0.00350359588682,0.0036149554483,0.00338500030994,0.00338540268221,
        0.00330426853322]
'''
for k in range(len(teff_set)):

 start = time.time()

 jac = 0
 balmer = 0
 X = load_dataset()
 for i in range(5):
     input = X[i].reshape((1, X.shape[1], 1))
     jac = jac + jacobian(input, model)
 jac = jac/5
 for j in range(2*width):
  balmer = balmer + np.power(np.squeeze(jac[0])[(mu-width+j-4450)*20],2)*weight[(mu-width+j-4450)*20]
 line = np.append(line,balmer)
 chosen_teff = np.append(chosen_teff,teff_set[k])
 del jac
 del balmer
 '''
 fig, ax = plt.subplots(1, 1, figsize=(14, 5))
 ax.set_xlabel("Wavelength A")
 ax.set_ylabel("Average Derivative w.r.t T_eff")
 plt.xticks(np.arange(min(lamb), max(lamb)+1, 20))
 plt.xlim(xmin=4440, xmax=5000)
 fig.tight_layout()
 plt.plot(lamb, np.squeeze(jac[0])*np.squeeze(jac[0]))
 '''
 end = time.time()
 print (end - start), X.shape[0], k
 del X
plt.plot(chosen_teff, line)
plt.show()
