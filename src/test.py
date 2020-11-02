'''
After training is done and the model's wieghts are saved, the network can be thus used to predict on new datasets. This code
applies the converged model to new data. The testing dataset is noised using gaussian noise prior to test.
'''
import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras
import random
import pylab as p
from pylab import savefig

train_dataset = 'n_s_training_dataset.hdf5'
F = h5py.File(train_dataset, 'r')          #loading the dataset and the labels
spectra = F['X']
labels  = np.column_stack((F['TEMP'][:],
                           F['LOGG'][:],
                           F['VROT'][:],
                           F['META'][:]))
norm = F['moments']

tot_example  = spectra.shape[0]                         #total number of examples
num_train    = int(np.floor(spectra.shape[0]*0.6))      #number used for training
num_cv       = int(np.floor(spectra.shape[0]*0.2))      #number used for validation
num_test     = int(np.floor(spectra.shape[0]*0.2))      #number used for testing
num_labels   = labels.shape[1]                          #number of labels
batch_size   = 64                                       #batch size for testing

def noise(X, noise_min, noise_max):
    n = np.zeros((X.shape[0],))
    X_n = np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        s_n = np.squeeze(random.sample(range(noise_min,noise_max),1))+0.01    #get random S/N ratios from the desired interval of noise
        n[i] = s_n
        mu_s = np.mean(X[i,:])                                                #given that the spectra dont have zero mean, the mean is calculated to ge the variance of the gaussian
        std_n = mu_s/s_n                                                      #corresponding standard deviation to the S/N ratio
        noise = np.random.normal(0,std_n,(X.shape[1],))                       #define the corresponding guassian for the noise
        X_n[i,:] = X[i,:] + noise
    print n
    return X_n, np.squeeze(n)

def load_test_dataset(spectra):                         #choose the remaining spectra
    indices = random.sample(range(num_train+num_cv, tot_example), num_test) #chossing the rest spectra
    indices = np.sort(indices)
    X = spectra[indices, :]                            #reshape the spectra to be used on the network
    return X, indices

def denormalize(file_name):                                          #denormalizing the labels to get physical units for the prediction
    for i in range(num_labels):
        for j in range(num_test):
            file_name[j][i] = file_name[j][i]*(norm[i][1]+1e-011) + norm[i][0]
    return file_name

X, indices = load_test_dataset(spectra)                       #loading the testing dataset
labels_test = denormalize(labels[indices, :])                 #loading the correct labels after un-standarization
model = keras.models.load_model('model0.h5')                   #loading the converged model
names = ['T_eff', 'logg', 'vsini', 'meta']                    #esthetics

for i in range(3):
    noise_min = [20,100,200]                                  #three intervals for noise (20-100, 100-200, 200-400)
    noise_max = [100,200,400]
    X_n, n = noise(X, noise_min[i], noise_max[i])             #add noise to the spectra
    X_n = X_n.reshape(len(X), X.shape[1],1)                   #reshape the input data to be inputed to the network
    test_predictions = denormalize(model.predict(X_n, batch_size=batch_size, verbose=1)) #get the prediction of the model
    std = [0,0,0,0]
    for j in range(num_labels):                               #a loop to calculate the RMSE for each label class
        diff=0
        for k in np.arange(num_test):
            diff = diff + (test_predictions[k,j]-labels_test[k,j])**2
        std[j] = ((diff)/num_cv)**0.5
    plt.figure(figsize=(12,8))
    for l in range(num_labels):                       #PLOT
        plt.subplot(2,2,l+1)
        r = labels_test[:,l]-test_predictions[:,l]
        s = p.scatter(labels_test[:,l],test_predictions[:,l],c=n,s=30,cmap='inferno',label=str('RMSE= '+ str(std[l])))
        cb = p.colorbar(s)
        cb.ax.set_ylabel('S/N')
        p.plot(labels_test[:,l],labels_test[:,l],'r')
        p.legend(loc='best')
        p.ylabel('Predicted '+names[l])
        p.xlabel('Synthetic '+names[l])
        p.tight_layout()
    savefig('Noisy Prediciton '+str(noise_min[i])+' to '+str(noise_max[i])) #save in png file
