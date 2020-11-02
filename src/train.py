'''
KERAS is a neural networks application program interface (API) writtern in python and running on top of TensorFlow. The network
is composed of two 1-dimensional convolutional layers, 1-dimensional maxpooling layers, and two dense layers linearily stacked
in Sequential mode. The model is optimized by the first-order adaptive gradient-based optimization scheme ADAM using mean-squared error
cost funcrtion equipped with EarlyStopping callbacks
'''
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

folder_path = ''
train_dataset = 'n_s_training_dataset.hdf5'

def load_batch(data_file, num_objects, batch_size, indx):     #load the desired batches from the h5py datasets

    indices = random.sample(range(indx, indx+num_objects), batch_size)
    indices = np.sort(indices)

    F = h5py.File(data_file, 'r')
    X = F['X']
    temp = F['TEMP']
    logg = F['LOGG']
    vrot = F['VROT']
    meta = F['META']

    X = X[indices,:]
    X = X.reshape(len(X), X.shape[1], 1)
    y = np.column_stack((temp[:][indices],
                         logg[:][indices],
                         vrot[:][indices],
                         meta[:][indices]))


    return X, y

def generate_train_batch(data_file, num_objects, batch_size, indx): #generates training batches for ADAM

    while True:
        x_batch, y_batch = load_batch(data_file,
                                      num_objects,
                                      batch_size,
                                      indx)
        yield (x_batch, y_batch)

def generate_cv_batch(data_file, num_objects, batch_size, indx):  #generates validation batches for ADAM

    while True:
        x_batch, y_batch = load_batch(data_file,
                                      num_objects,
                                      batch_size,
                                      indx)
        yield (x_batch, y_batch)

with h5py.File(train_dataset, 'r') as F:
    spectra  = F['X']                                   #load the shuffled spectra
    num_flux = spectra.shape[1]                         #number of flux point for each spectrum
    num_labels = 4                                      #load the TEMP, LOGG, VROT, META datasets
    num_train  = int(np.floor(spectra.shape[0]*0.6))    #60% of entire dataset
    num_cv     = int(np.floor(spectra.shape[0]*0.2))    #20% of entire dataset

activation  = 'relu'                           #rectified Linear Unit (ReLu) activation function
initializer = 'he_normal'                      #He parameter initialization
input_shape = (None, num_flux, 1)              #vector input shape for nn
num_filters = [4, 16]                          #number of filters for convolutional layers
num_hidden  = [256, 128]                       #number of nodes for dense layers
filter_length = 8                              #filter dimension for convolutional layers
pool_length   = 4                              #pool dimension for maxpooling layers

model = Sequential([                           #construction of Neural Network in sequential mode

          InputLayer(batch_input_shape=input_shape),
          Conv1D(filters=num_filters[0], kernel_size=filter_length, padding="same",
                 activation=activation, kernel_initializer=initializer),
          Conv1D(filters=num_filters[1], kernel_size=filter_length, padding="same",
                 activation=activation, kernel_initializer=initializer),
          MaxPooling1D(pool_size=pool_length),
          Flatten(),   #flatten the feature map to a vector so that it could be inputed to the fully-connected layers
          Dense(units=num_hidden[0], activation=activation, kernel_initializer=initializer),
          Dense(units=num_hidden[1], activation=activation, kernel_initializer=initializer),
          Dense(units=num_labels, activation='linear', input_dim=num_hidden[1]), #linear activation in this layer to perform regression

          ])

lr = 0.0007                                     #initial learning rate for ADAM
beta_1 = 0.9                                    #decay parameter for estimator (default value)
beta_2 = 0.999                                  #decay parameter fot estimator (default value)
optimizer_epsilon = 1e-08                       #smoothing constant to avoid dividing by zero

optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0, amsgrad=True) #ADAM

early_stopping_min_delta = 0.0001               #minimum change to qualify as improvement
early_stopping_patience  = 6                    #number of epochs with no improvement to stop training

early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                               patience=early_stopping_patience, verbose=1, mode='min') #EARLY_STOP

loss_function = 'mean_squared_error'             #mean squared error cost function
metrics = ['accuracy']                           #tracking accuracy and mean error

model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics) #COMPILE

batch_size = 64                                  #batch size for each iteration (num_train/batch_size should be an int)
max_epochs = 22                                  #maximum number of epochs before stoping training

'''
The below line is the core of this code as it is where the fitting (training) happens. This function calls both the generate_train_batch
and generate_cv_batch functions and use them for training and validation respectively. As well as stores the history of the callbacks
to check for early_stop conditions and stores the history of the training throuhgout the entire training phase.
'''
history = model.fit_generator(generate_train_batch(folder_path+train_dataset, num_train, batch_size, 0),
                              verbose=1,
                              validation_data=generate_cv_batch(folder_path+train_dataset, num_cv, batch_size, num_train),
                              steps_per_epoch=num_train/batch_size,
                              epochs=max_epochs,
                              callbacks=[early_stopping],
                              max_queue_size=10, validation_steps=num_cv/batch_size)

neural_network = 'nn_model.h5'
model.save(neural_network)                      #save the parameter of the model to be used for testing
print(neural_network + ' saved.')

plt.plot(history.history['loss'])                #plot the model's training loss
plt.plot(history.history['val_loss'])            #plot the model's validation loss
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
