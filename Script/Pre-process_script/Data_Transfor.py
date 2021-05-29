import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy import io


#------------------------read npy to mat-------------------------------


# keys = np.load('keylist_3.npy')
# io.savemat('keys_3.mat', {'keys_3': keys})


traces = np.load('all__0.npy')
io.savemat('traces_first.mat', {'traces_first': traces})





#----------------------------------------------------------------------



