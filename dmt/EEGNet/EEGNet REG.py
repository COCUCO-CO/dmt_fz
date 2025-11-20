
#%% EEGNet


"""
 ARL_EEGModels - A collection of Convolutional Neural Network models for EEG
 Signal Processing and Classification, using Keras and Tensorflow
 Requirements:
    (1) tensorflow == 2.X (as of this writing, 2.0 - 2.3 have been verified
        as working)

 To run the EEG/MEG ERP classification sample script, you will also need
    (4) mne >= 0.17.1
    (5) PyRiemann >= 0.2.5
    (6) scikit-learn >= 0.20.1
    (7) matplotlib >= 2.2.3

 To use:

    (1) Place this file in the PYTHONPATH variable in your IDE (i.e.: Spyder)
    (2) Import the model as

        from EEGModels import EEGNet

        model = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)

    (3) Then compile and fit the model

        model.compile(loss = ..., optimizer = ..., metrics = ...)
        fitted    = model.fit(...)
        predicted = model.predict(...)
 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those
 portions are released world-wide under the terms of the Creative Commons Zero
 1.0 (CC0) license.

 Other portions of this project are subject to domestic copyright protection
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0
 license.  The complete text of the license governing this material is in
 the file labeled LICENSE.TXT that is a part of this project's official
 distribution.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm


def EEGNet(Chans = 64, Samples = 128,
           dropoutRate = 0.5, kernLength = 64, F1 = 8,
           D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
    advised to do some model searching to get optimal performance on your
    particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)

    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)

    flatten      = Flatten(name = 'flatten')(block2)

    dense        = Dense(1, name = 'dense',
                         kernel_constraint = max_norm(norm_rate))(flatten)
    
    sigmoid      = Activation('sigmoid', name = 'sigmoid')(dense)

    return Model(inputs=input1, outputs=sigmoid)


#%%

import pandas as pd

folder = "E:\\FedeZ\\DMT\\EEGNet\\"
targets = pd.read_csv(folder+"targets.csv", index_col=0)
labels = targets.columns.tolist()

#%%
import os
from tqdm import tqdm
from mne.io import read_epochs_eeglab
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

folder = "E:\\FedeZ\\DMT\\EEG\\Clean EEGLAB\\"
scaler = MinMaxScaler()

def open_eeg_files(cond):
    files_list = sorted([x for x in os.listdir(folder+cond) if x[-3:]=="set"])
    data_list = [read_epochs_eeglab(folder+cond+"\\"+x, verbose=0) for x in tqdm(files_list)]
    conc_x = data_list[0].get_data(copy=False)
    groups = [0 for x in range(conc_x.shape[0])]
    conc_y = np.repeat(targets.iloc[0,:].values.reshape(1,-1), repeats=conc_x.shape[0], axis=0)
    for i in range(1,len(data_list)):
        data_x = data_list[i].get_data(copy=False)
        data_y = np.repeat(targets.iloc[i,:].values.reshape(1,-1), repeats=data_x.shape[0], axis=0)
        groups += [i for x in range(data_x.shape[0])]
        conc_x = np.concatenate((conc_x, data_x), axis=0)
        conc_y = np.concatenate((conc_y, data_y), axis=0)
    print(np.shape(conc_x), np.shape(conc_y))
    return conc_x, conc_y, np.array(groups)

dmt_epochs, y_raw, groups = open_eeg_files("DMT")
y = scaler.fit_transform(y_raw)
#closed_epochs, groups = open_eeg_files("EC") # if fname not in ["S07_EC_ICA_pruned.set"]]
#open_epochs, groups = open_eeg_files("EO")
    
#%%

kernels, channels, samples = 1, 24, 1000
X = dmt_epochs.reshape(dmt_epochs.shape[0], channels, samples, kernels)

#%%
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=420)

for (train_index, valtest_index) in splitter.split(X, y, groups):
    X_train = X[train_index]
    y_train = y[train_index]
    train_groups = groups[train_index]
    
    X_valtest = X[valtest_index]
    y_valtest = y[valtest_index]
    valtest_groups = groups[valtest_index]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=420)

for (test_index, val_index) in splitter.split(X_valtest, y_valtest, valtest_groups):
    X_test = X_valtest[test_index]
    y_test = y_valtest[test_index]
    test_groups = valtest_groups[test_index]

    X_val = X_valtest[val_index]
    y_val = y_valtest[val_index]
    val_groups = valtest_groups[val_index]

#%%

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

# Define custom R2 metric function
def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

#%%

import json

folder = "E:\\FedeZ\\DMT\\EEGNet\\models\\"
score_history = pd.DataFrame([], columns=["Scores"])
score_history.to_csv(folder+"score_history.csv")

#%%

def training(i):
    print(labels[i])

    model = EEGNet(Chans=channels, Samples=samples,
                   kernLength=32, F1=8, D=2, F2=16,
                   dropoutRate=0.5, dropoutType='Dropout')  
    
    model.compile(loss="mean_squared_error", optimizer='adam', metrics = [r2_score])
    
    filepath = folder+labels[i]+".h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    
    fittedModel = model.fit(X_train, y_train[:,i], batch_size=16, epochs=1000,
                            verbose=2, validation_data=(X_val, y_val[:,i]),
                            callbacks=[checkpointer, earlystopping])
    
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred).numpy()
    
    score_history = pd.read_csv(folder+"score_history.csv")
    new_score = pd.DataFrame([score], columns=["Scores"], index=[labels[i]])
    pd.concat([score_history, new_score]).to_csv(folder+"score_history.csv")
    
    history_dict = fittedModel.history
    with open(folder+labels[i]+".json", 'w') as json_file:
        json.dump(history_dict, json_file)
    
    print(score+"\n")

#%%

from multiprocessing import Process

if __name__ == '__main__':
    for i in range(23):
        p = Process(target=training, args=(i,))
        p.start()
        p.join()
    
