# Libraries
from __future__ import print_function
from keras.layers import Input, Dense, MaxPooling1D, Conv1D
from keras.models import Model
import pandas as pd 
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
from scipy import stats
from sklearn.metrics import classification_report
from keras.layers import Dropout, Flatten
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import os
from keras.utils import plot_model
# %%
# Loading the data from given .txt
# Here is dropped the activity 0, which represents the state in which
# sensors were only being put on (and off respectively)
def load_the_data(file_path):

    column_names = ['time',
                    'x-one',
                    'y-one',
                    'z-one',
                    'x-two',
                    'y-two',
                    'z-two',
                    'x-three',
                    'y-three',
                    'z-three',
                    'activity']
    frame = pd.read_csv(file_path,
                     header=None, 
                     delim_whitespace=True,
                     names=column_names)
    
    frame_2 = frame[frame['activity'] != 0]
    return frame_2

# Loading the second patient for training
train = load_the_data('patients/S02R02.txt')

# Loading the second patient for testing
test = load_the_data('patients/S02R01.txt')
# %%
# Data normalization
def normalize_train(train):
    mu = np.mean(train,axis = 0)
    sigma = np.std(train,axis = 0)
    return (train - mu)/sigma

def normalize_test(test):
    mu = np.mean(test,axis = 0)
    sigma = np.std(test,axis = 0)
    return (test - mu)/sigma

# Each column is being normalized separately in order to awoid ruining the structure of the given data set
train['x-one'] = normalize_train(train['x-one'])
train['y-one'] = normalize_train(train['y-one'])
train['z-one'] = normalize_train(train['z-one'])

train['x-two'] = normalize_train(train['x-two'])
train['y-two'] = normalize_train(train['y-two'])
train['z-two'] = normalize_train(train['z-two'])

train['x-three'] = normalize_train(train['x-three'])
train['y-three'] = normalize_train(train['y-three'])
train['z-three'] = normalize_train(train['z-three'])

test['x-one'] = normalize_test(test['x-one'])
test['y-one'] = normalize_test(test['y-one'])
test['z-one'] = normalize_test(test['z-one'])

test['x-two'] = normalize_test(test['x-two'])
test['y-two'] = normalize_test(test['y-two'])
test['z-two'] = normalize_test(test['z-two'])

test['x-three'] = normalize_test(test['x-three'])
test['y-three'] = normalize_test(test['y-three'])
test['z-three'] = normalize_test(test['z-three'])
# %%
# Data segmentation
# Data are being segmented into frames which include 256 rows, thus reducing the overall size of the data set
# This is the most important part which significantly boosts the models performance
def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data,window_size = 256):
    segments = np.empty((0,window_size,9))
    labels = np.empty((0))
    for (start, end) in windows(data["time"], window_size):
        x_1 = data["x-one"][start:end]
        y_1 = data["y-one"][start:end]
        z_1 = data["z-one"][start:end]
        x_2 = data["x-two"][start:end]
        y_2 = data["y-two"][start:end]
        z_2 = data["z-two"][start:end]
        x_3 = data["x-three"][start:end]
        y_3 = data["y-three"][start:end]
        z_3 = data["z-three"][start:end]
        if(len(data["time"][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3])])
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])
    return segments, labels

# The modified version now includes 3 dimensions (rows, frame size, features)
# Which is being saved to the disk as well
X_train, Y_train = segment_signal(train)
Y_train = np.asarray(pd.get_dummies(Y_train), dtype = np.float32)
train_reshaped = X_train.reshape(len(X_train),256, 9).astype('float32')
with open('train_reshaped.txt', 'w') as outfile_train:
    for slice_2d_train in train_reshaped:
        np.savetxt(outfile_train, slice_2d_train)
print("Saved the reshaped train to disk")

# Same thing for the test part
X_test, Y_test = segment_signal(test)
Y_test = np.asarray(pd.get_dummies(Y_test), dtype = np.float32)
test_reshaped = X_test.reshape(len(X_test),256, 9).astype('float32')
with open('test_reshaped.txt', 'w') as outfile_test:
    for slice_2d_test in test_reshaped:
        np.savetxt(outfile_test, slice_2d_test)
print("Saved the reshaped test to disk")
# %%
# Modeling
# This is a CNN1D model, which consists out of three independant heads, 
# which are then being merged into a single variable
# Judging by the structure of the given head part it includes:
# 1. input variable which requires the frame size and number of features
# 2. convolutional layer with a respective kernel_size
# 3. dense layer to increase the amount of training parameters
# 4. pooling layer to avoid overlapping
# 5. flattening layer to decrease the dimensions from 3 to 2 in order to merge these heads together    
# After merging the dropout layer is used to drop even more irrelevant parameters to improve the learning rate
# Then follows the batchnormalization and an another dense layer (which has much higher value this time)
# Softmax is used here, because the are 2 different values which this neural network must differ/classify
# At last the SGD is used to optimize the learning procedure
verbose, epochs, batch_size = 1, 25, 128
n_timesteps, n_features, n_outputs = train_reshaped.shape[1], train_reshaped.shape[2], Y_train.shape[1]
# Head 1
inputs1 = Input(shape=(n_timesteps,n_features))
conv1 = Conv1D(filters=50, kernel_size=(33))(inputs1)
dense1 = Dense(50, activation='relu')(conv1)
pool1 = MaxPooling1D(pool_size=2)(dense1)
flat1 = Flatten()(pool1)
# Head 2
inputs2 = Input(shape=(n_timesteps,n_features))
conv2 = Conv1D(filters=40, kernel_size=(36))(inputs2)
dense2 = Dense(40, activation='relu')(conv2)
pool2 = MaxPooling1D(pool_size=2)(dense2)
flat2 = Flatten()(pool2)
# Head 3
inputs3 = Input(shape=(n_timesteps,n_features))
conv3 = Conv1D(filters=20, kernel_size=(37))(inputs3)
dense3 = Dense(20, activation='relu')(conv3)
pool3 = MaxPooling1D(pool_size=2)(dense3)
flat3 = Flatten()(pool3)
# Merging
merged = concatenate([flat1, flat2, flat3])
# Interpretation
drop1 = Dropout(0.5)(merged)
batch1 = BatchNormalization()(drop1)
dense1 = Dense(100, activation='relu')(dense1)
outputs = Dense(n_outputs, activation='softmax')(batch1)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# Optimizing
opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())
# %%
# Callbacks
# Before fitting the created model callbacks are being set
# This includes the checkpoint and ReduceLPOnPlateu parameters
# These two are the key saviours which keep this network from overlapping and making false predictions
checkpoint_path = "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', save_weights_only=True, verbose=1, monitor = 'val_loss', mode = 'min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.001)
callbacks_list = [cp_callback,reduce_lr]
# %%
# Fit model
# The model is being trained 3 times which includes 25 epochs for each loop
for x in range(5):
    model.fit([train_reshaped,train_reshaped,train_reshaped], Y_train, epochs=epochs, validation_data=([test_reshaped,test_reshaped,test_reshaped], Y_test), batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)
# %%
# Model evaluation
acc = model.evaluate([test_reshaped,test_reshaped,test_reshaped], Y_test, batch_size=batch_size, verbose=verbose)
pred = model.predict([test_reshaped,test_reshaped,test_reshaped])
# %%
# Metrics & CM
# The predicted values are being saved as well
# Four most relevant metrics are being calulated from the Confussion Matrix
Y_oracle = np.around(pred)
np.savetxt("predicted_labels.txt", Y_oracle)
precision, recall, fscore, support = score(Y_test, Y_oracle)
results = multilabel_confusion_matrix(Y_test, Y_oracle) 
print(results)
accuracy_score(Y_test, Y_oracle) 
classification_report(Y_test, Y_oracle)
# %%
# Plotting and saving the model
# After successful training attempts the model is being saved in .png and .h5 format
# The .h5 format model can be used elsewhere, since it contains everything needed for prediction
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.save("model_cnn.h5")
print("Saved model to disk")