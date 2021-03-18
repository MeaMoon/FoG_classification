# Libraries
import tensorflow as tf
import numpy as np
import pandas as pd

# Loading the data set with no given labeled column
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
                    'z-three']
    frame = pd.read_csv(file_path,
                     header=None, 
                     delim_whitespace=True,
                     names=column_names)

    return frame

# In this situation the time column is required to segment the 256 rows of data in 64 sized frames
try_on_that = load_the_data('patients/S02R02_FoG.txt')

# Data normalization
def normal(df):
    mu = np.mean(df,axis = 0)
    sigma = np.std(df,axis = 0)
    return (df - mu)/sigma

try_on_that['x-one'] = normal(try_on_that['x-one'])
try_on_that['y-one'] = normal(try_on_that['y-one'])
try_on_that['z-one'] = normal(try_on_that['z-one'])

try_on_that['x-two'] = normal(try_on_that['x-two'])
try_on_that['y-two'] = normal(try_on_that['y-two'])
try_on_that['z-two'] = normal(try_on_that['z-two'])

try_on_that['x-three'] = normal(try_on_that['x-three'])
try_on_that['y-three'] = normal(try_on_that['y-three'])
try_on_that['z-three'] = normal(try_on_that['z-three'])

# Data segmentation

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data,window_size = 64):
    segments = np.empty((0,window_size,9))
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
    return segments

x_modified = segment_signal(try_on_that)

new_model = tf.keras.models.load_model('model_cnn_64.h5')
new_model.summary()

pred = new_model.predict([x_modified,x_modified,x_modified])

Y_predicted = np.around(pred)
np.savetxt("new_predicted_labels_64.txt", Y_predicted)