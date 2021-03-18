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

# Dropping the time column since this file only has 256 rows (1 frame)
try_on_that = load_the_data('patients/S02R02_FoG.txt')
try_on_that = try_on_that.drop(['time'], axis = 1)

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

# Modifying the data into required shape
x_modified = try_on_that.values.reshape(1,256,9).astype('float32')

# Reviewing the model one more time
new_model = tf.keras.models.load_model('model_cnn.h5')
new_model.summary()

# Making and saving the prediction
pred = new_model.predict([x_modified,x_modified,x_modified])
Y_predicted = np.around(pred)
np.savetxt("new_predicted_labels.txt", Y_predicted)