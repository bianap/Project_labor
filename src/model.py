import random

import numpy as np
import pandas as pd
from keras import Input
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import BatchNormalization, Activation
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

INPUT_PATH = "/media/bianap/Data/Suli/Egyetem/2018-2019-1/Témalabor/building_facade/Project_labor/Images/input_edited/"
OUTPUT_PATH = "/media/bianap/Data/Suli/Egyetem/2018-2019-1/Témalabor/building_facade/Project_labor/Images/output_edited/"

train_split = 0.5
valid_split = 0.2
test_split = 0.3

entry = pd.read_csv("data.csv")
sample_num = 50 #len(entry)

train_size = train_split * sample_num
valid_size = valid_split * sample_num
test_size = test_split * sample_num

# Initialize the set inputs and outputs
samples = np.zeros(sample_num, dtype=[('input', float, (256, 256, 3)), ('output', float, (256, 256, 3))])

for row in range(0, sample_num):
    samples[row] = imread(INPUT_PATH + entry['input'][row])/255, imread(OUTPUT_PATH + entry['output'][row])

train = samples[0:int(sample_num*(1-valid_split-test_split))]
valid = samples[int(sample_num*(1-valid_split-test_split)):int(len(entry)*(1-test_split))]
test = samples[int(sample_num*(1-test_split)):]

train_x = np.reshape(train['input'], (len(train), 256, 256, 3))
valid_x = np.reshape(valid['input'], (len(valid), 256, 256, 3))
test_x = np.reshape(test['input'], (len(test), 256, 256, 3))

seed = 42
random.seed = seed
np.random.seed = seed

###############################
#           Model             #
###############################
class TrainingHistory(Callback):

    # Initializes empty lists for metric storing
    def on_train_begin(self, logs={}):
        # Error on training data
        self.losses = []
        # Error on validation data
        self.valid_losses = []
        # Stores how good is the model (on training data)
        self.accs = []
        # Stores how good is the model (on validation data)
        self.valid_accs = []
        # Number of epochs
        self.epoch = 0

    # At the end of an epoch, save the performance of the actual network
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.valid_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('acc'))
        self.valid_accs.append(logs.get('val_acc'))
        self.epoch += 1


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_normal",
               padding="same", input_shape=np.shape(input_tensor))(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=4, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u4 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    u4 = Dropout(dropout)(u4)
    c4 = conv2d_block(u4, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u5 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    u5 = Dropout(dropout)(u5)
    c5 = conv2d_block(u5, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

################################
#    Training & testing        #
################################

history = TrainingHistory()
input_img = Input((256, 256, 3), name='img')
model = get_unet(input_img)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Configure early stopping
patience = 5
early_stopping = EarlyStopping(patience=patience, verbose=1)

# Saves the best model (using validation error)
checkpointer = ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)

# Reduces learning rate automatically
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=10e-5)


model.fit(train_x, train['output'],
          # Size of our batch
          batch_size=20,
          # Number of epochs
          nb_epoch=50,
          # Verbose parameter
          verbose=1,
          # Validation runs in parallel with training
          validation_data=(valid_x, valid['output']),
          # Save important metrics in 'history'
          callbacks=[reduce_lr, checkpointer, early_stopping, history],
          # Shuffle input data
          shuffle=True)

plt.figure(figsize=(10, 5))
plt.title('Measure of error')
plt.plot(np.arange(history.epoch), history.losses, color ='g', label='Measure of error on training data')
plt.plot(np.arange(history.epoch), history.valid_losses, color ='b', label='sure of error on validation data')
plt.legend(loc='upper right')
plt.xlabel('Number of training iterations')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Load the best model
model = load_model('weights.hdf5')

# Predicating with test data
preds = model.predict(test_x)

# Calculating the error on test data
test_mse = mean_squared_error(test['output'], preds)
print("Test MSE: %f" % (test_mse))
model.summary()

for image in range(0, sample_num):
    imsave("Images/results/str(image).jpg", preds[image])