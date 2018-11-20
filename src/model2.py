import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras import Input
from keras.optimizers import Adam
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Activation, Dense, Flatten, regularizers
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

INPUT_PATH = cfg.INPUT_PATH
OUTPUT_PATH = cfg.OUTPUT_PATH



entry = pd.read_csv("data.csv")
sample_num = len(entry)

train_size = sample_num


# Initialize the set inputs and outputs
samples = np.zeros(sample_num, dtype=[('input', float, (256, 256, 3)), ('output', float, (256, 256, 3))])

for row in range(0, sample_num):
    samples[row] = imread(INPUT_PATH + entry['input'][row]) / 255, imread(OUTPUT_PATH + entry['output'][row])

train_set = samples[:]

train_x = train_set['input']
train_y = train_set['output']

shp = train_x.shape[1:]
dropout = 0.25
n_filters = 4
batchnorm = True
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)


def conv2d_block(input_tensor, n_filters=4, kernel_size=3, batchnorm=True, lamda_l1 = 0.001):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_normal",
               padding="same", kernel_regularizer=regularizers.l1(lamda_l1))(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_normal",
               padding="same", kernel_regularizer=regularizers.l1(lamda_l1))(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

#Generative model
def generator(n_filters=4, dropout=0.5, batchnorm=True):
    input = Input((256, 256, 3), name='img')
    # contracting path
    c1 = conv2d_block(input_tensor=input, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)
    #out: 128x128x16

    c2 = conv2d_block(p1, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    #out: 64x64x32
    c3 = conv2d_block(p2, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    #out: 64x64x64

    # expansive path
    u4 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    u4 = Dropout(dropout)(u4)
    c4 = conv2d_block(u4, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u5 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    u5 = Dropout(dropout)(u5)
    c5 = conv2d_block(u5, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c5)
    model = Model(input, outputs)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    model.summary()

    return model

def discriminator(n_filters=4, dropout=0.5, batchnorm=True):
    input = Input((256, 256, 3), name='img2')
    c1 = conv2d_block(input_tensor=input, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)
    # out: 128x128x16
    c2 = conv2d_block(p1, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    # out: 64x64x32
    c3 = conv2d_block(p2, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    # out: 64x64x64
    flatten = Flatten()(c3)
    outputs = Dense(2, input_shape=np.shape(c3), activation='softmax')(flatten)
    model = Model(input, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=dopt)
    return model

gen = generator()
discr = discriminator()

# Freeze weights in the discriminator for stacked training
make_trainable(discr, False)

# Build stacked GAN model
gan_input = Input((256, 256, 3))
generated_img = gen(gan_input)
discr_out = discr(generated_img)
GAN = Model(gan_input, discr_out)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()


def plot_loss(losses):
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.savefig('11-20-18:30' + ".png")

# Pre-train the discriminator network ...

generated_images = gen.predict(train_x[:50])
X = np.concatenate((train_y[:50], generated_images))
#n = train_y.shape[0]
n =50
y = np.zeros([2*n,2])
y[:n, 1] = 1
y[n:, 0] = 1

make_trainable(discr,True)
discr.fit(X,y, epochs=1, batch_size=10)
y_hat = discr.predict(X)

y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print ("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))

# set up loss storage vector
losses = {"d": [], "g": []}


def train_for_n(nb_epoch=25, plt_frq=5, BATCH_SIZE=5):
    for e in tqdm(range(nb_epoch)):

        # Make generative images
        random_samples = np.random.randint(0, train_y.shape[0], size=BATCH_SIZE)
        real_image_inBatch = train_y[random_samples, :, :, :]
        semantic_images_inBatch = train_x[random_samples, :, :, :]
        generated_images = gen.predict(semantic_images_inBatch)

        # Train discriminator on generated images
        x = np.concatenate((real_image_inBatch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        x, y = shuffle(x, y)
        make_trainable(discr, True)
        d_loss = discr.train_on_batch(x, y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on semantic images to non-generated output class
        random_samples = np.random.randint(0, train_x.shape[0], size=BATCH_SIZE)
        semantic_images_inBatch = train_x[random_samples, :, :, :]
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        make_trainable(discr, False)
        g_loss = GAN.train_on_batch(semantic_images_inBatch, y2)
        losses["g"].append(g_loss)
        print("d_loss: " + str(d_loss) + "  g_loss: " + str(g_loss))
        # Updates plots
        if e % plt_frq == plt_frq - 1:

            for i in range (0, len(generated_images)):
                imsave("11-20-18:30"+str(e)+ "_"+ str(i) + ".png", generated_images[i])
    plot_loss(losses)
    #GAN.save(filepath='model.hdf5')
    GAN.save_weights(filepath='weights.hdf5')


train_for_n(nb_epoch=300, plt_frq=100, BATCH_SIZE=10)