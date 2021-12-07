# -*- coding: utf-8 -*-

We have created a function ```super_res_interpolate```, which carries out super-resolution using basic interpolation (bilinear or bicubic), with which you can compare your results visually and numerically.
"""

# Commented out IPython magic to ensure Python compatibility.

# # Load packages

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from scipy import interpolate
print(tf.keras.__version__)



# choice of the interpolation method
interp_method = 'linear'
# upsampling factor
delta = 2
# the maximum number of data to take from mnist (to go a bit faster)
n_max = 5000

# upsample by a factor of delta
# by definition, the new grid has a step size of 1/delta
def super_res_interpolate(imgs_in,delta,interp_method = 'linear'):
	imgs_out = tf.image.resize( tf.constant(imgs_in),\
		[delta*imgs_in.shape[1],delta*imgs_in.shape[2]], method='bilinear').numpy()

	return(imgs_out)


from keras.datasets import mnist
(X_train, Y_train_scalar), (X_test, Y_test_scalar) = mnist.load_data()

n_max = 5000
X_train = X_train[0:n_max, :, :]
X_test = X_test[0:n_max, :, :]
Y_train_scalar = Y_train_scalar[0:n_max]
Y_test_scalar = Y_test_scalar[0:n_max]

mnist_label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

"""### Réduction de la taille des images

On utilise la fonction ```resize``` de Tensorflow
"""

def downsample(x):
    return tf.image.resize(tf.constant(x[tf.newaxis, ..., tf.newaxis]), [int(0.5*x.shape[0]), int(0.5*x.shape[1])], method='lanczos3').numpy()[0, :, :, 0]


X_train_down = np.array([downsample(img) for img in X_train])
X_test_down = np.array([downsample(img) for img in X_test])

"""### Préparation des données

On permute les entrées par défaut de MNIST avec nos données downsamplées pour s'adapter au modèle et on s'assure que toutes les images ont le même format
"""

X_train, Y_train, X_test, Y_test = X_train_down, X_train, X_test_down, X_test

img_rows, img_cols, nb_channels = X_train.shape[1], X_train.shape[2], 1

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, nb_channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
Y_train /= 255
Y_test /= 255

"""### Définition du modèle

On procède par test avec différentes configuration de modèle, jusqu'à aboutir à celle-ci qui semble satisfaisante. On utilise ```ReduceLROnPlateau``` pour adapter le learning rate au fur et à mesuer de l'apprentissage et ```EarlyStopping``` pour arrêter l'entraînement lorsque celui-ci ne fait plus progresser le réseau (i.e qu'il a atteint des performances satisfaisantes ou qu'il est tombé dans un minimum local)
"""

batch_size = 64
model = Sequential()
model.add(Input(shape=X_train[0].shape))
model.add(UpSampling2D(interpolation="bilinear"))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(8, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
model.summary()

"""__Attention__: Si le réseau s'arrête vers l'epoch numéro 10-15, ou que sa loss stationne autour de 10, c'est qu'il est bloqué dans un minimum local, et qu'il faut donc le recompiler ET le réentrainer. Cela arrive malheureusement ~4 fois sur 5"""

learning_rate = 0.01
model.compile(loss='MeanSquaredError', optimizer=optimizers.Adam(learning_rate), metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1, mode='min', baseline=None, restore_best_weights=True)

n_epochs = 80
model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size, use_multiprocessing=True, callbacks=[reduce_lr, early_stopping])

score = model.evaluate(X_test, Y_test, verbose=False)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### Vérification des résultats

On affiche les résultats du réseau de neurone pour comparer
"""

plt.figure(figsize=(20, 8))
for idx in range(0, 5):
    rand_ind = np.random.randint(0, X_train_down.shape[0])

    plt.subplot(3, 10, idx+1)
    plt.imshow(Y_test[rand_ind, :, :], cmap='gray')
    plt.title(mnist_label_list[int(Y_test_scalar[rand_ind])] + ": truth")

    plt.subplot(3, 10, idx+1+10)
    plt.imshow(model(np.expand_dims(X_test[rand_ind, :, :], axis=0))[
               0, :, :, 0], cmap='gray')
    plt.title(mnist_label_list[int(Y_test_scalar[rand_ind])] + ": network")

    plt.subplot(3, 10, idx+1+20)
    plt.imshow(super_res_interpolate(np.expand_dims(
        X_test[rand_ind, :, :], axis=0), delta)[0, :, :, 0], cmap='gray')
    plt.title(mnist_label_list[int(Y_test_scalar[rand_ind])] + ": bilinear")
