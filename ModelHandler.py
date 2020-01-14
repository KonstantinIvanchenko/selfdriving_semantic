#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Author: Konstantin Ivanchenko
# Date: August 03, 2019

import keras
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Dropout, Activation, Reshape
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import img_to_array, load_img


import tensorflow as tf

import numpy as np
import random
import matplotlib.pyplot as plt


from ImageHandler import ImageHandler


class ModelHandler:

    def __init__(self, load_from=None, image_shape=(320, 480, 3)):

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        self.config.log_device_placement = True  # to log device placement (on which device the operation ran)
        self.sess = tf.Session(config=self.config)
        set_session(self.sess)  # set this TensorFlow session as the default session for Keras

        self.num_classes = 36

        self.sparse = False

        self.image_handler = ImageHandler()

        if load_from is None:

            # initialize new model
            self.model = Sequential()

            self.model.add(Conv2D(64, 5, strides=(2, 2), padding='same', input_shape=image_shape, activation='relu'))

            # self.model.add(ZeroPadding2D(padding=(0, 2)))

            self.model.add(Conv2D(64, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            #TODO: removed last
            self.model.add(Conv2D(64, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            #self.model.add(Dropout(0.5))

            self.model.add(Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            #self.model.add(Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu'))

            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(256, 5, strides=(1, 1), padding='same', activation='relu'))
            #self.model.add(Conv2D(256, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(256, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))

            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(512, 5, strides=(1, 1), padding='same', activation='relu'))
            #self.model.add(Conv2D(512, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            #self.model.add(Conv2D(512, 5, strides=(1, 1), padding='same', activation='relu'))

            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
            #self.model.add(Dropout(0.5))
            self.model.add(UpSampling2D(size=(2, 2)))

            # Adding zero padding on one side to compensate for smaller X size at the output
            # self.model.add(ZeroPadding2D(padding=(0, 1)))

            self.model.add(Conv2D(512, 5, strides=(1, 1), padding='same', activation='relu'))
            #self.model.add(Conv2D(512, 5, strides=(1, 1), padding='same', activation='relu'))
            #self.model.add(Dropout(0.5))
            self.model.add(Conv2D(512, 5, strides=(1, 1), padding='same', activation='relu'))
            #self.model.add(Dropout(0.5))
            self.model.add(UpSampling2D(size=(2, 2)))
            self.model.add(Conv2D(256, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(256, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(256, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(UpSampling2D(size=(2, 2)))
            self.model.add(Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(128, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(UpSampling2D(size=(2, 2)))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(64, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(Conv2D(64, 5, strides=(1, 1), padding='same', activation='relu'))
            self.model.add(UpSampling2D(size=(2, 2), name='to_out'))
            # not required
            #self.model.add(LocallyConnected2D(32, (3,3)))

            #check amount of coonvolution filters to be 255. Older value =1
            if self.sparse is not True:
                self.model.add(
                    Conv2D(self.num_classes + 1, 1, strides=(1, 1), padding='same', activation='softmax', name='output'))
                out_layer = self.model.get_layer(name='output')
                newdim = tuple([x for x in out_layer.output_shape if x != 1 and x is not None])
                self.model.add(Reshape(newdim))

                optimizer = Adam(lr=0.00001, beta_1=0.995, beta_2=0.999, epsilon=1e-08, decay=0)  # learning rate was 0.000025
                #optimizer = Adam(lr=0.00001)
                #optimizer = SGD(lr=0.01, momentum=0.0, decay=0.99, nesterov=False)
                self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            else:
                self.model.add(
                    Conv2D(self.num_classes + 1, 1, strides=(1, 1), padding='same', activation='softmax', name='output'))
                out_layer = self.model.get_layer(name='output')
                newdim = tuple([x for x in out_layer.output_shape if x != 1 and x is not None])
                self.model.add(Reshape(newdim))
                optimizer = Adam(lr=0.0001)
                self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


            #Reshape(newdim)(out_layer)
        else:
            # load model
            self.model = keras.models.load_model(load_from)
            #self.model = self.model.load_weights(load_from)

            """
            prev_layer = self.model.get_layer(name='to_out')
            last_layer = Conv2D(1, 1, strides=(1, 1), padding='same', activation='softmax')(prev_layer)
            newdim = tuple([x for x in last_layer.shape.as_list() if x != 1 and x is not None])
            last_layer = Reshape(newdim)(last_layer)
            self.model.add(last_layer)
            #self.model.add(Activation('softmax'))
            """

    def get_model_parameters(self):
        return self.model.summary()

    def get_model_output(self, x):
        return self.model.predict_classes(x)

    def input_image_to_array(self, img):
        return img_to_array(img)

    # normalizes to 22 classes: 21 defined + 1 unidentified
    def label_class_normalize(self, label_img, n_classes):
        if 0 < n_classes <= 255:
            label_img = label_img * 255 / (n_classes+1)
            it = np.nditer(label_img, op_flags=['readwrite'])
            for x in it:
                if x > 1.0:
                    x[...] = 1.0

    def group_bins(self, label_img, n_groups):
        hist, bins = np.histogram(label_img, n_groups)
        it = np.nditer(label_img, op_flags=['readwrite'])
        label_img_reg = label_img

        for j in range(n_groups):
            for x in range(label_img.shape(0)):
                for y in range(label_img.shape(1)):
                    if label_img[x][y] >= bins[j] and label_img[x][y] < bins[j+1]:
                        label_img_reg[x][y] = bins[j]

        return label_img

    def check_exceed_one(self, label_img):
        it = np.nditer(label_img, op_flags=['readwrite'])
        for x in it:
            if x > 1.0:
                print("BUG: further check required; X= ", x)

    def batch_generator_lrc(self, img_paths_l, img_paths_r, label_paths_l, label_paths_r, batch_size, is_training):
        while True:
            batch_img = []
            batch_label = []

            for i in range(batch_size):
                random_i = random.randint(0, len(img_paths_l) - 1)
                random_b = random.randint(0, 1)

                if is_training:
                    # add augmentation
                    if random_b:
                        img, label = self.image_handler.random_augmentation(img_paths_l[random_i],
                                                                            label_paths_l[random_i])
                    else:
                        img, label = self.image_handler.random_augmentation(img_paths_r[random_i],
                                                                            label_paths_r[random_i])

                else:
                    if random_b:
                        img, label = self.image_handler.load_resized(img_paths_l[random_i],
                                                                            label_paths_l[random_i])
                    else:
                        img, label = self.image_handler.load_resized(img_paths_r[random_i],
                                                                            label_paths_r[random_i])

                #TODO: insert here visual image check
                '''
                fig_img, axs = plt.subplots(1, 2, figsize=(15, 50))
                fig_img.tight_layout()

                axs[0].imshow(img)
                axs[0].set_title('Original input image')
                axs[1].imshow(label)
                axs[1].set_title('Augmented input image')

                plt.show(fig_img)
                '''

                label = to_categorical(label * self.num_classes, num_classes=self.num_classes + 1, dtype='float')

                batch_img.append(img)
                batch_label.append(label)

            yield (np.asarray(batch_img), np.asarray(batch_label))

    def batch_generator_lrc_sparse(self, img_paths_l, img_paths_r, label_paths_l, label_paths_r, batch_size, is_training):
        while True:
            batch_img = []
            batch_label = []

            for i in range(batch_size):
                random_i = random.randint(0, len(img_paths_l) - 1)

                if is_training:
                    # add augmentation
                    img_l, label_l = self.image_handler.random_augmentation(img_paths_l[random_i],
                                                                            label_paths_l[random_i])


                    # img_r, label_r = self.image_handler.random_augmentation(img_paths_r[random_i], label_paths_r[random_i])

                else:
                    # take original image pairs
                    img_l, label_l = self.image_handler.load_resized(img_paths_l[random_i], label_paths_l[random_i])

                    # label_l = to_categorical(label_l, num_classes=25)
                    # img_r, label_r = self.image_handler.load_preprocessed(img_paths_r[random_i], label_paths_r[random_i])

                batch_img.append(img_l)
                # batch_img.append(img_r)
                batch_label.append(label_l)
                # batch_label.append(label_r)

            yield (np.asarray(batch_img), np.asarray(batch_label))


    def model_fit(self, img_paths_l, img_paths_r, label_paths_l, label_paths_r,
                  img_paths_l_v, img_paths_r_v, label_paths_l_v, label_paths_r_v):

        if self.sparse is not True:
            model_history = self.model.fit_generator(self.batch_generator_lrc(img_paths_l, img_paths_r, label_paths_l,
                                                              label_paths_r, batch_size=1, is_training=1),
                                     steps_per_epoch=200, epochs=30,
                                     validation_data=self.batch_generator_lrc(img_paths_l_v, img_paths_r_v, label_paths_l_v,
                                                             label_paths_r_v, batch_size=1, is_training=0),
                                     validation_steps=10, verbose=1, shuffle=1)
        else:
            model_history = self.model.fit_generator(
                self.batch_generator_lrc_sparse(img_paths_l, img_paths_r, label_paths_l,
                                                label_paths_r, batch_size=1, is_training=1),
                steps_per_epoch=100, epochs=50,
                validation_data=self.batch_generator_lrc_sparse(img_paths_l_v, img_paths_r_v, label_paths_l_v,
                                                                label_paths_r_v, batch_size=1, is_training=0),
                validation_steps=30, verbose=1, shuffle=1)

        return model_history

    def model_save(self):
        self.model.save('model_x.h5')

    def model_plot_result(self, history):
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.show()

        plt.clf()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(['training', 'validation'])
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

