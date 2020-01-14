#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Author: Konstantin Ivanchenko
# Date: August 03, 2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from DatasetHandler import DatasetHandler
from ModelHandler import ModelHandler

from ImageHandler import ImageHandler

from sklearn.model_selection import train_test_split as tts

import cv2
from keras.preprocessing.image import array_to_img

im_handler = ImageHandler()

ds_handler = DatasetHandler()
ds_handler.init_apollo_data_source()
ds_handler.read_frame_big_batch('Apollo')

print(ds_handler.data_paths_apollo_leftcam)

model_train = False
new_model = False

# TODO: uncomment to see augmentations
image_paths_l = ds_handler.data_paths_apollo_leftcam#[100]
image_paths_r = ds_handler.data_paths_apollo_rightcam
output_paths_l = ds_handler.label_paths_apollo_leftcam#[100]
output_paths_r = ds_handler.label_paths_apollo_rightcam
#image_handler.augment_batch_visualize(image_paths, valid_paths)

if len(image_paths_l) > len(image_paths_r):
    del image_paths_l[len(image_paths_r):]
    del output_paths_l[len(image_paths_r):]
elif len(image_paths_l) < len(image_paths_r):
    del image_paths_r[len(image_paths_l):]
    del output_paths_r[len(image_paths_l):]


train_input_l, valid_input_l, train_input_r, valid_input_r, train_output_l, valid_output_l, \
train_output_r, valid_output_r = tts(image_paths_l, image_paths_r, output_paths_l, output_paths_r,
                                         test_size=0.01, random_state=5)


# plt.imshow(o)

if model_train:
    if new_model:
        model_handler = ModelHandler()
        model_handler.get_model_parameters()
        history = model_handler.model_fit(train_input_l, train_input_r, train_output_l, train_output_r,
                                valid_input_l, valid_input_r, valid_output_l, valid_output_r)

        model_handler.model_save()
        model_handler.model_plot_result(history)
    else:
        model_handler = ModelHandler(load_from='model_x.h5')
        model_handler.get_model_parameters()
        history = model_handler.model_fit(train_input_l, train_input_r, train_output_l, train_output_r,
                                          valid_input_l, valid_input_r, valid_output_l, valid_output_r)

        model_handler.model_save()
        model_handler.model_plot_result(history)

else:
    model_handler = ModelHandler(load_from='model_x.h5')
    model_handler.get_model_parameters()

    i_batch = []
    i_batch_size = 2
    i_batch_start_ind = 20

    for ind in range(i_batch_size):
        path_i = image_paths_r[i_batch_start_ind+ind]
        path_o = output_paths_r[i_batch_start_ind+ind]
        i, o = im_handler.load_resized(path_i, path_o)
        i = model_handler.input_image_to_array(i)
        i_batch.append(i)

    i_batch = np.array(i_batch)

    #i = np.reshape(i, (None, np.shape(i)))

    y_batch = model_handler.get_model_output(i_batch)

    y_batch = np.array(y_batch)

    fig_img, axs = plt.subplots(i_batch_size, 2, figsize=(25, 50))
    fig_img.tight_layout()

    # create overlay
    for ind in range(i_batch_size):
        i_gray = cv2.cvtColor(i_batch[ind], cv2.COLOR_BGR2GRAY)
        o_gray = np.copy(i_gray)

        print(i_gray.dtype)

        # inp_image = array_to_img(i_gray)
        # output = array_to_img(i_gray)
        # print(np.shape(i_batch[ind]))

        raise_rank = np.expand_dims(y_batch[ind], axis=2)
        #raise_rank = raise_rank[np.newaxis]
        print(np.shape(raise_rank))
        print(raise_rank.dtype)
        raise_rank = raise_rank.astype(np.float32)
        raise_rank = raise_rank #/ 36.0
        # out_overlay = array_to_img(raise_rank)
        cv2.addWeighted(raise_rank, 0.2, i_gray, 0.8, 0, o_gray)

        axs[ind][0].imshow(i_batch[ind])
        axs[ind][0].set_title('Input image')
        axs[ind][1].imshow(o_gray)
        axs[ind][1].set_title('Trained model output')

    plt.show(fig_img)

''' Use it for a single image test
    print(np.shape(i))  # this is a Numpy array with shape (320, 480, 3)

    i = i[np.newaxis, ...]  # dimension added to fit input size

    y = model_handler.get_model_output(i)

    print(np.shape(y))
    y = np.squeeze(y, axis=0)
    print(np.shape(y))

    plt.imshow(y)
    plt.show()
'''