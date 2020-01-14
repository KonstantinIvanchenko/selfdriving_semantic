import matplotlib.image as mpimg
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from imgaug import augmenters as iaa
#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Author: Konstantin Ivanchenko
# Date: August 03, 2019

class ImageHandler:
    def __init__(self):
        self.scale = 1.3 # 30%
        self.translate = 0.1 # 10%
        self.multiply = 0.80 # from 80% to 100%
        self.affine_scale = (0.10, 0.15)

        self.resize_dim = (480, 320)

    # augmentation with zoom in
    def zoom(self, image_in, image_out):
        # augmented zoom in limit to 30%
        zoom = iaa.Affine(scale=(1, self.scale), deterministic=True)
        image_in = zoom.augment_image(image_in)
        image_out = zoom.augment_image(image_out)
        return image_in, image_out


#TODO: apply the same to all augmentation to generate same images
    # augmentation with pan in x an y directions
    def pan(self, image_in, image_out):
        # augmented pan by 10% in both directions
        pan = iaa.Affine(translate_percent={"x": (-self.translate, self.translate), "y": (-self.translate, self.translate)}, deterministic=True)
        image_in = pan.augment_image(image_in)
        image_out = pan.augment_image(image_out)
        return image_in, image_out

    def affine(self, image_in, image_out):
        affine = iaa.PiecewiseAffine(scale=self.affine_scale, nb_rows=2, nb_cols=2, deterministic=True)
        image_in = affine.augment_image(image_in)
        image_out = affine.augment_image(image_out)
        return image_in, image_out

    # augmentation with brightness regulation
    def brightness(self, image_in, image_out):
        # augmentation with brightness level between 80%..100%
        brightness = iaa.Multiply((self.multiply, 1.0), deterministic=True)
        image_in = brightness.augment_image(image_in)
        image_out = brightness.augment_image(image_out)
        return image_in, image_out

    # augmentation with image flipping (mirroring)
    # not a part of imgaug library - use opencv instead.
    # Remark: essentially it allows for better data balancing
    def flip(self, image_in, image_out):
            image_in = cv2.flip(image_in, 1)
            image_out = cv2.flip(image_out, 1)
            return image_in, image_out

    # add resizeing to images
    def img_resize(self, img):
        #img = mpimg.imread(img)
        # change color space. Use YUV format instead of RGB (Recommended by nVIDIA)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        # Gaussian blur
        #img = cv2.GaussianBlur(img, (3, 3), 0)
        # resize
        img = cv2.resize(img, self.resize_dim)
        # normalize
        #img = img/255
        return img

    def img_normalize(self, img):
        img = img.astype(np.float32)
        img = img/255
        return img

    def load_image(self, im_path):
        return mpimg.imread(im_path)

    def load_image_pair(self, im_path1, im_path2):
        return mpimg.imread(im_path1), mpimg.imread(im_path2)

    def get_image_size(self, im_path):
        img = mpimg.imread(im_path)
        return np.shape(img)

    def load_resized(self, im_path1, label_path2):
        image, label_img = self.load_image_pair(im_path1, label_path2)
        return self.img_normalize(self.img_resize(image)), self.img_resize(label_img)

    # apply augmentations randomly
    def random_augmentation(self, im_path1, label_path2):
        image, label_img = self.load_image_pair(im_path1, label_path2)

        # rand generates a number from 0..1
        #if np.random.rand() < 0.5:
        #    image, label_img = self.pan(image, label_img)

        image = self.img_resize(image)
        image = self.img_normalize(image)
        label_img = self.img_resize(label_img)  # label_img is already normalized

        if np.random.rand() < 0.5:
            image, label_img = self.affine(image, label_img)
        # no zoom and no pan now
        #if np.random.rand() < 0.5:
        #    image, label_img = self.zoom(image, label_img)
        if np.random.rand() < 0.5:
            image, label_img = self.brightness(image, label_img)
        if np.random.rand() < 0.5:
            image, label_img = self.flip(image, label_img)

        return image, label_img

# TODO: test visualization of augmentation
    def augment_batch_visualize(self, image_paths, label_img_paths, position=1000, quantity=5):
        if quantity <= 0 or quantity > 20:
            quantity = 10
        if position <= 0 or position > len(image_paths):
            position = 0

        fig_img, axs = plt.subplots(quantity, 4, figsize=(15,50))
        fig_img.tight_layout()

        for n in range(quantity):
            elem = n+position
            orig_img, orig_label = self.load_image_pair(image_paths[elem], label_img_paths[elem])
            im, lb = self.random_augmentation(image_paths[elem], label_img_paths[elem])

            axs[n][0].imshow(orig_img)
            axs[n][0].set_title('Original input image')
            axs[n][1].imshow(im)
            axs[n][1].set_title('Augmented input image')
            axs[n][2].imshow(orig_label)
            axs[n][2].set_title('Original output image')
            axs[n][3].imshow(lb)
            axs[n][3].set_title('Augmented output image')

        plt.show(fig_img)

