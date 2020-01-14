#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Author: Konstantin Ivanchenko
# Date: August 03, 2019
import pandas as pd

class DatasetHandler:
    def __init__(self):

        self.images_batch = []
        self.outputs_batch = []

        self.datadir = "/media/konstiva/My Book/trainDataset"

        self.data_source = {'Apollo', }
        self.data_attribute_types_apollo = ['record_n', 'camera_n', 'image_n']
        self.apollo_image_subdir = "/annotation-apollo_scape_label-train_image-apollo-1.5"
        self.apollo_label_subdir = "/annotation-apollo_scape_label-train_label-apollo-1.5"

        self.data_paths_apollo_leftcam = []
        self.data_paths_apollo_rightcam = []
        self.label_paths_apollo_leftcam = []
        self.label_paths_apollo_rightcam = []

        self.apollo_init = False
        self.data_attributes_apollo = []
        self.record_num = 0  # Number of record packages


    def init_apollo_data_source(self):
        # self.data_paths_apollo = pd.read_csv(datadir+'/annotation-apollo_scape_label-train-apollo-1.5.txt',
        #                                  header=None)
        self.data_attributes_apollo = pd.read_csv(self.datadir + '/annotation-apollo_scape_label-train-apollo-1.5.txt',
                                                  sep="/", header=None, index_col=False)
        self.data_attributes_apollo.columns = self.data_attribute_types_apollo
        # self.data_attributes_apollo = self.path_leaf_apollo()
        self.apollo_init = True

    def read_frame_big_batch(self, source, record_index=0):
        if source == 'Apollo' and self.apollo_init is True:

            self.record_num = len(self.data_attributes_apollo.groupby('record_n').nunique())
            record_names = self.data_attributes_apollo['record_n'].unique()

            if record_index >= self.record_num:
                record_index = self.record_num-1
            # print(self.data_attributes_apollo)

            # create condition
            record_by_name = self.data_attributes_apollo['record_n'] == record_names[record_index]
            df = self.data_attributes_apollo[record_by_name]
            record_by_cam5 = self.data_attributes_apollo['camera_n'] == "Camera 5"
            record_by_cam6 = self.data_attributes_apollo['camera_n'] == "Camera 6"
            df_cam5 = df[record_by_cam5]
            df_cam6 = df[record_by_cam6]

            for index, row in df_cam5.iterrows():
                # print(row)
                self.data_paths_apollo_leftcam.append(self.datadir+self.apollo_image_subdir+'/image/'+row['record_n']+
                                                      '/Camera 5/'+row['image_n']+'.jpg')
                self.label_paths_apollo_leftcam.append(self.datadir+self.apollo_label_subdir+'/label/'+row['record_n']+
                                                      '/Camera 5/'+row['image_n']+'.png')
                # self.data_paths_apollo
            for index, row in df_cam6.iterrows():
                # print(row)
                self.data_paths_apollo_rightcam.append(self.datadir+self.apollo_image_subdir+'/image/'+row['record_n']+
                                                       '/Camera 6/'+row['image_n']+'.jpg')
                self.label_paths_apollo_rightcam.append(self.datadir+self.apollo_label_subdir+'/label/'+row['record_n']+
                                                      '/Camera 6/'+row['image_n']+'.png')

        return self.record_num
        #return self.images_batch

    def read_frame(self, data_path):
        #read image
        #read semantic output

        return # image, semantic_output