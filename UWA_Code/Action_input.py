# Copyright 2016 Inwoong Lee All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the MSR Action3D text file format."""
import tensorflow as tf
import numpy as np
import os
import csv
import random

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\t", "\n").split()

####### 2017-05-30 Addition
def read(DATA_PATH, VIEW_NUMBER, config=None):
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
        file_list = files

    sklt_input = []
    sklt_label = []

    for fileNo in range(len(file_list)):
        for camNo in VIEW_NUMBER:
            # print file_list[fileNo][1:4]
            if int(file_list[fileNo][13:15]) == camNo:
                # print int(file_list[fileNo][5:7])
                # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
                f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
                csv_reader = csv.reader(f, delimiter=',')
                temp_sklt = []
                for row in csv_reader:
                    temp_sklt.append(row)
                sklt_input.append(temp_sklt)
                sklt_label.append(int(file_list[fileNo][1:3]))
                f.close()
            else:
                pass

    # if CAMERA_OR_SUBJECT == 1:
    #     # CAMERA
    #     for fileNo in range(len(file_list)):
    #         for camNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][5:8]) == camNo:
    #                 # print int(file_list[fileNo][5:7])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass
    # else:
    #     # SUBJECT
    #     for fileNo in range(len(file_list)):
    #         for subjNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][9:12]) == subjNo:
    #                 # print int(file_list[fileNo][9:12])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass

    return sklt_input, sklt_label



####### 2017-05-30 Deletion
# def read(cv_set, DATA_PATH, CAMERA_OR_SUBJECT, config=None):
#     for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
#         file_list = files
#
#     sklt_input = []
#     sklt_label = []
#
#     if CAMERA_OR_SUBJECT == 1:
#         # CAMERA
#         for fileNo in range(len(file_list)):
#             for camNo in cv_set:
#                 # print file_list[fileNo][1:4]
#                 if int(file_list[fileNo][5:8]) == camNo:
#                     # print int(file_list[fileNo][5:7])
#                     # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
#                     f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
#                     csv_reader = csv.reader(f, delimiter=',')
#                     temp_sklt = []
#                     for row in csv_reader:
#                         temp_sklt.append(row)
#                     sklt_input.append(temp_sklt)
#                     sklt_label.append(int(file_list[fileNo][17:20]))
#                     f.close()
#                 else:
#                     pass
#     else:
#         # SUBJECT
#         for fileNo in range(len(file_list)):
#             for subjNo in cv_set:
#                 # print file_list[fileNo][1:4]
#                 if int(file_list[fileNo][9:12]) == subjNo:
#                     # print int(file_list[fileNo][9:12])
#                     # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
#                     f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
#                     csv_reader = csv.reader(f, delimiter=',')
#                     temp_sklt = []
#                     for row in csv_reader:
#                         temp_sklt.append(row)
#                     sklt_input.append(temp_sklt)
#                     sklt_label.append(int(file_list[fileNo][17:20]))
#                     f.close()
#                 else:
#                     pass
#
#     return sklt_input, sklt_label

def MSR_iterator(raw_data, label, batch_size, input_size, num_steps, is_training):
    # raw_data = np.array(raw_data, dtype=np.float32)
    #
    data_len = len(raw_data)
    batch_len = data_len // batch_size

    batch_index = np.arange(batch_size)

    # data = np.zeros([batch_size, batch_len], dtype=np.float32)
    # for i in range(batch_size):
    #   data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    #
    epoch_size = (batch_len - 1) // num_steps
    #
    # if epoch_size == 0:
    #   raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(batch_len):
        if is_training == True:
            for randNo in range(batch_size):
                batch_index[randNo] = random.randint(0, data_len-1)
            x = raw_data[batch_index, :, :]
            y = label[batch_index, :]
        else:
            x = raw_data[i*batch_size:(i+1)*batch_size, :, :]
            y = label[i*batch_size:(i+1)*batch_size, :]
        yield (x, y)