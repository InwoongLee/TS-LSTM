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

import time

from datetime import datetime
from time import localtime, strftime

import tensorflow as tf
import numpy as np

import os
import csv
import sys

import Action_input

from PreProcesses import Human_Cognitive_Coordinate, bone_length, extract_AS, extract_data_label, extract_feature, count_zeros, feature_only_diff, one_hot_labeling

# from ops import batch_norm_rnn, batch_norm

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

win_size = [164, 155, 145, 88, 84, 42]
stride = [80, 80, 80, 40, 40, 20]
start_time = [1, 3, 7, 3, 7, 1]

train_set = [int(sys.argv[1]), int(sys.argv[2])]
test_set1 = [int(sys.argv[3])]
test_set2 = [int(sys.argv[4])]

# train_set = [1, 2]
# test_set1 = [3]
# test_set2 = [4]

_now = datetime.now()
BatchDivider = 30
myKeepProb = 0.3
myHiddenSize = 50
view_subject = 'Ensemble TS-LSTM v2'
myFolderPath = './TrialSaver/%s/%04d%02d%02d_%02d%02d_BD_%d_KB_%.2f_TrV_%01d_%01d_TeV_%01d_%01d'%(view_subject,_now.year,
                                                                           _now.month,
                                                                           _now.day,
                                                                           _now.hour,
                                                                           _now.minute,
                                                                           BatchDivider,
                                                                           myKeepProb, train_set[0], train_set[1], test_set1[0], test_set2[0])
os.mkdir(myFolderPath)

def getDevice():
    return FLAGS.num_gpus

device_index = [ 0, 0, 0, 0, 0, 0 ]

gradient_device = ['/gpu:0','/gpu:0','/gpu:0','/gpu:0']

class MP_runner(object):
    # logits_0, logits_1, logits_2, cost, accuracy, final_state, pred_labels, given_labels, eval_op, {m.input_data: x, m.targets: y}

    """The MSR model."""

    def __init__(self, is_training, config, labels):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20

        counter = 0

        self.data_initializer = tf.placeholder(tf.float32, [batch_size, num_steps, feature_size], name="x-input")
        self.label_initializer = tf.placeholder(tf.float32, [batch_size, class_size], name="y-input")

        self.input_data = tf.Variable(self.data_initializer, trainable=False, collections=[], name="x-top")
        self.targets = tf.Variable(self.label_initializer, trainable=False, collections=[], name="y-top")

        if is_training:
            sw_0 = sw_1 = sw_2 = sw_3 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = sw_3 = '/cpu:0'

        with tf.device(sw_0):
            with tf.name_scope('%s_%d' % ('mGPU', 0)) as scope:
                self.mL = mL = TS_Long_LSTM_Top(is_training, config, labels, self.input_data, self.targets, size)
                logits_l, cross_entropy_l, state_0 = mL.get_inference()
                self._logits_0, self._cross_entropy_0, self._final_state = \
                    logits_0, cross_entropy_0, final_state = logits_l, cross_entropy_l, state_0

                self._cost_L = tf.reduce_sum(cross_entropy_l / 4 ) / batch_size

                if is_training:
                    self.tvars_L = tf.trainable_variables()
                    count = len(self.tvars_L)
                    print("Long : ",len(self.tvars_L))
                    print(self.tvars_L)
                    self.grad_before_sum_L = tf.gradients(self._cost_L, self.tvars_L)

        with tf.device(sw_1):
            with tf.name_scope('%s_%d' % ('mGPU', 1)) as scope:
                self.mM = mM = TS_Medium_LSTM_Top(is_training, config, labels, self.input_data, self.targets, size)
                logits_m, cross_entropy_m, state_3 = mM.get_inference()
                self._logits_1, self._cross_entropy_1, self._state_3 = \
                    logits_1, cross_entropy_1, state_3 = logits_m, cross_entropy_m, state_3

                self._cost_M = tf.reduce_sum(cross_entropy_m / 4) / batch_size
                if is_training:
                    temp_tvars_M = tf.trainable_variables()
                    self.tvars_M = temp_tvars_M[count:len(temp_tvars_M)]
                    count = len(temp_tvars_M)
                    print("Medium : ", len(self.tvars_M))
                    print(self.tvars_M)
                    self.grad_before_sum_M = tf.gradients(self._cost_M, self.tvars_M)

        with tf.device(sw_2):
            with tf.name_scope('%s_%d' % ('mGPU', 2)) as scope:
                self.mS = mS = TS_Short_LSTM_Top(is_training, config, labels, self.input_data, self.targets, size)
                logits_s, cross_entropy_s, state_4 = mS.get_inference()
                self._logits_2, self._cross_entropy_2, self._state_4 = \
                    logits_2, cross_entropy_2, state_4 = logits_s, cross_entropy_s, state_4

                self._cost_S = tf.reduce_sum(cross_entropy_s / 4) / batch_size
                if is_training:
                    temp_tvars_S = tf.trainable_variables()
                    self.tvars_S = temp_tvars_S[count:len(temp_tvars_S)]
                    count = len(temp_tvars_S)
                    print("Short : ", len(self.tvars_S))
                    print(self.tvars_S)
                    self.grad_before_sum_S = tf.gradients(self._cost_S, self.tvars_S)

        with tf.device(sw_3):
            with tf.name_scope('%s_%d' % ('mGPU', 3)) as scope:
                self.mS = mS = TS_Motion_LSTM_Top(is_training, config, labels, self.input_data, self.targets, size)
                logits_motion, cross_entropy_motion, state_motion = mS.get_inference()
                self._logits_motion, self._cross_entropy_motion, self._state_motion = \
                    logits_motion, cross_entropy_motion, state_motion = logits_motion, cross_entropy_motion, state_motion

                self._cost_motion = tf.reduce_sum(cross_entropy_motion / 4) / batch_size
                if is_training:
                    temp_tvars_motion = tf.trainable_variables()
                    self.tvars_motion = temp_tvars_motion[count:len(temp_tvars_motion)]
                    count = len(temp_tvars_motion)
                    print("Motion : ", len(self.tvars_motion))
                    print(self.tvars_motion)
                    self.grad_before_sum_motion = tf.gradients(self._cost_motion, self.tvars_motion)

        with tf.device(sw_3):
            print("Parallelized Model is on building!!")

            self._cost = self._cost_L + self._cost_M + self._cost_S


            if not is_training:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = (logits_0 + logits_1 + logits_2 + logits_motion) / 4
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size
            else:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = (logits_l + logits_m + logits_s + logits_motion) / 4
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size


        if is_training:
            with tf.device(gradient_device[3]):
                with tf.name_scope("train") as scope:
                    with tf.name_scope("Merging_Gradient"):
                        print("L : ", len(self.grad_before_sum_L))
                        print("M : ", len(self.grad_before_sum_M))
                        print("S : ", len(self.grad_before_sum_S))
                        print("Motion : ", len(self.grad_before_sum_motion))
                        self.grad_after_sum = self.grad_before_sum_L + self.grad_before_sum_M + self.grad_before_sum_S + self.grad_before_sum_motion
                        print("L+M+S : ", len(self.grad_after_sum))
                        self._lr = tf.Variable(0.0, trainable=False)
                        self.grads, _ = tf.clip_by_global_norm(self.grad_after_sum, config.max_grad_norm)
                        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                        self.tvars = tf.trainable_variables()
                        with tf.name_scope("Applying-Gradient"):
                            # print("List : ",self.grads)
                            self._train_op = self.optimizer.apply_gradients(zip(self.grads,self.tvars))

        print("Calculating Graph is fully connected!!")

    def assign_lr(self, sess, lr_value):
        sess.run(tf.assign(self.lr, lr_value))

    def mp_init(self, sess):
        self.mL.init_all_var(sess)
        self.mM.init_all_var(sess)
        self.mS.init_all_var(sess)

    @property
    def initial_state_L(self):
        return self.mL.initial_state

    @property
    def initial_state_M(self):
        return self.mM.initial_state

    @property
    def initial_state_S(self):
        return self.mS.initial_state

    @property
    def logits_0(self):
        return self._logits_0

    @property
    def logits_1(self):
        return self._logits_1

    @property
    def logits_2(self):
        return self._logits_2

    @property
    def cost_L(self):
        return self._cost_L

    @property
    def cost_M(self):
        return self._cost_M

    @property
    def cost_S(self):
        return self._cost_S

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def correct_prediction(self):
        return self._correct_prediction

    @property
    def training(self):
        return self._training

    @property
    def ac_summ(self):
        return self._ac_summ

    @property
    def summary_op(self):
        return self._summary_op


class TS_Long_LSTM_Top(object):
    # logits_0, logits_1, logits_2, cost, accuracy, final_state, pred_labels, given_labels, eval_op, {m.input_data: x, m.targets: y}

    def __init__(self, is_training, config, labels, top_input_data, top_targets, top_hidden_size):
        if is_training:
            sw_0 = sw_1 = sw_2 = sw_3 = sw_4 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = sw_3 = sw_4 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps  # 76
            self.class_size = class_size = config.class_size
            size = top_hidden_size  # 20

            self.input_data = top_input_data
            self.targets = top_targets

            num_LSTMs_0 = len(range(start_time[0], num_steps - win_size[0] + start_time[0], stride[0]))
            num_LSTMs_1 = len(range(start_time[1], num_steps - win_size[1] + start_time[1], stride[1]))
            num_LSTMs_2 = len(range(start_time[2], num_steps - win_size[2] + start_time[2], stride[2]))

        # self.scale_intput = scale_intput = tf.Variable(80.0, name="scale_intput")
        with tf.variable_scope("Long_Sliding"):
            print("Long_Sliding_Top")

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Long0_GPU', device_index[0])) as scope:
                    with tf.variable_scope('l0'):
                        self.mL0 = mL0 = TS_LSTM_Long_0(is_training, config, self.input_data)
                        self._initial_state = mL0.initial_state
                        output_depthconcat_long_0 = mL0.get_depth_concat_output()
                        self.output_depthconcat_long_0 = output_depthconcat_long_0

            with tf.device(sw_2):
                with tf.name_scope('%s_%d' % ('Long1_GPU', device_index[1])) as scope:
                    with tf.variable_scope('l1'):
                        self.mL1 = mL1 = TS_LSTM_Long_1(is_training, config, self.input_data)
                        self._initial_state = mL0.initial_state
                        output_depthconcat_long_1 = mL1.get_depth_concat_output()
                        self.output_depthconcat_long_1 = output_depthconcat_long_1

            with tf.device(sw_3):
                with tf.name_scope('%s_%d' % ('Long2_GPU', device_index[2])) as scope:
                    with tf.variable_scope('l2'):
                        self.mL2 = mL2 = TS_LSTM_Long_2(is_training, config, self.input_data)
                        self._initial_state = mL0.initial_state
                        output_depthconcat_long_2 = mL2.get_depth_concat_output()
                        self.output_depthconcat_long_2 = output_depthconcat_long_2

            with tf.device(sw_4):
                with tf.variable_scope("Concat_0"):
                    output_real_temp_0 = tf.concat([output_depthconcat_long_0, output_depthconcat_long_1], 1)
                    output_real_0 = tf.concat([output_real_temp_0, output_depthconcat_long_2], 1)

                with tf.variable_scope("Drop_0"):
                    if is_training and config.keep_prob < 1:
                        output_real_0 = tf.nn.dropout(output_real_0, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_0 = output_real_0 * config.keep_prob

                with tf.variable_scope("Softmax_0"):
                    self.softmax_w_0 = softmax_w_0 = tf.get_variable("softmax_w_0",
                                                                     [(num_LSTMs_0 + num_LSTMs_1 + num_LSTMs_2) * size,
                                                                      class_size])
                    self.softmax_b_0 = softmax_b_0 = tf.get_variable("softmax_b_0", [class_size])
                    self.logits = logits = tf.nn.softmax(tf.matmul(output_real_0, softmax_w_0) + softmax_b_0)

                    ######
                    # self.cross_entropy = cross_entropy = -tf.reduce_sum(self.targets * tf.log(logits))
                    self.cross_entropy = cross_entropy = -tf.reduce_sum(
                        self.targets * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
                    self._final_state = mL0.get_state()
                    self._lr = tf.Variable(0.0, trainable=False)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value, validate_shape=False))

    def get_inference(self):
        return self.logits, self.cross_entropy, self.final_state

    def init_all_var(self, session):
        init = tf.global_variables_initializer()
        session.run(init)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def training(self):
        return self._training


class TS_Medium_LSTM_Top(object):
    def __init__(self, is_training, config, labels, top_input_data, top_targets, top_hidden_size):
        if is_training:
            sw_0 = sw_1 = sw_2 = sw_3 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = sw_3 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps  # 76
            self.class_size = class_size = config.class_size
            size = top_hidden_size  # 20

            self.input_data = top_input_data
            self.targets = top_targets

            num_LSTMs_3 = len(range(start_time[3], num_steps - win_size[3] + start_time[3], stride[3]))
            num_LSTMs_4 = len(range(start_time[4], num_steps - win_size[4] + start_time[4], stride[4]))

        with tf.variable_scope("Medium_Sliding"):
            print("Medium_Sliding_Top")

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Medium3_GPU', device_index[3])) as scope:
                    with tf.variable_scope('m3'):
                        self.mM3 = mM3 = TS_LSTM_Medium_3(is_training, config, self.input_data)
                        self._initial_state = mM3.initial_state
                        output_depthconcat_medium_3 = mM3.get_depth_concat_output()
                        self.output_depthconcat_medium_3 = output_depthconcat_medium_3

            with tf.device(sw_2):
                with tf.name_scope('%s_%d' % ('Medium4_GPU', device_index[4])) as scope:
                    with tf.variable_scope('m4'):
                        self.mM4 = mM4 = TS_LSTM_Medium_4(is_training, config, self.input_data)
                        self._initial_state = mM4.initial_state
                        output_depthconcat_medium_4 = mM4.get_depth_concat_output()
                        self.output_depthconcat_medium_4 = output_depthconcat_medium_4

            with tf.device(sw_3):
                with tf.variable_scope("Concat_1"):
                    output_real_1 = tf.concat([output_depthconcat_medium_3, output_depthconcat_medium_4], 1)

                with tf.variable_scope("Drop_1"):
                    if is_training and config.keep_prob < 1:
                        output_real_1 = tf.nn.dropout(output_real_1, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_1 = output_real_1 * config.keep_prob

                with tf.variable_scope("Softmax_1"):
                    self.softmax_w_1 = softmax_w_1 = tf.get_variable("softmax_w_1",
                                                                     [(num_LSTMs_3 + num_LSTMs_4) * size, class_size])
                    self.softmax_b_1 = softmax_b_1 = tf.get_variable("softmax_b_1", [class_size])
                    self.logits = logits = tf.nn.softmax(tf.matmul(output_real_1, softmax_w_1) + softmax_b_1)
                    ######
                    # self.cross_entropy = cross_entropy = -tf.reduce_sum(self.targets * tf.log(logits))
                    self.cross_entropy = cross_entropy = -tf.reduce_sum(
                        self.targets * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
                    self._final_state = mM3.state_3
                    self._lr = tf.Variable(0.0, trainable=False)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value, validate_shape=False))

    def get_inference(self):
        return self.logits, self.cross_entropy, self.final_state

    def init_all_var(self, session):
        init = tf.global_variables_initializer()
        session.run(init)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def training(self):
        return self._training


class TS_Short_LSTM_Top(object):
    def __init__(self, is_training, config, labels, top_input_data, top_targets, top_hidden_size):
        if is_training:
            sw_0 = sw_1 = sw_2 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps  # 76
            self.class_size = class_size = config.class_size
            size = top_hidden_size  # 20

            self.input_data = top_input_data
            self.targets = top_targets

            num_LSTMs_5 = len(range(start_time[5], num_steps - win_size[5] + start_time[5], stride[5]))

        with tf.variable_scope("Short_Sliding"):
            print("Short_Sliding_Top")

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Short5_GPU', device_index[5])) as scope:
                    with tf.variable_scope('s5'):
                        self.mS5 = mS5 = TS_LSTM_Short_5(is_training, config, self.input_data)
                        self._initial_state = mS5.initial_state
                        output_depthconcat_short_5 = mS5.get_depth_concat_output()
                        self.output_depthconcat_short_5 = output_depthconcat_short_5

            with tf.device(sw_2):
                with tf.variable_scope("Concat_2"):
                    output_real_2 = output_depthconcat_short_5

                with tf.variable_scope("Drop_2"):
                    if is_training and config.keep_prob < 1:
                        output_real_2 = tf.nn.dropout(output_real_2, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_2 = output_real_2 * config.keep_prob

                with tf.variable_scope("Softmax_2"):
                    self.softmax_w_2 = softmax_w_2 = tf.get_variable("softmax_w", [num_LSTMs_5 * size, class_size])
                    self.softmax_b_2 = softmax_b_2 = tf.get_variable("softmax_b", [class_size])
                    self.logits = logits = tf.nn.softmax(tf.matmul(output_real_2, softmax_w_2) + softmax_b_2)
                    ######
                    # self.cross_entropy = cross_entropy = -tf.reduce_sum(self.targets * tf.log(logits))
                    self.cross_entropy = cross_entropy = -tf.reduce_sum(
                        self.targets * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
                    self._final_state = mS5.state_5
                    self._lr = tf.Variable(0.0, trainable=False)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value, validate_shape=False))

    def get_inference(self):
        return self.logits, self.cross_entropy, self.final_state

    def init_all_var(self, session):
        init = tf.global_variables_initializer()
        session.run(init)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def training(self):
        return self._training

class TS_Motion_LSTM_Top(object):
    def __init__(self, is_training, config, labels, top_input_data, top_targets, top_hidden_size):
        if is_training:
            sw_0 = sw_1 = sw_2 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps  # 76
            self.class_size = class_size = config.class_size
            size = top_hidden_size  # 20

            self.input_data = top_input_data
            self.targets = top_targets

            num_LSTMs_5 = len(range(start_time[5], num_steps - win_size[5] + start_time[5], stride[5]))

        with tf.variable_scope("Motion_Sliding"):
            print("Short_Sliding_Top")

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Short5_GPU', device_index[5])) as scope:
                    with tf.variable_scope('s5'):
                        self.mS5 = mS5 = TS_LSTM_Motion_6(is_training, config, self.input_data)
                        self._initial_state = mS5.initial_state
                        output_depthconcat_short_5 = mS5.get_depth_concat_output()
                        self.output_depthconcat_short_5 = output_depthconcat_short_5


            with tf.device(sw_2):
                with tf.variable_scope("Concat_2"):
                    output_real_2 = output_depthconcat_short_5

                with tf.variable_scope("Drop_2"):
                    if is_training and config.keep_prob < 1:
                        output_real_2 = tf.nn.dropout(output_real_2, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_2 = output_real_2 * config.keep_prob

                with tf.variable_scope("Softmax_2"):
                    self.softmax_w_2 = softmax_w_2 = tf.get_variable("softmax_w", [num_LSTMs_5 * size, class_size])
                    self.softmax_b_2 = softmax_b_2 = tf.get_variable("softmax_b", [class_size])
                    self.logits = logits = tf.nn.softmax(tf.matmul(output_real_2, softmax_w_2) + softmax_b_2)
                    ######
                    # self.cross_entropy = cross_entropy = -tf.reduce_sum(self.targets * tf.log(logits))
                    self.cross_entropy = cross_entropy = -tf.reduce_sum(
                        self.targets * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
                    self._final_state = mS5.state_5
                    self._lr = tf.Variable(0.0, trainable=False)


    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value, validate_shape=False))

    def get_inference(self):
        return self.logits, self.cross_entropy, self.final_state

    def init_all_var(self, session):
        init = tf.initialize_all_variables()
        session.run(init)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def training(self):
        return self._training

class TS_LSTM_Long_0(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_0 = []
        self.state_0 = state_0 = []
        win_size_0 = win_size[0]  # 75 frames
        stride_0 = stride[0]  # df 1
        start_time_0 = start_time[0]  # 1
        num_LSTMs_0 = len(range(start_time_0, num_steps - win_size_0 + start_time_0, stride_0))
        print(range(start_time_0, num_steps - win_size_0 + start_time_0, stride_0))
        print("num_LSTMs_0: ", num_LSTMs_0)
        for time_step in range(start_time_0, num_steps):
            for win_step in range(num_LSTMs_0):
                if time_step == start_time_0:
                    self.outputs_0.append([])
                    state_0.append([])
                    state_0[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_0 + win_step * stride_0:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_0[win_step].append(cell_output)
                    elif time_step >= start_time_0 + win_step * stride_0 and time_step < start_time_0 + win_step * stride_0 + win_size_0:
                        if time_step > start_time_0 + win_step * stride_0: tf.get_variable_scope().reuse_variables()
                        distance = (
                                       inputs[:, time_step, :] - inputs[:, time_step - start_time_0, :]) / (
                                       start_time_0 + 1)
                        (cell_output, state_0[win_step]) = cell(distance * 100, state_0[win_step])
                        self.outputs_0[win_step].append(cell_output)
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_0[win_step].append(cell_output)
        output_0 = []
        for win_step in range(num_LSTMs_0):
            output_0.append([])
            output_0[win_step] = tf.reshape(tf.concat(self.outputs_0[win_step], 1), [-1, size])

        with tf.variable_scope("Dep_Con_0"):
            temp_output_0 = []
            for win_step in range(num_LSTMs_0):
                temp_output_0.append([])
                temp_output_0[win_step] = tf.reshape(output_0[win_step],
                                                     [batch_size, num_steps - start_time_0, size])
                if win_step == 0:
                    input_0 = temp_output_0[win_step]
                else:
                    input_0 = tf.concat([input_0, temp_output_0[win_step]], 1)
            input_0 = tf.reshape(input_0, [batch_size, num_LSTMs_0, num_steps - start_time_0, size])
            concat_output_real_0 = tf.reduce_sum(input_0, 2)
            self.out_concat_output_real_0 = tf.reshape(concat_output_real_0, [batch_size, num_LSTMs_0 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_0

    def get_state(self):
        return self.state_0

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class TS_LSTM_Long_1(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_1 = []
        self.state_1 = state_1 = []
        win_size_1 = win_size[1]  # 71 frames
        stride_1 = stride[1]  # df 5
        start_time_1 = start_time[1]  # 5
        num_LSTMs_1 = len(range(start_time_1, num_steps - win_size_1 + start_time_1, stride_1))
        print(range(start_time_1, num_steps - win_size_1 + start_time_1, stride_1))
        print("num_LSTMs_1: ", num_LSTMs_1)
        for time_step in range(start_time_1, num_steps):
            for win_step in range(num_LSTMs_1):
                if time_step == start_time_1:
                    self.outputs_1.append([])
                    state_1.append([])
                    state_1[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_1 + win_step * stride_1:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_1[win_step].append(cell_output)
                    elif time_step >= start_time_1 + win_step * stride_1 and time_step < start_time_1 + win_step * stride_1 + win_size_1:
                        if time_step > start_time_1 + win_step * stride_1: tf.get_variable_scope().reuse_variables()
                        distance = (
                                       inputs[:, time_step, :] - inputs[:, time_step - start_time_1, :]) / (
                                       start_time_1 + 1)
                        (cell_output, state_1[win_step]) = cell(distance * 100, state_1[win_step])
                        self.outputs_1[win_step].append(cell_output)
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_1[win_step].append(cell_output)
        output_1 = []
        for win_step in range(num_LSTMs_1):
            output_1.append([])
            output_1[win_step] = tf.reshape(tf.concat(self.outputs_1[win_step], 1), [-1, size])

        with tf.variable_scope("Dep_Con_1"):
            temp_output_1 = []
            for win_step in range(num_LSTMs_1):
                temp_output_1.append([])
                temp_output_1[win_step] = tf.reshape(output_1[win_step],
                                                     [batch_size, num_steps - start_time_1, size])
                if win_step == 0:
                    input_1 = temp_output_1[win_step]
                else:
                    input_1 = tf.concat([input_1, temp_output_1[win_step]], 1)
            input_1 = tf.reshape(input_1, [batch_size, num_LSTMs_1, num_steps - start_time_1, size])
            concat_output_real_1 = tf.reduce_sum(input_1, 2)
            self.out_concat_output_real_1 = tf.reshape(concat_output_real_1, [batch_size, num_LSTMs_1 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_1

    def get_state(self):
        return self.state_1

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class TS_LSTM_Long_2(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_2 = []
        self.state_2 = state_2 = []
        win_size_2 = win_size[2]  # 66 frames
        stride_2 = stride[2]  # df 10
        start_time_2 = start_time[2]  # 10
        num_LSTMs_2 = len(range(start_time_2, num_steps - win_size_2 + start_time_2, stride_2))
        print(range(start_time_2, num_steps - win_size_2 + start_time_2, stride_2))
        print("num_LSTMs_2: ", num_LSTMs_2)
        for time_step in range(start_time_2, num_steps):
            for win_step in range(num_LSTMs_2):
                if time_step == start_time_2:
                    self.outputs_2.append([])
                    state_2.append([])
                    state_2[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_2 + win_step * stride_2:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_2[win_step].append(cell_output)
                    elif time_step >= start_time_2 + win_step * stride_2 and time_step < start_time_2 + win_step * stride_2 + win_size_2:
                        if time_step > start_time_2 + win_step * stride_2: tf.get_variable_scope().reuse_variables()
                        distance = (
                                       inputs[:, time_step, :] - inputs[:, time_step - start_time_2, :]) / (
                                       start_time_2 + 1)
                        (cell_output, state_2[win_step]) = cell(distance * 100, state_2[win_step])
                        self.outputs_2[win_step].append(cell_output)
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_2[win_step].append(cell_output)
        output_2 = []
        for win_step in range(num_LSTMs_2):
            output_2.append([])
            output_2[win_step] = tf.reshape(tf.concat(self.outputs_2[win_step], 1), [-1, size])

        with tf.variable_scope("Dep_Con_2"):
            temp_output_2 = []
            for win_step in range(num_LSTMs_2):
                temp_output_2.append([])
                temp_output_2[win_step] = tf.reshape(output_2[win_step],
                                                     [batch_size, num_steps - start_time_2, size])
                if win_step == 0:
                    input_2 = temp_output_2[win_step]
                else:
                    input_2 = tf.concat([input_2, temp_output_2[win_step]], 1)
            input_2 = tf.reshape(input_2, [batch_size, num_LSTMs_2, num_steps - start_time_2, size])
            concat_output_real_2 = tf.reduce_sum(input_2, 2)
            self.out_concat_output_real_2 = tf.reshape(concat_output_real_2, [batch_size, num_LSTMs_2 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_2

    def get_state(self):
        return self.state_2

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class TS_LSTM_Medium_3(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_3 = []
        self.state_3 = state_3 = []
        win_size_3 = win_size[3]  # 40 frames
        stride_3 = stride[3]  # df 35
        start_time_3 = start_time[3]  # 1
        num_LSTMs_3 = len(range(start_time_3, num_steps - win_size_3 + start_time_3, stride_3))
        print(range(start_time_3, num_steps - win_size_3 + start_time_3, stride_3))
        print("num_LSTMs_3: ", num_LSTMs_3)
        for time_step in range(start_time_3, num_steps):
            for win_step in range(num_LSTMs_3):
                if time_step == start_time_3:
                    self.outputs_3.append([])
                    state_3.append([])
                    state_3[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_3 + win_step * stride_3:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_3[win_step].append(cell_output)
                    elif time_step >= start_time_3 + win_step * stride_3 and time_step < start_time_3 + win_step * stride_3 + win_size_3:
                        if time_step > start_time_3 + win_step * stride_3: tf.get_variable_scope().reuse_variables()
                        distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_3, :]) / (
                            start_time_3 + 1)
                        (cell_output, state_3[win_step]) = cell(distance * 100, state_3[win_step])
                        self.outputs_3[win_step].append(cell_output)
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_3[win_step].append(cell_output)
        output_3 = []
        for win_step in range(num_LSTMs_3):
            output_3.append([])
            output_3[win_step] = tf.reshape(tf.concat(self.outputs_3[win_step], 1), [-1, size])

        with tf.variable_scope("Dep_Con_3"):
            temp_output_3 = []
            for win_step in range(num_LSTMs_3):
                temp_output_3.append([])
                temp_output_3[win_step] = tf.reshape(output_3[win_step],
                                                     [batch_size, num_steps - start_time_3, size])
                if win_step == 0:
                    input_3 = temp_output_3[win_step]
                else:
                    input_3 = tf.concat([input_3, temp_output_3[win_step]], 1)
            input_3 = tf.reshape(input_3, [batch_size, num_LSTMs_3, num_steps - start_time_3, size])
            concat_output_real_3 = tf.reduce_sum(input_3, 2)
            self.out_concat_output_real_3 = tf.reshape(concat_output_real_3, [batch_size, num_LSTMs_3 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_3

    def get_state(self):
        return self.state_3

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class TS_LSTM_Medium_4(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_4 = []
        self.state_4 = state_4 = []
        win_size_4 = win_size[4]  # 36 frames
        stride_4 = stride[4]  # df 35
        start_time_4 = start_time[4]  # 5
        num_LSTMs_4 = len(range(start_time_4, num_steps - win_size_4 + start_time_4, stride_4))
        print(range(start_time_4, num_steps - win_size_4 + start_time_4, stride_4))
        print("num_LSTMs_4: ", num_LSTMs_4)
        for time_step in range(start_time_4, num_steps):
            for win_step in range(num_LSTMs_4):
                if time_step == start_time_4:
                    self.outputs_4.append([])
                    state_4.append([])
                    state_4[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_4 + win_step * stride_4:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_4[win_step].append(cell_output)
                    elif time_step >= start_time_4 + win_step * stride_4 and time_step < start_time_4 + win_step * stride_4 + win_size_4:
                        if time_step > start_time_4 + win_step * stride_4: tf.get_variable_scope().reuse_variables()
                        distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_4, :]) / (
                            start_time_4 + 1)
                        (cell_output, state_4[win_step]) = cell(distance * 100, state_4[win_step])
                        self.outputs_4[win_step].append(cell_output)
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_4[win_step].append(cell_output)
        output_4 = []
        for win_step in range(num_LSTMs_4):
            output_4.append([])
            output_4[win_step] = tf.reshape(tf.concat(self.outputs_4[win_step], 1), [-1, size])

        with tf.variable_scope("Dep_Con_4"):
            temp_output_4 = []
            for win_step in range(num_LSTMs_4):
                temp_output_4.append([])
                temp_output_4[win_step] = tf.reshape(output_4[win_step],
                                                     [batch_size, num_steps - start_time_4, size])
                if win_step == 0:
                    input_4 = temp_output_4[win_step]
                else:
                    input_4 = tf.concat([input_4, temp_output_4[win_step]], 1)
            input_4 = tf.reshape(input_4, [batch_size, num_LSTMs_4, num_steps - start_time_4, size])
            concat_output_real_4 = tf.reduce_sum(input_4, 2)
            self.out_concat_output_real_4 = tf.reshape(concat_output_real_4, [batch_size, num_LSTMs_4 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_4

    def get_state(self):
        return self.state_4

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class TS_LSTM_Short_5(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_5 = []
        self.state_5 = state_5 = []
        win_size_5 = win_size[5]  # 15 frames
        stride_5 = stride[5]  # df 15
        start_time_5 = start_time[5]  # 1
        num_LSTMs_5 = len(range(start_time_5, num_steps - win_size_5 + start_time_5,
                                stride_5))  # range ( start, end, delta), len = 5
        print(range(start_time_5, num_steps - win_size_5 + start_time_5, stride_5))
        print("num_LSTMs_5: ", num_LSTMs_5)
        for time_step in range(start_time_5, num_steps):  # time_step is like i (1,76)
            for win_step in range(num_LSTMs_5):  # win_step (1,5)
                if time_step == start_time_5:
                    self.outputs_5.append([])
                    state_5.append([])
                    state_5[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_5 + win_step * stride_5:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)
                    elif time_step >= start_time_5 + win_step * stride_5 and time_step < start_time_5 + win_step * stride_5 + win_size_5:
                        if time_step > start_time_5 + win_step * stride_5: tf.get_variable_scope().reuse_variables()
                        distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_5, :]) / (
                            start_time_5 + 1)  # [batch_size, num_steps, feature_size]
                        (cell_output, state_5[win_step]) = cell(distance * 100, state_5[win_step])
                        self.outputs_5[win_step].append(cell_output)
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)
        output_5 = []
        for win_step in range(num_LSTMs_5):
            output_5.append([])
            output_5[win_step] = tf.reshape(tf.concat(self.outputs_5[win_step], 1), [-1, size])

        with tf.variable_scope("Dep_Con_5"):
            temp_output_5 = []
            for win_step in range(num_LSTMs_5):
                temp_output_5.append([])
                temp_output_5[win_step] = tf.reshape(output_5[win_step],
                                                     [batch_size, num_steps - start_time_5, size])
                if win_step == 0:
                    input_5 = temp_output_5[win_step]
                else:
                    input_5 = tf.concat([input_5, temp_output_5[win_step]], 1)
            input_5 = tf.reshape(input_5, [batch_size, num_LSTMs_5, num_steps - start_time_5, size])
            concat_output_real_5 = tf.reduce_sum(input_5, 2)
            self.out_concat_output_real_5 = tf.reshape(concat_output_real_5, [batch_size, num_LSTMs_5 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_5

    def get_state(self):
        return self.state_5

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class TS_LSTM_Motion_6(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size  # 20
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_5 = []
        self.state_5 = state_5 = []
        win_size_5 = win_size[5]  # 15 frames
        stride_5 = stride[5]  # df 15
        start_time_5 = start_time[5]  # 1
        num_LSTMs_5 = len(range(start_time_5, num_steps - win_size_5 + start_time_5,
                                stride_5))  # range ( start, end, delta), len = 5
        print(range(start_time_5, num_steps - win_size_5 + start_time_5, stride_5))
        print("num_LSTMs_5: ", num_LSTMs_5)
        for time_step in range(start_time_5, num_steps):  # time_step is like i (1,76)
            for win_step in range(num_LSTMs_5):  # win_step (1,5)
                if time_step == start_time_5:
                    self.outputs_5.append([])
                    state_5.append([])
                    state_5[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_5 + win_step * stride_5:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)
                    elif time_step >= start_time_5 + win_step * stride_5 and time_step < start_time_5 + win_step * stride_5 + win_size_5:
                        if time_step > start_time_5 + win_step * stride_5: tf.get_variable_scope().reuse_variables()
                        # distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_5, :]) / (
                        #     start_time_5 + 1)  # [batch_size, num_steps, feature_size]
                        distance = (inputs[:, time_step, :]) / (start_time_5 + 1)
                        (cell_output, state_5[win_step]) = cell(distance * 100, state_5[win_step])
                        self.outputs_5[win_step].append(cell_output)
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)
        output_5 = []
        for win_step in range(num_LSTMs_5):
            output_5.append([])
            output_5[win_step] = tf.reshape(tf.concat(self.outputs_5[win_step], 1), [-1, size])

        with tf.variable_scope("Dep_Con_6"):
            temp_output_5 = []
            for win_step in range(num_LSTMs_5):
                temp_output_5.append([])
                temp_output_5[win_step] = tf.reshape(output_5[win_step],
                                                     [batch_size, num_steps - start_time_5, size])
                if win_step == 0:
                    input_5 = temp_output_5[win_step]
                else:
                    input_5 = tf.concat([input_5, temp_output_5[win_step]], 1)
            input_5 = tf.reshape(input_5, [batch_size, num_LSTMs_5, num_steps - start_time_5, size])
            concat_output_real_5 = tf.reduce_mean(input_5, 2)
            self.out_concat_output_real_5 = tf.reshape(concat_output_real_5, [batch_size, num_LSTMs_5 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_5

    def get_state(self):
        return self.state_5

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.01
    learning_rate2 = 0.005
    learning_rate3 = 0.001
    learning_rate4 = 0.0005
    learning_rate5 = 0.0001
    learning_rate6 = 0.00005
    learning_rate7 = 0.00001
    # learning_rate = 0.1
    # learning_rate2 = 0.05
    # learning_rate3 = 0.01
    # learning_rate4 = 0.005
    # learning_rate5 = 0.001
    clearning_rate = 0.05
    clearning_rate2 = 0.01
    clearning_rate3 = 0.005
    clearning_rate4 = 0.001
    clearning_rate5 = 0.0005
    max_grad_norm = 5
    num_layers = 1
    num_steps = 114
    hidden_size1 = 10
    hidden_size2 = 30
    hidden_size = 60
    max_epoch1 = 1000
    max_epoch2 = 2000
    max_epoch3 = 4000
    max_epoch4 = 6000
    max_epoch5 = 7000
    max_epoch6 = 8000
    max_max_epoch = 1501
    # max_epoch1 = 1600
    # max_epoch2 = 3200
    # max_epoch3 = 6000
    # max_max_epoch = 30000
    keep_prob = 1.0
    lr_decay = 0.99
    batch_size = 442
    input_size = 105
    feature_size = 105
    ori_class_size = 20
    class_size = 11
    fusion1_size = 20
    fusion2_size = 20
    AS1 = [2, 3, 5, 6, 10, 13, 18, 20]
    AS2 = [1, 4, 7, 8, 9, 11, 12, 14]
    AS3 = [6, 14, 15, 16, 17, 18, 19, 20]
    use_batch_norm_rnn = False
    use_seq_wise = False
    use_batch_norm = False
    num_frames = num_steps * batch_size
    num_zeros = 0
    mode = 0


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.7
    batch_size = 20
    class_size = 21


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    class_size = 21


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 114
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.8
    batch_size = 1
    class_size = 11


def run_epoch(session, m, data, label, eval_op, verbose=False, is_training=True):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    # start_time = time.time()

    costs = 0.0
    iters = 0
    accuracys = 0.0

    sumsums = 0.0
    p_ls = []
    g_ls = []
    # avg_time = 0.0
    # cnt = 0
    for step, (x, y) in enumerate(Action_input.MSR_iterator(data, label, m.batch_size, m.feature_size,
                                                            m.num_steps, is_training)):
        # start_batch = time.time()
        cost_L, cost, accuracy, state, p_l, g_l, _ = session.run(
            [m.cost_L, m.cost, m.accuracy, m.final_state, m.pred_labels,
             m.given_labels, eval_op],
            {m.input_data: x,
             m.targets: y})
        costs += cost
        iters += m.num_steps
        accuracys += accuracy
        sumsums += 1
        for element in p_l:
            p_ls.append(element)
        for element in g_l:
            g_ls.append(element)


    return costs, accuracys / sumsums, p_ls, g_ls


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)



def main(_):
    config = get_config()
    config.class_size = 30
    config.feature_size = 45
    config.input_size = 45
    config.hidden_size = myHiddenSize
    config.keep_prob = myKeepProb

    eval_config = get_config()
    eval_config.class_size = 30
    eval_config.feature_size = 45
    eval_config.input_size = 45
    eval_config.hidden_size = myHiddenSize
    eval_config.keep_prob = myKeepProb

    eval_config2 = get_config()
    eval_config2.class_size = 30
    eval_config2.feature_size = 45
    eval_config2.input_size = 45
    eval_config2.hidden_size = myHiddenSize
    eval_config2.keep_prob = myKeepProb

    ####################################################################################################################
    DATA_PATH = os.path.join('Databases', 'UWA 3D Multiview Activity II Database')

    train_sklt0, train_label0 = Action_input.read(DATA_PATH, train_set, config)
    test_sklt0, test_label0 = Action_input.read(DATA_PATH, test_set1, eval_config)
    test_sklt02, test_label02 = Action_input.read(DATA_PATH, test_set2, eval_config)

    # for i in range(len(train_sklt0)):
    #     np.save("npy_saver/train_sklt0/%03d_train_sklt0.npy" % i, np.asarray(train_sklt0[i]))
    # print("Save Complete")

    MAX_LENGTH = 0
    for batchNo in range(len(train_sklt0)):
        if len(train_sklt0[batchNo]) > MAX_LENGTH:
            MAX_LENGTH = len(train_sklt0[batchNo])
        else:
            pass

    for batchNo in range(len(test_sklt0)):
        if len(test_sklt0[batchNo]) > MAX_LENGTH:
            MAX_LENGTH = len(test_sklt0[batchNo])
        else:
            pass

    for batchNo in range(len(test_sklt02)):
        if len(test_sklt02[batchNo]) > MAX_LENGTH:
            MAX_LENGTH = len(test_sklt02[batchNo])
        else:
            pass

    print(MAX_LENGTH)
    config.num_steps = MAX_LENGTH
    eval_config.num_steps = MAX_LENGTH
    eval_config2.num_steps = MAX_LENGTH

    train_sklt1 = feature_only_diff(train_sklt0, MAX_LENGTH, config)
    test_sklt1 = feature_only_diff(test_sklt0, MAX_LENGTH, eval_config)
    test_sklt12 = feature_only_diff(test_sklt02, MAX_LENGTH, eval_config2)


    # for i in range(len(train_sklt1)):
    #     np.save("temp_Data/%03d_train_sklt1.npy"%i,np.asarray(train_sklt1[i]) )

    del train_sklt0, test_sklt0, test_sklt02

    feature_train = Human_Cognitive_Coordinate(train_sklt1)
    feature_test = Human_Cognitive_Coordinate(test_sklt1)
    feature_test2 = Human_Cognitive_Coordinate(test_sklt12)
    AS_train_label = one_hot_labeling(train_label0, config)
    AS_test_label = one_hot_labeling(test_label0, eval_config)
    AS_test_label2 = one_hot_labeling(test_label02, eval_config2)

    del train_sklt1, test_sklt1, test_sklt12
    del train_label0, test_label0, test_label02

    print(feature_train.shape)
    print(feature_test.shape)
    print(feature_test2.shape)

    config.batch_size = np.int32(len(feature_train) / BatchDivider)  ### batch_modifier
    eval_config.batch_size = np.int32(len(feature_test))
    eval_config2.batch_size = np.int32(len(feature_test2))
    config.num_steps = np.int32(len(feature_train[0]))
    eval_config.num_steps = np.int32(len(feature_test[0]))
    eval_config2.num_steps = np.int32(len(feature_test2[0]))

    print(config.batch_size, eval_config.batch_size)
    print("Total Training Set Length : %d, Traning Batch Size : %d, Eval Batch Size : %d"
          % (len(feature_train), config.batch_size, eval_config.batch_size))

    # TODO=========================================================================================== SAVED FILE PATH CONFIG

    csv_suffix = strftime("_%Y%m%d_%H%M.csv", localtime())
    folder_path = os.path.join(myFolderPath)  # folder_modifier

    checkpoint_path = os.path.join(folder_path, "NTU_{0}.ckpt".format(view_subject))
    timecsv_path = os.path.join(folder_path, "Auto" + csv_suffix)

    f = open(timecsv_path, 'w')
    csvWriter = csv.writer(f)

    # TODO=========================================================================================== LOAD BALANCING


    # TODO=========================================================================================== SESSION CONFIG

    sessConfig = tf.ConfigProto(log_device_placement=False)
    sessConfig.gpu_options.allow_growth = True

    writeConfig_tocsv = True

    if writeConfig_tocsv:
        csvWriter.writerow(['DateTime:', strftime("%Y%m%d_%H:%M:%S", localtime())])
        csvWriter.writerow([])
        csvWriter.writerow(['Total Dataset Length', 'Train Batch Divider', 'Train Batch Size', 'Eval Batch Size', 'Eval Batch Size2', ])
        csvWriter.writerow(
            [len(feature_train), len(feature_train) / config.batch_size, config.batch_size, eval_config.batch_size, eval_config2.batch_size])

        csvWriter.writerow(['Control', 'Long 0'])
        csvWriter.writerow(['win_size', win_size[0]])
        csvWriter.writerow(['stride', stride[0]])
        csvWriter.writerow(
            ['start_time', start_time[0]])
        csvWriter.writerow([])

    # TODO=========================================================================================== BUILD GRAPH
    with tf.Graph().as_default(), tf.Session(config=sessConfig) as session:
        with tf.device('/cpu:0'):
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = MP_runner(is_training=True, config=config, labels=AS_train_label)
            print("\nTraining Model Established!\n")

            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = MP_runner(is_training=False, config=eval_config, labels=AS_test_label)

            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest2 = MP_runner(is_training=False, config=eval_config2, labels=AS_test_label2)

            print("\nTesting Model Established!!\n")

            # summary_writer = tf.train.SummaryWriter('/home/inwoong/MSR_logs', graph=session.graph)

            init = tf.global_variables_initializer()  # TF ver 0.11
            # init = tf.global_variables_initializer() #TF ver 0.12
            session.run(init)

            saver = tf.train.Saver(tf.global_variables())

            # saver.restore(session, "./result_apply_170116/rNTU_view.ckpt-6000")
            print("Model restored.")

            stt_loop = time.time()
            print(strftime("%Y%m%d_%H:%M:%S", localtime()))

            csvWriter.writerow(['Time', 'Epoch #', 'Epoch Time', 'Train Accuracy', 'Train Cost'])
            for i in range(config.max_max_epoch):

                stt_lr = time.time()

                if i == 0:
                    print("First Learning Rate is assigned!!")
                    m.assign_lr(session, config.learning_rate)
                elif i == config.max_epoch1:
                    m.assign_lr(session, config.learning_rate2)
                elif i == config.max_epoch2:
                    m.assign_lr(session, config.learning_rate3)
                elif i == config.max_epoch3:
                    m.assign_lr(session, config.learning_rate4)
                elif i == config.max_epoch4:  # 6000
                    print("6000 Learning Rate is assigned!!")
                    m.assign_lr(session, config.learning_rate5)
                elif i == config.max_epoch5:  # 10,000
                    m.assign_lr(session, config.learning_rate6)
                elif i == config.max_epoch6:  # 10,000
                    m.assign_lr(session, config.learning_rate7)

                stt_epoch = time.time()

                if i == 0:
                    print("I'm Ready for First Epoch")
                train_cost, train_accuracy, tr_p_l, tr_g_l = run_epoch(
                    session, m, feature_train, AS_train_label,
                    m.train_op,
                    verbose=True)

                end_epoch = time.time()
                assert not np.isnan(train_cost), 'Model diverged with loss = NaN'
                # if (i % 10) == 0:
                #     feed_tr = {m.input_data: feature_train[0 * config.batch_size:(0 + 1) * config.batch_size, :, :],
                #                m.targets: AS_train_label[0 * config.batch_size:(0 + 1) * config.batch_size, :]}
                #     logits = session.run(m.logits, feed_dict=feed_tr)
                #     print(logits)
                #     summary_str_tr = session.run(m.summary_op, feed_dict=feed_tr)
                #     summary_writer.add_summary(summary_str_tr, i)

                # Save the model checkpoint periodically.
                if i % 100 == 0 or (i + 1) == config.max_max_epoch:
                    # checkpoint_path = os.path.join("./view_model1+b989+h70", "NTU_view_TS-LSTM.ckpt")
                    saver.save(session, checkpoint_path, global_step=i)

                if i % 10 == 0:
                    end_loop = time.time()
                    strtime = strftime("%Y%m%d_%H:%M:%S", localtime())
                    print(strtime)
                    print("----------Epoch Time: %.3f, per Assign: %.3f, per Epoch: %.3f" % (
                        (end_loop - stt_loop), (stt_epoch - stt_lr), (end_epoch - stt_epoch)))
                    print("Epoch: %d Learning rate: %.6f Train Accuracy: %.4f" % (i, session.run(m.lr), train_accuracy))
                    # train_cost = train_cost * config.batch_size / len(feature_train)
                    print("Train Cost: %.6f" % (
                        train_cost))
                    stt_loop = time.time()
                    print("\n")

                    csvWriter.writerow([strtime, i, (end_epoch - stt_epoch), train_accuracy, train_cost])

                if i % 100 == 0:
                    test_cost, test_accuracy, te_p_l, te_g_l = run_epoch(session, mtest, feature_test,
                                                                                      AS_test_label,
                                                                                      tf.no_op(),
                                                                                      is_training=False)
                    test_cost2, test_accuracy2, te_p_l2, te_g_l2 = run_epoch(session, mtest2, feature_test2,
                                                                                      AS_test_label2,
                                                                                      tf.no_op(),
                                                                                      is_training=False)
                    print("Test Accuracy: %.5f %.5f\n" % (test_accuracy, test_accuracy2))
                    csvWriter.writerow(["Test Accuracy :", test_accuracy, test_accuracy2])

                    confusion_matrix = np.zeros([config.class_size, config.class_size + 1])
                    class_prob = np.zeros([config.class_size])
                    for j in range(len(te_g_l)):
                        confusion_matrix[te_g_l[j]][te_p_l[j]] += 1
                    for j in range(config.class_size):
                        class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:config.class_size])
                    for j in range(config.class_size):
                        confusion_matrix[j][config.class_size] = class_prob[j]

                    confusion_matrix2 = np.zeros([config.class_size, config.class_size + 1])
                    class_prob2 = np.zeros([config.class_size])
                    for j in range(len(te_g_l2)):
                        confusion_matrix2[te_g_l2[j]][te_p_l2[j]] += 1
                    for j in range(config.class_size):
                        class_prob2[j] = confusion_matrix2[j][j] / np.sum(confusion_matrix2[j][0:config.class_size])
                    for j in range(config.class_size):
                        confusion_matrix2[j][config.class_size] = class_prob2[j]

                    with open(folder_path + "/view-test-" + str(i) + ".csv", "w") as csvfile:
                        csvwriter2 = csv.writer(csvfile)
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix[j])

                    with open(folder_path + "/view-test2-" + str(i) + ".csv", "w") as csvfile:
                        csvwriter2 = csv.writer(csvfile)
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix2[j])

    f.close()

if __name__ == "__main__":
    tf.app.run()