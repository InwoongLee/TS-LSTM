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

from PreProcesses import body_rotation, bone_length, extract_AS, extract_data_label, extract_feature, count_zeros, feature_only_diff_0, Pose_Motion, one_hot_labeling

# from ops import batch_norm_rnn, batch_norm

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

# win_size = [int(sys.argv[1])]
# stride = [int(sys.argv[2])]


N_S = 5

_now = datetime.now()
BatchDivider = 30
myKeepProb = 0.4
myHiddenSize = 100
view_subject = 'Generalized_TS-LSTM'
myFolderPath = './TrialSaver/%s/%04d%02d%02d_%02d%02d_BD_%d_KB_%.2f_NS_%02d'%(view_subject,_now.year,
                                                                           _now.month,
                                                                           _now.day,
                                                                           _now.hour,
                                                                           _now.minute,
                                                                           BatchDivider,
                                                                           myKeepProb, N_S)
os.mkdir(myFolderPath)

import random

# from ops import batch_norm_rnn, batch_norm

flags = tf.flags
logging = tf.logging

# win_size_data = [120, 200, 20, 40, 120, 60, 80, 180, 120, 200]
# stride_data = [80, 100, 20, 40, 80, 46, 60, 20, 80, 100]
# start_time_data = [2, 6, 2, 8, 3, 4, 6, 2, 1, 2]

# win_size_data = [200, 120, 180, 60, 20, 80, 200, 120, 40, 120]
# stride_data = [100, 80, 20, 46, 20, 60, 100, 80, 40, 80]
# start_time_data = [1, 3, 4, 4, 1, 2, 5, 3, 1, 7]

# win_size_data = [80, 180, 180, 40, 20, 80, 160, 140, 200, 140]
# stride_data = [60, 20, 20, 40, 20, 60, 40, 60, 100, 60]
# start_time_data = [4, 5, 4, 10, 1, 2, 8, 2, 2, 10]

# win_size_data = [200, 100, 180, 160, 20, 80, 160, 140, 200, 140]
# stride_data = [100, 100, 20, 40, 20, 60, 40, 60, 100, 60]
# start_time_data = [1, 1, 4, 7, 1, 2, 8, 2, 7, 10]

# win_size_data = [200, 40, 20, 80, 160, 140, 160]
# stride_data = [100, 40, 20, 60, 40, 60, 40]
# start_time_data = [2, 9, 1, 2, 8, 2, 9]

# win_size_data = [20, 80, 160, 140, 160, 40, 200]
# stride_data = [20, 60, 40, 60, 40, 40, 100]
# start_time_data = [1, 2, 8, 2, 9, 9, 2]

win_size_data = [20, 80, 160, 140, 160, 40, 200]
stride_data = [20, 60, 40, 60, 40, 40, 100]
start_time_data = [1, 2, 8, 2, 9, 9, 2]

# win_size_data = [60, 80, 100, 160, 160, 180, 200]
# stride_data = [46, 60, 100, 40, 40, 20, 100]
# start_time_data = [1, 2, 8, 8, 9, 2, 2]


win_size = win_size_data[0:N_S]
stride = stride_data[0:N_S]
start_time = start_time_data[0:N_S]



def getDevice():
    return FLAGS.num_gpus


class MP_runner(object):

    #logits_0, logits_1, logits_2, cost, accuracy, final_state, pred_labels, given_labels, eval_op, {m.input_data: x, m.targets: y}

    """The MSR model."""

    def __init__(self, is_training, config, labels, win_sizes, strides, start_times):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size # 20

        self.data_initializer = tf.placeholder(tf.float32, [batch_size, num_steps, feature_size], name="x-input")
        self.label_initializer = tf.placeholder(tf.float32, [batch_size, class_size], name="y-input")

        self.input_data = tf.Variable(self.data_initializer, trainable=False, collections=[], name="x-top")
        self.targets = tf.Variable(self.label_initializer, trainable=False, collections=[], name="y-top")

        self.mL = []
        self.depthconcat = []
        self._logits_0 = []
        self._cross_entropy_0 = []
        self._final_state = []
        self._cost_L = []
        self.tvars_L = []
        self.grad_before_sum_L = []
        for wi in range(len(win_sizes)):
            self.mL.append([])
            self.depthconcat.append([])
            self._logits_0.append([])
            self._cross_entropy_0.append([])
            self._final_state.append([])
            self._cost_L.append([])
            self.tvars_L.append([])
            self.grad_before_sum_L.append([])
            if is_training:
                sw_general = '/gpu:0'
            else:
                sw_general = '/cpu:0'
            with tf.device(sw_general):
                with tf.name_scope('%s_%d' % ('mGPU', wi)) as scope:
                    self.mL[wi] = self.mL[wi] = TS_Long_LSTM_Top(is_training, config, labels, self.input_data, self.targets, size,
                                               win_sizes[wi], strides[wi], start_times[wi], sw_general, wi)
                    self._logits_0[wi], self._cross_entropy_0[wi], self._final_state[wi] = self.mL[wi].get_inference()
                    self._cost_L[wi] = tf.reduce_sum(self._cross_entropy_0[wi] / len(win_sizes) ) / batch_size

                    self.depthconcat[wi] = self.mL[wi].get_depth_concat_output()

                if is_training:
                    if wi == 0:
                        self.tvars_L[wi] = tf.trainable_variables()
                        count = len(self.tvars_L[wi])
                    else:
                        temp_tvars = tf.trainable_variables()
                        self.tvars_L[wi] = temp_tvars[count:len(temp_tvars)]
                        count = len(temp_tvars)
                    print("number of LSTM parameters : ", len(self.tvars_L[wi]))
                    print(self.tvars_L[wi])
                    self.grad_before_sum_L[wi] = tf.gradients(self._cost_L[wi], self.tvars_L[wi])

        with tf.device(sw_general):

            for ci in range(len(self._cost_L)):
                if ci ==0:
                    self._cost = self._cost_L[ci]
                else:
                    self._cost = self._cost + self._cost_L[ci]

            if not is_training:
                with tf.name_scope("Accuracy") as scope:
                    for ri in range(len(self._logits_0)):
                        if ri == 0:
                            real_logits = self._logits_0[ri] / len(self._logits_0)
                        else:
                            real_logits = real_logits + self._logits_0[ri]/len(self._logits_0)
                    for ri in range(len(self._logits_0)):
                        if ri == 0:
                            real_logits_geo = tf.log(self._logits_0[ri]) / len(self._logits_0)
                        else:
                            real_logits_geo = real_logits_geo + tf.log(self._logits_0[ri]) / len(self._logits_0)
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self._correct_prediction_geo = tf.equal(tf.argmax(real_logits_geo, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self.pred_labels_geo = tf.argmax(real_logits_geo, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size
                    self._accuracy_geo = tf.reduce_sum(tf.cast(self._correct_prediction_geo, tf.float32)) / batch_size
            else:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = 0
                    for ri in range(len(self._logits_0)):
                        real_logits = real_logits + self._logits_0[ri] / len(self._logits_0)
                    for ri in range(len(self._logits_0)):
                        if ri == 0:
                            real_logits_geo = tf.log(self._logits_0[ri]) / len(self._logits_0)
                        else:
                            real_logits_geo = real_logits_geo + tf.log(self._logits_0[ri]) / len(self._logits_0)
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self._correct_prediction_geo = tf.equal(tf.argmax(real_logits_geo, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self.pred_labels_geo = tf.argmax(real_logits_geo, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size
                    self._accuracy_geo = tf.reduce_sum(tf.cast(self._correct_prediction_geo, tf.float32)) / batch_size

        if is_training:
            with tf.device('/gpu:0'):
                with tf.name_scope("train") as scope:
                    with tf.name_scope("Merging_Gradient"):
                        for gi in range(len(self.grad_before_sum_L)):
                            sys.stdout.write("%d : %d" % (gi, len(self.grad_before_sum_L[gi])))
                        print("")

                        for gi in range(len(self.grad_before_sum_L)):
                            if gi == 0:
                                self.grad_after_sum = self.grad_before_sum_L[gi]
                            else:
                                self.grad_after_sum = self.grad_after_sum + self.grad_before_sum_L[gi]
                        print("All gradient sum : ", len(self.grad_after_sum))
                        self._lr = tf.Variable(0.0, trainable=False)
                        self.grads, _ = tf.clip_by_global_norm(self.grad_after_sum, config.max_grad_norm)
                        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                        self.tvars = tf.trainable_variables()
                        with tf.name_scope("Applying-Gradient"):
                            print("List : ",self.grads)
                            self._train_op = self.optimizer.apply_gradients(zip(self.grads,self.tvars))

    def assign_lr(self, sess, lr_value):
        sess.run(tf.assign(self.lr, lr_value))

    @property
    def initial_state_L(self):
        return self.mL.initial_state

    @property
    def logits_0(self):
        return self._logits_0

    def get_depth_concat_output(self):
        return self.depthconcat

    @property
    def cost_L(self):
        return self._cost_L

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
    def accuracy_geo(self):
        return self._accuracy_geo

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

    def __init__(self, is_training, config, labels, top_input_data, top_targets, top_hidden_size, win_size_0, stride_0, start_time_0, sw_general, wi):

        with tf.device(sw_general):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps  # 76
            self.class_size = class_size = config.class_size
            size = top_hidden_size  # 20

            self.input_data = top_input_data
            self.targets = top_targets

            num_LSTMs_0 = len(range(0, num_steps - win_size_0 + start_time_0, stride_0))
            # num_LSTMs_2 = len(range(start_time[2], num_steps - win_size[2] + start_time[2], stride[2]))

        # self.scale_intput = scale_intput = tf.Variable(80.0, name="scale_intput")
        with tf.variable_scope('Type%d' % wi):
            with tf.device(sw_general):
                with tf.name_scope('%s_%d' % ('GPU', wi)) as scope:
                    with tf.variable_scope('l0'):
                        self.mL0 = mL0 = TS_LSTM_Long_0(is_training, config, self.input_data, win_size_0, stride_0, start_time_0)
                        self._initial_state = mL0.initial_state
                        output_depthconcat_long_0 = mL0.get_depth_concat_output()
                        self.output_depthconcat_long_0 = output_depthconcat_long_0


            with tf.device(sw_general):
                with tf.variable_scope("Concat_0"):
                    output_real_0 = output_depthconcat_long_0

                with tf.variable_scope("Drop_0"):
                    if is_training and config.keep_prob < 1:
                        output_real_0 = tf.nn.dropout(output_real_0, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_0 = output_real_0 * config.keep_prob

                with tf.variable_scope("Softmax_0"):
                    self.softmax_w_0 = softmax_w_0 = tf.get_variable("softmax_w_0", [num_LSTMs_0 * size, class_size])
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

    def get_depth_concat_output(self):
        return self.output_depthconcat_long_0

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



class TS_LSTM_Long_0(object):
    def __init__(self, is_training, config, input_data, win_size_0, stride_0, start_time_0):
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
        num_LSTMs_0 = len(range(0, num_steps - win_size_0 + start_time_0, stride_0))
        print(range(0, num_steps - win_size_0 + start_time_0, stride_0))
        print("num_LSTMs_0: ", num_LSTMs_0)
        for time_step in range(num_steps):
            for win_step in range(num_LSTMs_0):
                if time_step == 0:
                    self.outputs_0.append([])
                    state_0.append([])
                    state_0[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < win_step * stride_0:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_0[win_step].append(cell_output)
                    elif time_step >= win_step * stride_0 and time_step < start_time_0 + win_step * stride_0 + win_size_0:
                        if time_step > win_step * stride_0: tf.get_variable_scope().reuse_variables()
                        if time_step < start_time_0 + win_step * stride_0:
                            distance = tf.reduce_sum(inputs[:, 0:time_step+1, :], 1) / (start_time_0 + 1)
                        else:
                            if start_time_0 == 1:
                                distance = inputs[:, time_step, :] / (start_time_0 + 1)
                            else:
                                distance = tf.reduce_sum(inputs[:, time_step - start_time_0:time_step+1, :], 1) / (start_time_0 + 1)
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
                temp_output_0[win_step] = tf.reshape(output_0[win_step], [batch_size, num_steps, size])
                if win_step == 0:
                    input_0 = temp_output_0[win_step]
                else:
                    input_0 = tf.concat([input_0, temp_output_0[win_step]], 1)
            input_0 = tf.reshape(input_0, [batch_size, num_LSTMs_0, num_steps, size])
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

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.1
    learning_rate2 = 0.05
    learning_rate3 = 0.01
    learning_rate4 = 0.005
    learning_rate5 = 0.001
    learning_rate6 = 0.0005
    learning_rate7 = 0.0001
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
    max_epoch1 = 1001
    max_epoch2 = 2000
    max_epoch3 = 4000
    max_epoch4 = 6000
    max_epoch5 = 7000
    max_epoch6 = 8000
    max_max_epoch = 501
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
    #start_time = time.time()

    costs = 0.0
    costs_L = np.zeros_like([len])
    iters = 0
    accuracys = 0.0
    accuracys_geo = 0.0

    sumsums = 0.0
    p_ls = []
    p_ls_geo = []
    g_ls = []
    #avg_time = 0.0
    cnt = 0
    for step, (x, y) in enumerate(Action_input.MSR_iterator(data, label, m.batch_size, m.feature_size,
                                                            m.num_steps, is_training)):
        # start_batch = time.time()
        cost_L, cost, accuracy, accuracy_geo, state, p_l, p_l_geo, g_l, _ = session.run([m.cost_L, m.cost, m.accuracy, m.accuracy_geo, m.final_state, m.pred_labels, m.pred_labels_geo,
                                                          m.given_labels, eval_op],
                                                         {m.input_data: x,
                                                          m.targets: y})
        costs += cost
        if cnt == 0:
            costs_L = cost_L
        else:
            for ii in range(len(cost_L)):
                costs_L[ii] = costs_L[ii] + cost_L[ii]
        iters += m.num_steps
        accuracys += accuracy
        accuracys_geo += accuracy_geo
        sumsums += 1
        cnt += 1
        for element in p_l:
            p_ls.append(element)
        for element2 in p_l_geo:
            p_ls_geo.append(element2)
        for element3 in g_l:
            g_ls.append(element3)
        # end_batch = time.time()
        # avg_time = avg_time + (end_batch - start_batch)
        # cnt = cnt + 1
        # if cnt % 10 == 1:
        #     print ("Batch %d_%d: %.3f" % (m.batch_size, cnt, avg_time / cnt))

    #print ("Batch %d_%d: %.3f" % (m.batch_size, cnt, avg_time/cnt))

    # print(sumsums)
    # summary_strs = np.append(summary_strs, summary_str)

    return costs_L, costs, accuracys / sumsums, accuracys_geo / sumsums, p_ls, p_ls_geo, g_ls

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
    config.class_size = 10
    config.feature_size = 60
    config.input_size = 60
    config.hidden_size = myHiddenSize
    config.keep_prob = myKeepProb

    eval_config = get_config()
    eval_config.class_size = 10
    eval_config.feature_size = 60
    eval_config.input_size = 60
    eval_config.hidden_size = myHiddenSize
    eval_config.keep_prob = myKeepProb

    DATA_PATH = os.path.join('NWUCLA_csv')

    train_set = [1, 2]
    test_set = [3]

    train_sklt0, train_label0 = Action_input.read(DATA_PATH, train_set, config)
    test_sklt0, test_label0 = Action_input.read(DATA_PATH, test_set, eval_config)

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

    print(MAX_LENGTH)
    config.num_steps = MAX_LENGTH
    eval_config.num_steps = MAX_LENGTH

    train_sklt1 = feature_only_diff_0(train_sklt0, MAX_LENGTH, config)
    test_sklt1 = feature_only_diff_0(test_sklt0, MAX_LENGTH, eval_config)

    train_sklt2 = body_rotation(train_sklt1)
    test_sklt2 = body_rotation(test_sklt1)

    # for i in range(len(train_sklt1)):
    #     np.save("temp_Data/%03d_train_sklt1.npy"%i,np.asarray(train_sklt1[i]) )

    del train_sklt0, test_sklt0

    feature_train = Pose_Motion(train_sklt2)
    feature_test = Pose_Motion(test_sklt2)
    AS_train_label = one_hot_labeling(train_label0, config)
    AS_test_label = one_hot_labeling(test_label0, eval_config)

    del train_sklt2, test_sklt2
    del train_sklt1, test_sklt1, train_label0, test_label0

    print(feature_train.shape)
    print(feature_test.shape)


    # feature_train = feature_train[0:60,:,:]
    # feature_test = feature_test[0:60,:,:]

    config.batch_size = np.int32(len(feature_train) / BatchDivider)  ### batch_modifier
    eval_config.batch_size = np.int32(len(feature_test))
    config.num_steps = np.int32(len(feature_train[0]))
    eval_config.num_steps = np.int32(len(feature_test[0]))

    print(config.batch_size, eval_config.batch_size)
    print ("Total Training Set Length : %d, Traning Batch Size : %d, Eval Batch Size : %d"
           % (len(feature_train),config.batch_size,eval_config.batch_size))

    #TODO=========================================================================================== SAVED FILE PATH CONFIG

    csv_suffix = strftime("_%Y%m%d_%H%M.csv", localtime())
    folder_path = os.path.join(myFolderPath) #folder_modifier

    checkpoint_path = os.path.join(folder_path, "NTU_{0}.ckpt".format(view_subject))
    timecsv_path = os.path.join(folder_path, "Auto" + csv_suffix)

    f = open(timecsv_path,'w')
    csvWriter = csv.writer(f)

    #TODO=========================================================================================== SESSION CONFIG

    sessConfig = tf.ConfigProto(log_device_placement=False)
    sessConfig.gpu_options.allow_growth = True

    writeConfig_tocsv = True

    if writeConfig_tocsv:
        csvWriter.writerow(['DateTime:', strftime("%Y%m%d_%H:%M:%S", localtime())])
        csvWriter.writerow([])
        csvWriter.writerow(['Total Dataset Length', 'Train Batch Divider', 'Train Batch Size', 'Eval Batch Size', 'Dropout prob', ])
        csvWriter.writerow(
            [len(feature_train), len(feature_train) / config.batch_size, config.batch_size, eval_config.batch_size, myKeepProb])
        # csvWriter.writerow(
        #     ['win_size', win_size[0], win_size[1], win_size[2], win_size[3], win_size[4], win_size[5], win_size[6],
        #      win_size[7], win_size[8], win_size[9]])
        # csvWriter.writerow(
        #     ['stride', stride[0], stride[1], stride[2], stride[3], stride[4], stride[5], stride[6], stride[7],
        #      stride[8], stride[9]])
        # csvWriter.writerow(
        #     ['start_time', start_time[0], start_time[1], start_time[2], start_time[3], start_time[4], start_time[5],
        #      start_time[6], start_time[7], start_time[8], start_time[9]])
        csvWriter.writerow([])
        csvWriter.writerow([win_size_data])
        csvWriter.writerow([stride_data])
        csvWriter.writerow([start_time_data])
        csvWriter.writerow([])

    #TODO=========================================================================================== BUILD GRAPH
    with tf.Graph().as_default(), tf.Session(config=sessConfig) as session:
        with tf.device('/cpu:0'):
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = MP_runner(is_training=True, config=config, labels=AS_train_label, win_sizes=win_size, strides = stride, start_times = start_time)
            print("\nTraining Model Established!\n")

            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, win_sizes=win_size, strides = stride, start_times = start_time)

            print("\nTesting Model Established!!\n")

            # summary_writer = tf.train.SummaryWriter('/home/inwoong/MSR_logs', graph=session.graph)

            init = tf.global_variables_initializer() #TF ver 0.11
            #init = tf.global_variables_initializer() #TF ver 0.12
            session.run(init)

            saver = tf.train.Saver(tf.global_variables())

            #saver.restore(session, "./result_apply_170116/rNTU_view.ckpt-6000")
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
                elif i == config.max_epoch4: #6000
                    print("6000 Learning Rate is assigned!!")
                    m.assign_lr(session, config.learning_rate5)
                elif i == config.max_epoch5: #10,000
                    m.assign_lr(session, config.learning_rate6)
                elif i == config.max_epoch6: #10,000
                    m.assign_lr(session, config.learning_rate7)

                stt_epoch = time.time()

                if i == 0:
                    print("I'm Ready for First Epoch")
                train_cost_L, train_cost, train_accuracy, train_accuracy_geo, tr_p_l, tr_p_l_geo, tr_g_l = run_epoch(
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
                    strtime =  strftime("%Y%m%d_%H:%M:%S", localtime())
                    print(strtime)
                    print("----------Epoch Time: %.3f, per Assign: %.3f, per Epoch: %.3f" % (
                    (end_loop - stt_loop), (stt_epoch - stt_lr), (end_epoch - stt_epoch)))
                    print("Epoch: %d Learning rate: %.6f Train Accuracy_AM: %.4f Train Accuracy_GM: %.4f" % (i, session.run(m.lr), train_accuracy, train_accuracy_geo))
                    # train_cost = train_cost * config.batch_size / len(feature_train)
                    sys.stdout.write("Train Cost: %.6f " % train_cost)
                    for ci in range(len(train_cost_L)):
                        sys.stdout.write("Cost_%d: % .6f " % (ci, train_cost_L[ci]))
                    print("")

                    stt_loop = time.time()
                    print("\n")

                    csvWriter.writerow([strtime, i, (end_epoch - stt_epoch), train_accuracy, train_accuracy_geo, train_cost])

                if i % 100 == 0:
                    test_cost_L, test_cost, test_accuracy, test_accuracy_geo, te_p_l, te_p_l_geo, te_g_l = run_epoch(session,
                                                                                                                mtest,
                                                                                                                feature_test,
                                                                                                                AS_test_label,
                                                                                                                tf.no_op(), is_training=False)
                    print("Test Accuracy_AM: %.5f Test Accuracy_GM : %.5f\n" % (test_accuracy, test_accuracy_geo))
                    csvWriter.writerow(["Test Accuracy_AM :", test_accuracy, "Test Accuracy_GM :", test_accuracy_geo])

                    confusion_matrix = np.zeros([config.class_size, config.class_size + 1])
                    class_prob = np.zeros([config.class_size])
                    for j in range(len(te_g_l)):
                        confusion_matrix[te_g_l[j]][te_p_l[j]] += 1
                    for j in range(config.class_size):
                        class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:config.class_size])

                    confusion_matrix_geo = np.zeros([config.class_size, config.class_size + 1])
                    class_prob_geo = np.zeros([config.class_size])
                    for j in range(len(te_g_l)):
                        confusion_matrix_geo[te_g_l[j]][te_p_l_geo[j]] += 1
                    for j in range(config.class_size):
                        class_prob_geo[j] = confusion_matrix_geo[j][j] / np.sum(confusion_matrix_geo[j][0:config.class_size])
                    for j in range(config.class_size):
                        confusion_matrix[j][config.class_size] = class_prob[j]
                        confusion_matrix_geo[j][config.class_size] = class_prob_geo[j]


                    with open(folder_path + "/view-test-AM-" + str(i) + ".csv", "w") as csvfile:
                        csvwriter2 = csv.writer(csvfile)
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix[j])
                    with open(folder_path + "/view-test-GM-" + str(i) + ".csv", "w") as csvfile_geo:
                        csvwriter2_geo = csv.writer(csvfile_geo)
                        for j in range(config.class_size):
                            csvwriter2_geo.writerow(confusion_matrix_geo[j])

    f.close()


if __name__ == "__main__":
    tf.app.run()