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

import Action_input
import csv

import random

# from ops import batch_norm_rnn, batch_norm

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

tf.app.flags.DEFINE_integer('num_gpus', 4, """How many GPUs to use""")

# win_size = [154, 150, 145, 103, 99, 77]
# stride = [145, 145, 145, 98, 98, 74]
# start_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

win_size = [210]
stride = [90]
start_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

"""
win_size = [154, 150, 145, 103, 99, 77]
stride = [145, 145, 145, 98, 98, 74]
start_time = [1, 5, 10, 1, 5, 1]
"""

#computaion_scale = [508, 500, 490, 46, 58, 14]

#TODO=========================================================================================== Assignment
############## Long Top, Medeum Top, Short Top, Ensemble
#runner_assign = ['/gpu:0', '/gpu:1', '/gpu:0', '/gpu:1']

################### LT pre , Long 0 , Long 1 , Long 2 , LT post
#Top_long_assign = ['/gpu:0', '/gpu:0', '/gpu:1', '/gpu:0', '/gpu:0']

##################### MT pre, Medium 3 , Medium 4, MT post
#Medium_long_assign = ['/gpu:1', '/gpu:1', '/gpu:1', '/gpu:1']

##################### ST pre , Short 5 , ST post
#Short_long_assign = ['/gpu:0', '/gpu:0', '/gpu:0']

gradient_device = ['/gpu:0','/gpu:1','/gpu:2','/gpu:3']

_now = datetime.now()
BatchDivider = 60
myKeepProb = 0.4
folder_name = 'motion_diff'
view_subject = 'subject'

myFolderPath = './TrialSaver/%s/%04d%02d%02d_%02d%02d_BD_%d_KB_%.2f_WS_%02d_ST_%02d'%(folder_name,_now.year,
                                                                           _now.month,
                                                                           _now.day,
                                                                           _now.hour,
                                                                           _now.minute,
                                                                           BatchDivider,
                                                                           myKeepProb, win_size[0], stride[0])
os.mkdir(myFolderPath)


def getDevice():
    return FLAGS.num_gpus


class MP_runner(object):

    #logits_0, logits_1, logits_2, cost, accuracy, final_state, pred_labels, given_labels, eval_op, {m.input_data: x, m.targets: y}

    """The MSR model."""

    def __init__(self, is_training, config, labels, motion_diff):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps # 76
        self.class_size = class_size = config.class_size
        size = config.hidden_size # 20

        counter = 0

        self.data_initializer = tf.placeholder(tf.float32, [batch_size, num_steps, feature_size], name="x-input")
        self.label_initializer = tf.placeholder(tf.float32, [batch_size, class_size], name="y-input")

        self.input_data = tf.Variable(self.data_initializer, trainable=False, collections=[], name="x-top")
        self.targets = tf.Variable(self.label_initializer, trainable=False, collections=[], name="y-top")

        if is_training:
            sw_0 = runner_assign[0]
            sw_3 = runner_assign[3]
        else:
            sw_0 = sw_1 = sw_2 = sw_3 = '/cpu:0'

        with tf.device(sw_0):
            with tf.name_scope('%s_%d' % ('mGPU', 0)) as scope:
                self.mL = mL = TS_Long_LSTM_Top(is_training, config, labels, motion_diff, self.input_data, self.targets, size)
                logits_l, cross_entropy_l, state_0 = mL.get_inference()
                self._logits_0, self._cross_entropy_0, self._final_state = \
                    logits_0, cross_entropy_0, final_state = logits_l, cross_entropy_l, state_0

                self._cost_L = tf.reduce_sum(cross_entropy_l) / batch_size

                if is_training:
                    self.tvars_L = tf.trainable_variables()
                    count = len(self.tvars_L)
                    print("Long : ",len(self.tvars_L))
                    print(self.tvars_L)
                    self.grad_before_sum_L = tf.gradients(self._cost_L, self.tvars_L)

        with tf.device(sw_3):
            print "Parallelized Model is on building!!"

            self._cost = self._cost_L

            # with tf.name_scope("Cost") as scope:
            #     cross_entropy = (cross_entropy_l + cross_entropy_m + cross_entropy_s) / 3
            #     self._cost = cost = tf.reduce_sum(cross_entropy) / batch_size
            # self._final_state = state_0

            if not is_training:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = logits_0
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size
            else:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = logits_l
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size


        if is_training:
            with tf.device(runner_assign[3]):
                with tf.name_scope("train") as scope:
                    with tf.name_scope("Merging_Gradient"):
                        print("L : ", len(self.grad_before_sum_L))
                        self.grad_after_sum = self.grad_before_sum_L
                        print("L : ", len(self.grad_after_sum))
                        self._lr = tf.Variable(0.0, trainable=False)
                        self.grads, _ = tf.clip_by_global_norm(self.grad_after_sum, config.max_grad_norm)
                        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                        self.tvars = tf.trainable_variables()
                        # self.tvars_sum = self.tvars_L + self.tvars_M + self.tvars_S
                        # with open("grad_after_sum.txt", "a") as f:
                        #     f.write("%d %d\n"%(len(self.tvars_sum), len(self.grad_after_sum)))
                        #     f.write(str(self.tvars_sum))
                        #     f.write("\n")
                        #     f.write(str(self.grad_after_sum))
                        #     f.write("\n")
                        with tf.name_scope("Applying-Gradient"):
                            print("List : ",self.grads)
                            self._train_op = self.optimizer.apply_gradients(zip(self.grads,self.tvars))

        print "Calculating Graph is fully connected!!"

    def assign_lr(self, sess, lr_value):
        sess.run(tf.assign(self.lr, lr_value))

    def mp_init(self,sess):
        self.mL.init_all_var(sess)

    @property
    def initial_state_L(self):
        return self.mL.initial_state

    @property
    def logits_0(self):
        return self._logits_0

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

    def __init__(self, is_training, config, labels, motion_diff, top_input_data, top_targets, top_hidden_size):
        if is_training:
            sw_0 = runner_assign[0]
            sw_1 = runner_assign[1]
            sw_2 = runner_assign[2]
            sw_3 = runner_assign[3]
            sw_4 = runner_assign[3]
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

            num_LSTMs_0 = len(range(0, num_steps - win_size[0] + motion_diff, stride[0]))

        # self.scale_intput = scale_intput = tf.Variable(80.0, name="scale_intput")
        with tf.variable_scope("Long_Sliding"):
            print "Long_Sliding_Top"

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Long0_GPU', 0)) as scope:
                    with tf.variable_scope('l0'):
                        self.mL0 = mL0 = TS_LSTM_Long_0(is_training, config, self.input_data, motion_diff)
                        self._initial_state = mL0.initial_state
                        output_depthconcat_long_0 = mL0.get_depth_concat_output()
                        self.output_depthconcat_long_0 = output_depthconcat_long_0

            with tf.device(sw_4):
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
    def __init__(self, is_training, config, input_data, motion_diff):
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
        start_time_0 = motion_diff  # 1
        num_LSTMs_0 = len(range(0, num_steps - win_size_0 + start_time_0, stride_0))
        print range(0, num_steps - win_size_0 + start_time_0, stride_0)
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
            output_0[win_step] = tf.reshape(tf.concat(1, self.outputs_0[win_step]), [-1, size])

        with tf.variable_scope("Dep_Con_0"):
            temp_output_0 = []
            for win_step in range(num_LSTMs_0):
                temp_output_0.append([])
                temp_output_0[win_step] = tf.reshape(output_0[win_step], [batch_size, num_steps, size])
                if win_step == 0:
                    input_0 = temp_output_0[win_step]
                else:
                    input_0 = tf.concat(1, [input_0, temp_output_0[win_step]])
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
    learning_rate = 0.05
    learning_rate2 = 0.01
    learning_rate3 = 0.005
    learning_rate4 = 0.001
    learning_rate5 = 0.0005
    learning_rate6 = 0.0001
    learning_rate7 = 0.00005
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
    max_max_epoch = 2000
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
    costs_L = 0.0
    iters = 0
    accuracys = 0.0

    ###
    state = session.run(m.initial_state_L)


    sumsums = 0.0
    p_ls = []
    g_ls = []
    #avg_time = 0.0
    #cnt = 0
    for step, (x, y) in enumerate(Action_input.MSR_iterator(data, label, m.batch_size, m.feature_size,
                                                            m.num_steps, is_training)):
        # start_batch = time.time()
        cost, accuracy, state, p_l, g_l, _ = session.run([m.cost, m.accuracy, m.final_state, m.pred_labels,
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
        # end_batch = time.time()
        # avg_time = avg_time + (end_batch - start_batch)
        # cnt = cnt + 1
        # if cnt % 10 == 1:
        #     print ("Batch %d_%d: %.3f" % (m.batch_size, cnt, avg_time / cnt))

    #print ("Batch %d_%d: %.3f" % (m.batch_size, cnt, avg_time/cnt))

    # print(sumsums)
    # summary_strs = np.append(summary_strs, summary_str)

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

def extract_data_label(data, label, subject, sub_train_range, config):
    data_index = np.array([])
    for i in sub_train_range:
        a, b = np.where(subject == i)
        data_index = np.append(data_index, a)

    temp_exdata = np.zeros([len(data_index), len(data[0])])
    origin_label = np.zeros([len(data_index), 1])
    for i in range(0, len(data_index)):
        temp_exdata[i] = data[int(data_index[i])]
        origin_label[i] = label[int(data_index[i])]

    exdata = temp_exdata.reshape([len(data_index), config.num_steps, config.input_size])

    exlabel = np.zeros([len(data_index), config.ori_class_size])
    for i in range(0, len(data_index)):
        for j in range(0, config.ori_class_size):
            if j == (label[int(data_index[i])] - 1):
                exlabel[i][j] = 1

    return exdata, exlabel, origin_label

def extract_AS(data, label, origin_label, config):
    data_index = np.array([])
    for i in config.AS1:
        a, b = np.where(origin_label == i)
        data_index = np.append(data_index, a)

    exdata = np.zeros([len(data_index), config.num_steps, config.input_size])
    exlabel = np.zeros([len(data_index), config.ori_class_size])
    for i in range(0, len(data_index)):
        exdata[i] = data[int(data_index[i])]
        exlabel[i] = label[int(data_index[i])]

    temp_exlabel = np.zeros([len(data_index), 1])
    ind = 1
    for i in config.AS1:
        for j in range(0, len(data_index)):
            if origin_label[int(data_index[j])] == i:
                temp_exlabel[j] = ind
        ind = ind + 1

    exexlabel = np.zeros([len(data_index), config.class_size])
    for i in range(0, len(data_index)):
        for j in range(0, config.class_size):
            if j == (temp_exlabel[i] - 1):
                exexlabel[i][j] = 1

    return exdata, exexlabel

def extract_feature(data, config):
    mode = config.mode
    if mode == 0:
        feature = np.zeros([len(data), config.num_steps, config.feature_size])
        for i in range(0, len(data)):
            for j in range(0, config.num_steps):
                for k in range(1, config.feature_size / 3 + 1):
                    feature[i][j][3 * (k - 1):3 * k] = data[i][j][3 * k:3 * (k + 1)] - data[i][j][0:3]
    elif mode == 1:
        feature = np.zeros([len(data), config.num_steps, config.feature_size])
        for i in range(0, len(data)):
            for j in range(0, config.num_steps):
                for k in range(1, config.feature_size / 3 + 1):
                    feature[i][j][3 * (k - 1):3 * k] = data[i][j][3 * k:3 * (k + 1)] - data[i][j][0:3]
        for i in range(0, len(data)):
            for j in range(0, config.num_steps):
                for k in range(1, config.feature_size / 3 + 1):
                    length = 0
                    for m in range(0, 3):
                        length += (feature[i][j][3 * (k - 1) + m] * feature[i][j][3 * (k - 1) + m])
                    sqrt_length = np.sqrt(length)
                    for m in range(0, 3):
                        if sqrt_length == 0:
                            pass
                        else:
                            feature[i][j][3 * (k - 1) + m] = feature[i][j][3 * (k - 1) + m] / sqrt_length
    elif mode == 2:
        joint_ind = [[1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [7, 8], [3, 9], [9, 10], [10, 11],
                     [11, 12], [1, 13], [13, 14], [14, 15], [15, 16], [1, 17], [17, 18], [18, 19], [19, 20]]
        feature = np.zeros([len(data), config.num_steps, config.feature_size])
        for i in range(0, len(data)):
            for j in range(0, config.num_steps):
                for k in range(config.feature_size / 3):
                    ind_start = int(joint_ind[k][0] - 1)
                    ind_end = int(joint_ind[k][1] - 1)
                    feature[i][j][3 * k:3 * (k + 1)] = data[i][j][3 * ind_end:3 * (ind_end + 1)] - data[i][j][
                                                                                                   3 * ind_start:3 * (
                                                                                                       ind_start + 1)]
    elif mode == 3:
        joint_ind = [[1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [7, 8], [3, 9], [9, 10], [10, 11],
                     [11, 12], [1, 13], [13, 14], [14, 15], [15, 16], [1, 17], [17, 18], [18, 19], [19, 20]]
        feature = np.zeros([len(data), config.num_steps, config.feature_size])
        for i in range(0, len(data)):
            for j in range(0, config.num_steps):
                for k in range(config.feature_size / 3):
                    ind_start = int(joint_ind[k][0] - 1)
                    ind_end = int(joint_ind[k][1] - 1)
                    feature[i][j][3 * k:3 * (k + 1)] = data[i][j][3 * ind_end:3 * (ind_end + 1)] - data[i][j][
                                                                                                   3 * ind_start:3 * (
                                                                                                       ind_start + 1)]
        for i in range(0, len(data)):
            for j in range(0, config.num_steps):
                for k in range(1, config.feature_size / 3 + 1):
                    length = 0
                    for m in range(0, 3):
                        length += (feature[i][j][3 * (k - 1) + m] * feature[i][j][3 * (k - 1) + m])
                    sqrt_length = np.sqrt(length)
                    for m in range(0, 3):
                        if sqrt_length == 0:
                            pass
                        else:
                            feature[i][j][3 * (k - 1) + m] = feature[i][j][3 * (k - 1) + m] / sqrt_length
    else:
        pass

    return feature

def count_zeros(feature_train, config):
    num_zeros = 0.0
    num_frames = 0.0
    for i in range(config.batch_size):
        for j in range(config.num_steps):
            if feature_train[i, j, 0] == 0.0:
                num_zeros += 1.0

    for k in range(config.feature_size):
        num_frames = config.num_steps * config.batch_size - num_zeros

    config.num_frames = num_frames
    config.num_zeros = num_zeros

    return config

def bone_length(feature_train, feature_test):
    SEG_INFO = [[6, 3], [3, 2], [2, 19], [2, 0], [0, 7], [7, 9], [9, 11], [2, 1], [1, 8], [8, 10], [10, 12], [6, 4],
                [4, 13], [13, 15], [15, 17], [6, 5], [5, 14], [14, 16], [16, 18]]

    feature_train_bl = np.zeros(feature_train.shape)
    feature_test_bl = np.zeros(feature_test.shape)

    stdBoneLength = np.zeros(len(SEG_INFO))
    for frmNo in range(feature_train.shape[1]):
        if np.linalg.norm(feature_train[1][frmNo]) != 0:
            for segNo in range(len(SEG_INFO)):
                # print(feature_train[0][frmNo][3*SEG_INFO[segNo][1]:3*(SEG_INFO[segNo][1]+1)])
                # print(feature_train[0][frmNo][3*SEG_INFO[segNo][0]:3*(SEG_INFO[segNo][0]+1)] - feature_train[0][frmNo][3*SEG_INFO[segNo][1]:3*(SEG_INFO[segNo][1]+1)])
                # print(np.linalg.norm(feature_train[0][frmNo][3*SEG_INFO[segNo][0]:3*(SEG_INFO[segNo][0]+1)] - feature_train[0][frmNo][3*SEG_INFO[segNo][1]:3*(SEG_INFO[segNo][1]+1)]))
                stdBoneLength[segNo] = np.linalg.norm(
                    feature_train[1][frmNo][3 * SEG_INFO[segNo][0]:3 * (SEG_INFO[segNo][0] + 1)] - feature_train[1][
                                                                                                       frmNo][
                                                                                                   3 * SEG_INFO[segNo][
                                                                                                       1]:3 * (
                                                                                                       SEG_INFO[segNo][
                                                                                                           1] + 1)])
            break

    for batchNo in range(feature_train.shape[0]):
        for frmNo in range(feature_train.shape[1]):
            if np.linalg.norm(feature_train[batchNo][frmNo]) != 0:
                feature_train_bl[batchNo][frmNo][3 * SEG_INFO[0][0]:3 * (SEG_INFO[0][0] + 1)] \
                    = feature_train[batchNo][frmNo][3 * SEG_INFO[0][0]:3 * (SEG_INFO[0][0] + 1)]
                for segNo in range(len(SEG_INFO)):
                    segment_vector \
                        = stdBoneLength[segNo] * \
                          (feature_train[batchNo][frmNo][3 * SEG_INFO[segNo][1]:3 * (SEG_INFO[segNo][1] + 1)]
                           - feature_train[batchNo][frmNo][3 * SEG_INFO[segNo][0]:3 * (SEG_INFO[segNo][0] + 1)]) \
                          / np.linalg.norm(
                        feature_train[batchNo][frmNo][3 * SEG_INFO[segNo][1]:3 * (SEG_INFO[segNo][1] + 1)]
                        - feature_train[batchNo][frmNo][3 * SEG_INFO[segNo][0]:3 * (SEG_INFO[segNo][0] + 1)])
                    feature_train_bl[batchNo][frmNo][3 * SEG_INFO[segNo][1]:3 * (SEG_INFO[segNo][1] + 1)] \
                        = feature_train_bl[batchNo][frmNo][
                          3 * SEG_INFO[segNo][0]:3 * (SEG_INFO[segNo][0] + 1)] + segment_vector

    for batchNo in range(feature_test.shape[0]):
        for frmNo in range(feature_test.shape[1]):
            if np.linalg.norm(feature_test[batchNo][frmNo]) != 0:
                feature_test_bl[batchNo][frmNo][3 * SEG_INFO[0][0]:3 * (SEG_INFO[0][0] + 1)] \
                    = feature_test[batchNo][frmNo][3 * SEG_INFO[0][0]:3 * (SEG_INFO[0][0] + 1)]
                for segNo in range(len(SEG_INFO)):
                    segment_vector = stdBoneLength[segNo] * (
                        feature_test[batchNo][frmNo][3 * SEG_INFO[segNo][1]:3 * (SEG_INFO[segNo][1] + 1)]
                        - feature_test[batchNo][frmNo][
                          3 * SEG_INFO[segNo][0]:3 * (SEG_INFO[segNo][0] + 1)]) / np.linalg.norm(
                        feature_test[batchNo][frmNo][3 * SEG_INFO[segNo][1]:3 * (SEG_INFO[segNo][1] + 1)]
                        - feature_test[batchNo][frmNo][3 * SEG_INFO[segNo][0]:3 * (SEG_INFO[segNo][0] + 1)])
                    feature_test_bl[batchNo][frmNo][3 * SEG_INFO[segNo][1]:3 * (SEG_INFO[segNo][1] + 1)] \
                        = feature_test_bl[batchNo][frmNo][
                          3 * SEG_INFO[segNo][0]:3 * (SEG_INFO[segNo][0] + 1)] + segment_vector

    return feature_train_bl, feature_test_bl

def body_rotation(feature_train, feature_test):
    feature_hc_train = np.zeros(feature_train.shape)
    feature_hc_test = np.zeros(feature_test.shape)

    for batchNo in range(feature_train.shape[0]):
        first_index = False
        for frmNo in range(feature_train.shape[1]):
            if (np.linalg.norm(feature_train[batchNo][frmNo]) != 0) and (first_index == False):
                new_origin = (feature_train[batchNo][frmNo][36:39] + feature_train[batchNo][frmNo][48:51]) / 2
                v1 = feature_train[batchNo][frmNo][3:6] - feature_train[batchNo][frmNo][0:3]
                v1 = v1 / np.linalg.norm(v1)
                u2 = feature_train[batchNo][frmNo][36:39] - feature_train[batchNo][frmNo][48:51]
                # print(u2)
                v2 = u2 - [np.inner(u2, v1) * v1[0], np.inner(u2, v1) * v1[1], np.inner(u2, v1) * v1[2]]
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v2, v1)

                rot = np.linalg.inv([[v2[0], v1[0], v3[0]], [v2[1], v1[1], v3[1]], [v2[2], v1[2], v3[2]]])

                # print(rot)

                for nodeNo in range(feature_train.shape[2] / 3):
                    # print(np.dot(rot, feature_hc_train[batchNo][frmNo][3*nodeNo:3*(nodeNo+1)].T - new_origin.T))
                    feature_hc_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                     new_origin[
                                                                                                                         1],
                                                                                                                     0]

                first_index = True
            elif (np.linalg.norm(feature_train[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(feature_train.shape[2] / 3):
                    feature_hc_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                     new_origin[
                                                                                                                         1],
                                                                                                                     0]
            else:
                pass

    for batchNo in range(feature_test.shape[0]):
        first_index = False
        for frmNo in range(feature_test.shape[1]):
            if (np.linalg.norm(feature_test[batchNo][frmNo]) != 0) and (first_index == False):
                new_origin = np.mean([feature_test[batchNo][frmNo][36:39], feature_test[batchNo][frmNo][48:51]], 0)
                v1 = feature_test[batchNo][frmNo][3:6] - feature_test[batchNo][frmNo][0:3]
                v1 = v1 / np.linalg.norm(v1)
                u2 = feature_test[batchNo][frmNo][36:39] - feature_test[batchNo][frmNo][48:51]
                v2 = u2 - [np.inner(u2, v1) * v1[0], np.inner(u2, v1) * v1[1], np.inner(u2, v1) * v1[2]]
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v2, v1)

                rot = np.linalg.inv([[v2[0], v1[0], v3[0]], [v2[1], v1[1], v3[1]], [v2[2], v1[2], v3[2]]])

                for nodeNo in range(feature_test.shape[2] / 3):
                    # print(feature_hc_test[batchNo][frmNo][3*nodeNo:3*(nodeNo+1)])
                    feature_hc_test[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature_test[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                    new_origin[
                                                                                                                        1],
                                                                                                                    0]
                first_index = True
            elif (np.linalg.norm(feature_test[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(feature_test.shape[2] / 3):
                    feature_hc_test[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature_test[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                    new_origin[
                                                                                                                        1],
                                                                                                                    0]
            else:
                pass

    # print(feature_hc_train[0][60])

    return feature_hc_train, feature_hc_test

def Human_Cognitive_Coordinate(feature):
    feature_hcc = np.zeros(feature.shape)

    for batchNo in range(feature.shape[0]):
        first_index = False
        for frmNo in range(feature.shape[1]):
            if (np.linalg.norm(feature[batchNo][frmNo][0:75]) != 0) and (first_index == False):
                new_origin = (feature[batchNo][frmNo][36:39] + feature[batchNo][frmNo][48:51]) / 2
                v1 = feature[batchNo][frmNo][3:6] - feature[batchNo][frmNo][0:3]
                v1 = v1 / np.linalg.norm(v1)
                u2 = feature[batchNo][frmNo][36:39] - feature[batchNo][frmNo][48:51]
                # print(u2)
                v2 = u2 - [np.inner(u2, v1) * v1[0], np.inner(u2, v1) * v1[1], np.inner(u2, v1) * v1[2]]
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v2, v1)

                rot = np.linalg.inv([[v2[0], v1[0], v3[0]], [v2[1], v1[1], v3[1]], [v2[2], v1[2], v3[2]]])

                # print(rot)

                for nodeNo in range(feature.shape[2] / 3):
                    # print(np.dot(rot, feature_hc_train[batchNo][frmNo][3*nodeNo:3*(nodeNo+1)].T - new_origin.T))
                    feature_hcc[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                     new_origin[
                                                                                                                         1],
                                                                                                                     0]

                first_index = True
            elif (np.linalg.norm(feature[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(feature.shape[2] / 3):
                    feature_hcc[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                     new_origin[
                                                                                                                         1],
                                                                                                                     0]
            else:
                feature_hcc[batchNo][frmNo] = np.zeros(feature.shape[2])
        # print(batchNo)
        # print(np.sum(np.sum(np.isnan(feature_hcc[batchNo]))))
        if np.sum(np.sum(np.isnan(feature_hcc[batchNo]))) > 0:
            print('Nan')
            print(batchNo)

    return feature_hcc

def feature_only_diff(data, maxValue, config):
    new_data = np.zeros([len(data), maxValue, config.feature_size])

    for batch_step in range(len(data)):
        # print len(data[batch_step])

        new_data[batch_step][maxValue-len(data[batch_step]):maxValue] = data[batch_step]
        for time_step in range(maxValue):
            if np.sum(new_data[batch_step][time_step]) != 0:
                for ttime_step in range(time_step):
                    new_data[batch_step][ttime_step] = new_data[batch_step][time_step]
                break
            else:
                pass

    return new_data

def feature_only_diff_2(data, maxValue, config):
    new_data = np.zeros([len(data), maxValue, 2*config.feature_size])

    for batch_step in range(len(data)):
        # print len(data[batch_step])

        new_data[batch_step][maxValue-len(data[batch_step]):maxValue] = data[batch_step]
        for time_step in range(maxValue):
            if np.sum(new_data[batch_step][time_step][0:75]) != 0:
                for ttime_step in range(time_step):
                    new_data[batch_step][ttime_step][0:75] = new_data[batch_step][time_step][0:75]
                break
            else:
                pass
        for time_step in range(maxValue):
            if np.sum(new_data[batch_step][time_step][75:150]) != 0:
                for ttime_step in range(time_step):
                    new_data[batch_step][ttime_step][75:150] = new_data[batch_step][time_step][75:150]
                break
            else:
                pass

    return new_data

def feature_only_diff_0(data, maxValue, config):
    new_data = np.zeros([len(data), maxValue, config.feature_size])

    for batch_step in range(len(data)):
        new_data[batch_step][maxValue-len(data[batch_step]):maxValue] = data[batch_step]

    return new_data

def Pose_Motion(feature):
    Feature_PM = np.zeros(feature.shape)
    for time_step in range(feature.shape[1]):
        if time_step > 0:
            Feature_PM[:, time_step, :] = feature[:, time_step, :] - feature[:, time_step - 1, :]
        else:
            Feature_PM[:, time_step, :] = feature[:, time_step, :]

    return Feature_PM

def one_hot_labeling(original_label, config):
    new_label = np.zeros([len(original_label), config.class_size])

    for batch_step in range(len(original_label)):
        # print original_label[batch_step]
        new_label[batch_step][original_label[batch_step]-1] = 1

    return new_label

def main(_):
    # Original
    # config = get_config()
    # config.class_size = 60
    # config.feature_size = 75
    # config.input_size = 75
    # config.hidden_size = 70
    # config.keep_prob = 0.4
    #
    # eval_config = get_config()
    # eval_config.class_size = 60
    # eval_config.feature_size = 75
    # eval_config.input_size = 75
    # eval_config.hidden_size = 70
    # eval_config.keep_prob = 0.4

    config = get_config()
    config.class_size = 60
    config.feature_size = 150
    config.input_size = 150
    config.hidden_size = 100
    config.keep_prob = myKeepProb

    eval_config = get_config()
    eval_config.class_size = 60
    eval_config.feature_size = 150
    eval_config.input_size = 150
    eval_config.hidden_size = 100
    eval_config.keep_prob = myKeepProb

    # ###################################################################################################################
    # DATA_PATH = 'sklt_data_all'
    # CAMERA_OR_SUBJECT = 2
    #
    # train_set = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16]
    # test_set = [17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    # # test_set = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
    #
    # train_sklt0, train_label0 = Action_input.read(train_set, DATA_PATH, CAMERA_OR_SUBJECT, config)
    # test_sklt0, test_label0 = Action_input.read(test_set, DATA_PATH, CAMERA_OR_SUBJECT, eval_config)
    #
    # MAX_LENGTH = 0
    # for batchNo in range(len(train_sklt0)):
    #     if len(train_sklt0[batchNo]) > MAX_LENGTH:
    #         MAX_LENGTH = len(train_sklt0[batchNo])
    #     else:
    #         pass
    #
    # for batchNo in range(len(test_sklt0)):
    #     if len(test_sklt0[batchNo]) > MAX_LENGTH:
    #         MAX_LENGTH = len(test_sklt0[batchNo])
    #     else:
    #         pass
    #
    # print('MAX_LENGTH = ', MAX_LENGTH)
    #
    # train_sklt1 = feature_only_diff_0(train_sklt0, MAX_LENGTH, config)
    # test_sklt1 = feature_only_diff_0(test_sklt0, MAX_LENGTH, eval_config)
    #
    # del train_sklt0, test_sklt0
    #
    # train_sklt2 = Human_Cognitive_Coordinate(train_sklt1)
    # test_sklt2 = Human_Cognitive_Coordinate(test_sklt1)
    #
    # del train_sklt1, test_sklt1
    #
    # feature_train = Pose_Motion(train_sklt2)
    # feature_test = Pose_Motion(test_sklt2)
    #
    # AS_train_label = one_hot_labeling(train_label0, config)
    # AS_test_label = one_hot_labeling(test_label0, eval_config)
    #
    # del train_sklt2, test_sklt2, train_label0, test_label0
    # ###################################################################################################################
    # # NTU skeleton data: ALL
    # np.save('./sklt_npy_subject_TPAMI_val/feature_train_subject_all', feature_train)
    # np.save('./sklt_npy_subject_TPAMI_val/AS_train_label_subject_all', AS_train_label)
    # np.save('./sklt_npy_subject_TPAMI_val/feature_test_subject_all', feature_test)
    # np.save('./sklt_npy_subject_TPAMI_val/AS_test_label_subject_all', AS_test_label)
    # print 'Skeleton data stored.'

    # LOAD .npy: ALL
    feature_train = np.load('./sklt_npy_subject_TPAMI_val/feature_train_{0}_all.npy'.format(view_subject))
    AS_train_label = np.load('./sklt_npy_subject_TPAMI_val/AS_train_label_{0}_all.npy'.format(view_subject))
    feature_test = np.load('./sklt_npy_subject_TPAMI_val/feature_test_{0}_all.npy'.format(view_subject))
    AS_test_label = np.load('./sklt_npy_subject_TPAMI_val/AS_test_label_{0}_all.npy'.format(view_subject))
    print 'Skeleton data (camera_all) restored.'

    # feature_train = feature_train[:, :, 0:75]
    # feature_test = feature_test[:, :, 0:75]

    print feature_train.shape, feature_test.shape
    print AS_train_label.shape, AS_test_label.shape

    config.batch_size = len(feature_train) / BatchDivider### batch_modifier
    # config.batch_size = 256  ### batch_modifier
    eval_config.batch_size = 1
    config.num_steps = len(feature_train[0])
    eval_config.num_steps = len(feature_test[0])

    print config.batch_size, eval_config.batch_size
    print ("Total Training Set Length : %d, Traning Batch Size : %d, Eval Batch Size : %d"
           % (len(feature_train),config.batch_size,eval_config.batch_size))

    #TODO=========================================================================================== SAVED FILE PATH CONFIG

    csv_suffix = strftime("_%Y%m%d_%H%M.csv", localtime())
    folder_path = os.path.join(myFolderPath) #folder_modifier

    checkpoint_path = os.path.join(folder_path, "NTU_{0}.ckpt".format(view_subject))
    timecsv_path = os.path.join(folder_path, "Auto" + csv_suffix)

    f = open(timecsv_path,'w')
    csvWriter = csv.writer(f)

    #TODO=========================================================================================== LOAD BALANCING


    print "\nManual Parallelism"
    global runner_assign
    # runner_assign = ['/cpu:0', '/cpu:0', '/cpu:0', '/cpu:0']
    use_gpu = '/gpu:0'
    runner_assign = [use_gpu, use_gpu, use_gpu, use_gpu]
    ############### L0,L1,L2,M3,M4,S5


    #TODO=========================================================================================== SESSION CONFIG

    sessConfig = tf.ConfigProto(log_device_placement=False)
    sessConfig.gpu_options.allow_growth = True

    writeConfig_tocsv = True

    if writeConfig_tocsv:
        csvWriter.writerow(['DateTime:', strftime("%Y%m%d_%H:%M:%S", localtime())])
        csvWriter.writerow([])
        csvWriter.writerow(['Total Dataset Length', 'Train Batch Divider', 'Train Batch Size', 'Eval Batch Size', ])
        csvWriter.writerow(
            [len(feature_train), len(feature_train) / config.batch_size, config.batch_size, eval_config.batch_size])

        csvWriter.writerow(['Control', 'Long'])
        csvWriter.writerow(['win_size', win_size[0]])
        csvWriter.writerow(['stride', stride[0]])
        csvWriter.writerow(
            ['start_time', start_time[0], start_time[1], start_time[2], start_time[3], start_time[4], start_time[5],
             start_time[6], start_time[7], start_time[8], start_time[9]])
        csvWriter.writerow([])

    #TODO=========================================================================================== BUILD GRAPH
    with tf.Graph().as_default(), tf.Session(config=sessConfig) as session:
        with tf.device('/cpu:0'):
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            with tf.variable_scope("model1", reuse=None, initializer=initializer):
                m = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[0])

            with tf.variable_scope("model2", reuse=None, initializer=initializer):
                m2 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[1])

            with tf.variable_scope("model3", reuse=None, initializer=initializer):
                m3 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[2])

            with tf.variable_scope("model4", reuse=None, initializer=initializer):
                m4 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[3])

            with tf.variable_scope("model5", reuse=None, initializer=initializer):
                m5 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[4])

            with tf.variable_scope("model6", reuse=None, initializer=initializer):
                m6 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[5])

            with tf.variable_scope("model7", reuse=None, initializer=initializer):
                m7 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[6])

            with tf.variable_scope("model8", reuse=None, initializer=initializer):
                m8 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[7])

            with tf.variable_scope("model9", reuse=None, initializer=initializer):
                m9 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[8])

            with tf.variable_scope("model10", reuse=None, initializer=initializer):
                m10 = MP_runner(is_training=True, config=config, labels=AS_train_label, motion_diff=start_time[9])
            print "\nTraining Models Established!\n"

            with tf.variable_scope("model1", reuse=True, initializer=initializer):
                mtest = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[0])

            with tf.variable_scope("model2", reuse=True, initializer=initializer):
                mtest2 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[1])

            with tf.variable_scope("model3", reuse=True, initializer=initializer):
                mtest3 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[2])

            with tf.variable_scope("model4", reuse=True, initializer=initializer):
                mtest4 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[3])

            with tf.variable_scope("model5", reuse=True, initializer=initializer):
                mtest5 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[4])

            with tf.variable_scope("model6", reuse=True, initializer=initializer):
                mtest6 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[5])

            with tf.variable_scope("model7", reuse=True, initializer=initializer):
                mtest7 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[6])

            with tf.variable_scope("model8", reuse=True, initializer=initializer):
                mtest8 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[7])

            with tf.variable_scope("model9", reuse=True, initializer=initializer):
                mtest9 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[8])

            with tf.variable_scope("model10", reuse=True, initializer=initializer):
                mtest10 = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, motion_diff=start_time[9])
            print "\nTesting Models Established!!\n"

            # summary_writer = tf.train.SummaryWriter('/home/inwoong/MSR_logs', graph=session.graph)

            init = tf.initialize_all_variables() #TF ver 0.11
            #init = tf.global_variables_initializer() #TF ver 0.12
            session.run(init)

            saver = tf.train.Saver(tf.all_variables())

            saver.restore(session, "./TrialSaver/motion_diff/20180424_2158_BD_60_KB_0.40_WS_210_ST_90/NTU_subject.ckpt-400")
            print("Model restored.")

            stt_loop = time.time()
            print strftime("%Y%m%d_%H:%M:%S", localtime())

            csvWriter.writerow(
                ['Time', 'Epoch #', 'Epoch Time', 'Train Accuracy1', 'Train Cost1', 'Train Accuracy2', 'Train Cost2',
                 'Train Accuracy3', 'Train Cost3', 'Train Accuracy4', 'Train Cost4', 'Train Accuracy5', 'Train Cost5',
                 'Train Accuracy6', 'Train Cost6', 'Train Accuracy7', 'Train Cost7', 'Train Accuracy8', 'Train Cost8',
                 'Train Accuracy9', 'Train Cost9', 'Train Accuracy10', 'Train Cost10'])
            for i in range(config.max_max_epoch):

                stt_lr = time.time()

                if i == 0:
                    print "First Learning Rate is assigned!!"
                    m.assign_lr(session, config.learning_rate)
                    m2.assign_lr(session, config.learning_rate)
                    m3.assign_lr(session, config.learning_rate)
                    m4.assign_lr(session, config.learning_rate)
                    m5.assign_lr(session, config.learning_rate)
                    m6.assign_lr(session, config.learning_rate)
                    m7.assign_lr(session, config.learning_rate)
                    m8.assign_lr(session, config.learning_rate)
                    m9.assign_lr(session, config.learning_rate)
                    m10.assign_lr(session, config.learning_rate)
                elif i == config.max_epoch1:
                    m.assign_lr(session, config.learning_rate2)
                    m2.assign_lr(session, config.learning_rate2)
                    m3.assign_lr(session, config.learning_rate2)
                    m4.assign_lr(session, config.learning_rate2)
                    m5.assign_lr(session, config.learning_rate2)
                    m6.assign_lr(session, config.learning_rate2)
                    m7.assign_lr(session, config.learning_rate2)
                    m8.assign_lr(session, config.learning_rate2)
                    m9.assign_lr(session, config.learning_rate2)
                    m10.assign_lr(session, config.learning_rate2)

                stt_epoch = time.time()

                if i == 0:
                    print "I'm Ready for First Epoch"
                train_cost, train_accuracy, tr_p_l, tr_g_l = run_epoch(
                    session, m, feature_train, AS_train_label,
                    m.train_op,
                    verbose=True)
                train_cost2, train_accuracy2, tr_p_l2, tr_g_l2 = run_epoch(
                    session, m2, feature_train, AS_train_label,
                    m2.train_op,
                    verbose=True)
                train_cost3, train_accuracy3, tr_p_l3, tr_g_l3 = run_epoch(
                    session, m3, feature_train, AS_train_label,
                    m3.train_op,
                    verbose=True)
                train_cost4, train_accuracy4, tr_p_l4, tr_g_l4 = run_epoch(
                    session, m4, feature_train, AS_train_label,
                    m4.train_op,
                    verbose=True)
                train_cost5, train_accuracy5, tr_p_l5, tr_g_l5 = run_epoch(
                    session, m5, feature_train, AS_train_label,
                    m5.train_op,
                    verbose=True)
                train_cost6, train_accuracy6, tr_p_l6, tr_g_l6 = run_epoch(
                    session, m6, feature_train, AS_train_label,
                    m6.train_op,
                    verbose=True)
                train_cost7, train_accuracy7, tr_p_l7, tr_g_l7 = run_epoch(
                    session, m7, feature_train, AS_train_label,
                    m7.train_op,
                    verbose=True)
                train_cost8, train_accuracy8, tr_p_l8, tr_g_l8 = run_epoch(
                    session, m8, feature_train, AS_train_label,
                    m8.train_op,
                    verbose=True)
                train_cost9, train_accuracy9, tr_p_l9, tr_g_l9 = run_epoch(
                    session, m9, feature_train, AS_train_label,
                    m9.train_op,
                    verbose=True)
                train_cost10, train_accuracy10, tr_p_l10, tr_g_l10 = run_epoch(
                    session, m10, feature_train, AS_train_label,
                    m10.train_op,
                    verbose=True)


                end_epoch = time.time()
                assert not np.isnan(train_cost), 'Model1 diverged with loss = NaN'
                assert not np.isnan(train_cost2), 'Model2 diverged with loss = NaN'
                assert not np.isnan(train_cost3), 'Model3 diverged with loss = NaN'
                assert not np.isnan(train_cost4), 'Model4 diverged with loss = NaN'
                assert not np.isnan(train_cost5), 'Model5 diverged with loss = NaN'
                assert not np.isnan(train_cost6), 'Model6 diverged with loss = NaN'
                assert not np.isnan(train_cost7), 'Model7 diverged with loss = NaN'
                assert not np.isnan(train_cost8), 'Model8 diverged with loss = NaN'
                assert not np.isnan(train_cost9), 'Model9 diverged with loss = NaN'
                assert not np.isnan(train_cost10), 'Model10 diverged with loss = NaN'
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
                    print strtime
                    print("----------Epoch Time: %.3f, per Assign: %.3f, per Epoch: %.3f" % (
                        (end_loop - stt_loop), (stt_epoch - stt_lr), (end_epoch - stt_epoch)))
                    print("Epoch: %d Learning rate: %.6f Train Accuracy1: %.4f" % (i, session.run(m.lr), train_accuracy))
                    print("Train Cost1: %.6f" % (train_cost))
                    print("Epoch: %d Learning rate: %.6f Train Accuracy2: %.4f" % (i, session.run(m2.lr), train_accuracy2))
                    print("Train Cost2: %.6f" % (train_cost2))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy3: %.4f" % (i, session.run(m3.lr), train_accuracy3))
                    print("Train Cost3: %.6f" % (train_cost3))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy4: %.4f" % (i, session.run(m4.lr), train_accuracy4))
                    print("Train Cost4: %.6f" % (train_cost4))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy5: %.4f" % (i, session.run(m5.lr), train_accuracy5))
                    print("Train Cost5: %.6f" % (train_cost5))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy6: %.4f" % (i, session.run(m6.lr), train_accuracy6))
                    print("Train Cost6: %.6f" % (train_cost6))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy7: %.4f" % (i, session.run(m7.lr), train_accuracy7))
                    print("Train Cost7: %.6f" % (train_cost7))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy8: %.4f" % (i, session.run(m8.lr), train_accuracy8))
                    print("Train Cost8: %.6f" % (train_cost8))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy9: %.4f" % (i, session.run(m9.lr), train_accuracy9))
                    print("Train Cost9: %.6f" % (train_cost9))
                    print(
                    "Epoch: %d Learning rate: %.6f Train Accuracy10: %.4f" % (i, session.run(m10.lr), train_accuracy10))
                    print("Train Cost10: %.6f" % (train_cost10))
                    stt_loop = time.time()
                    print("\n")

                    csvWriter.writerow([strtime, i, (end_epoch - stt_epoch), train_accuracy, train_cost,
                                        train_accuracy2, train_cost2, train_accuracy3, train_cost3, train_accuracy4, train_cost4,
                                        train_accuracy5, train_cost5, train_accuracy6, train_cost6, train_accuracy7, train_cost7,
                                        train_accuracy8, train_cost8, train_accuracy9, train_cost9, train_accuracy10, train_cost10])

                if i % 100 == 0:
                    test_cost, test_accuracy, te_p_l, te_g_l = run_epoch(session,
                                                                                      mtest,
                                                                                      feature_test,
                                                                                      AS_test_label,
                                                                                      tf.no_op(), is_training=False)
                    test_cost2, test_accuracy2, te_p_l2, te_g_l2 = run_epoch(session,
                                                                         mtest2,
                                                                         feature_test,
                                                                         AS_test_label,
                                                                         tf.no_op(), is_training=False)
                    test_cost3, test_accuracy3, te_p_l3, te_g_l3 = run_epoch(session,
                                                                             mtest3,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    test_cost4, test_accuracy4, te_p_l4, te_g_l4 = run_epoch(session,
                                                                             mtest4,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    test_cost5, test_accuracy5, te_p_l5, te_g_l5 = run_epoch(session,
                                                                             mtest5,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    test_cost6, test_accuracy6, te_p_l6, te_g_l6 = run_epoch(session,
                                                                             mtest6,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    test_cost7, test_accuracy7, te_p_l7, te_g_l7 = run_epoch(session,
                                                                             mtest7,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    test_cost8, test_accuracy8, te_p_l8, te_g_l8 = run_epoch(session,
                                                                             mtest8,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    test_cost9, test_accuracy9, te_p_l9, te_g_l9 = run_epoch(session,
                                                                             mtest9,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    test_cost10, test_accuracy10, te_p_l10, te_g_l10 = run_epoch(session,
                                                                             mtest10,
                                                                             feature_test,
                                                                             AS_test_label,
                                                                             tf.no_op(), is_training=False)
                    print("Test Accuracy1: %.5f" % (test_accuracy))
                    print("Test Accuracy2: %.5f" % (test_accuracy2))
                    print("Test Accuracy3: %.5f" % (test_accuracy3))
                    print("Test Accuracy4: %.5f" % (test_accuracy4))
                    print("Test Accuracy5: %.5f" % (test_accuracy5))
                    print("Test Accuracy6: %.5f" % (test_accuracy6))
                    print("Test Accuracy7: %.5f" % (test_accuracy7))
                    print("Test Accuracy8: %.5f" % (test_accuracy8))
                    print("Test Accuracy9: %.5f" % (test_accuracy9))
                    print("Test Accuracy10: %.5f\n" % (test_accuracy10))
                    csvWriter.writerow(["Test Accuracy :", test_accuracy, test_accuracy2, test_accuracy3, test_accuracy4,
                                        test_accuracy5, test_accuracy6, test_accuracy7, test_accuracy8, test_accuracy9,
                                        test_accuracy10])

                    confusion_matrix = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix2 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix3 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix4 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix5 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix6 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix7 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix8 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix9 = np.zeros([config.class_size, config.class_size + 1])
                    confusion_matrix10 = np.zeros([config.class_size, config.class_size + 1])
                    class_prob = np.zeros([config.class_size])
                    class_prob2 = np.zeros([config.class_size])
                    class_prob3 = np.zeros([config.class_size])
                    class_prob4 = np.zeros([config.class_size])
                    class_prob5 = np.zeros([config.class_size])
                    class_prob6 = np.zeros([config.class_size])
                    class_prob7 = np.zeros([config.class_size])
                    class_prob8 = np.zeros([config.class_size])
                    class_prob9 = np.zeros([config.class_size])
                    class_prob10 = np.zeros([config.class_size])
                    for j in range(len(te_g_l)):
                        confusion_matrix[te_g_l[j]][te_p_l[j]] += 1
                        confusion_matrix2[te_g_l2[j]][te_p_l2[j]] += 1
                        confusion_matrix3[te_g_l3[j]][te_p_l3[j]] += 1
                        confusion_matrix4[te_g_l4[j]][te_p_l4[j]] += 1
                        confusion_matrix5[te_g_l5[j]][te_p_l5[j]] += 1
                        confusion_matrix6[te_g_l6[j]][te_p_l6[j]] += 1
                        confusion_matrix7[te_g_l7[j]][te_p_l7[j]] += 1
                        confusion_matrix8[te_g_l8[j]][te_p_l8[j]] += 1
                        confusion_matrix9[te_g_l9[j]][te_p_l9[j]] += 1
                        confusion_matrix10[te_g_l10[j]][te_p_l10[j]] += 1
                    for j in range(config.class_size):
                        class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:config.class_size])
                        class_prob2[j] = confusion_matrix2[j][j] / np.sum(confusion_matrix2[j][0:config.class_size])
                        class_prob3[j] = confusion_matrix3[j][j] / np.sum(confusion_matrix3[j][0:config.class_size])
                        class_prob4[j] = confusion_matrix4[j][j] / np.sum(confusion_matrix4[j][0:config.class_size])
                        class_prob5[j] = confusion_matrix5[j][j] / np.sum(confusion_matrix5[j][0:config.class_size])
                        class_prob6[j] = confusion_matrix6[j][j] / np.sum(confusion_matrix6[j][0:config.class_size])
                        class_prob7[j] = confusion_matrix7[j][j] / np.sum(confusion_matrix7[j][0:config.class_size])
                        class_prob8[j] = confusion_matrix8[j][j] / np.sum(confusion_matrix8[j][0:config.class_size])
                        class_prob9[j] = confusion_matrix9[j][j] / np.sum(confusion_matrix9[j][0:config.class_size])
                        class_prob10[j] = confusion_matrix10[j][j] / np.sum(confusion_matrix10[j][0:config.class_size])

                    # for j in range(config.class_size):
                    #     confusion_matrix[j][config.class_size] = class_prob[j]
                    #     print class_prob[j]*100

                    with open(folder_path + "/view-test-" + str(i) + ".csv", "wb") as csvfile:
                        csvwriter2 = csv.writer(csvfile)
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix2[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix3[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix4[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix5[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix6[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix7[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix8[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix9[j])
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix10[j])

    f.close()


if __name__ == "__main__":
    tf.app.run()