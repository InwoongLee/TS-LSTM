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
import sys

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
# start_time = [1, 5, 10, 1, 5, 1]

# win_size = [49, 99, 149, 199, 249, 299]
# stride = [49, 99, 149, 199, 249, 299]
# start_time = [1, 1, 1, 1, 1, 1]






# win_size = [35, 77, 103, 103, 154, 154]
# stride = [33, 74, 98, 98, 145, 145]
# start_time = [4, 4, 3, 5, 2, 3]

win_size = [300]
stride = [300]
start_time = [0]


# win_size = [77]
# stride = [74]
# start_time = [1]



# win_size = [35, 77, 103, 103, 154, 154, 154]
# stride = [33, 74, 98, 98, 145, 145, 145]
# start_time = [4, 4, 3, 5, 1, 2, 3]

# win_size = [20, 35, 77, 103, 103, 154, 154, 154]
# stride = [19, 33, 74, 98, 98, 154, 145, 145]
# start_time = [6, 4, 4, 3, 5, 1, 2, 3]

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
numofGPU = 4
_now = datetime.now()
BatchDivider = 30
myKeepProb = 1.0
view_subject = 'view'

myFolderPath = './TrialSaver/%s/%04d%02d%02d_%02d%02d_BD_%d_KB_%.2f'%(view_subject,_now.year,
                                                                           _now.month,
                                                                           _now.day,
                                                                           _now.hour,
                                                                           _now.minute,
                                                                           BatchDivider,
                                                                           myKeepProb)
os.mkdir(myFolderPath)


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
        self._logits_0 = []
        self._cross_entropy_0 = []
        self._final_state = []
        self._cost_L = []
        self.tvars_L = []
        self.grad_before_sum_L = []
        for wi in range(len(win_sizes)):
            self.mL.append([])
            self._logits_0.append([])
            self._cross_entropy_0.append([])
            self._final_state.append([])
            self._cost_L.append([])
            self.tvars_L.append([])
            self.grad_before_sum_L.append([])
            if is_training:
                sw_general = '/gpu:2'
            else:
                sw_general = '/cpu:0'
            with tf.device(sw_general):
                with tf.name_scope('%s_%d' % ('mGPU', wi % numofGPU)) as scope:
                    self.mL[wi] = self.mL[wi] = TS_Long_LSTM_Top(is_training, config, labels, self.input_data, self.targets, size,
                                               win_sizes[wi], strides[wi], start_times[wi], sw_general, wi)
                    self._logits_0[wi], self._cross_entropy_0[wi], self._final_state[wi] = self.mL[wi].get_inference()
                    self._cost_L[wi] = tf.reduce_sum(self._cross_entropy_0[wi] / len(win_sizes) ) / batch_size

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
            print "Parallelized Model is on building!!"

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
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size
            else:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = 0
                    for ri in range(len(self._logits_0)):
                        real_logits = real_logits + self._logits_0[ri] / len(self._logits_0)
                    self._correct_prediction = tf.equal(tf.argmax(real_logits, 1), tf.argmax(self.targets, 1))
                    self.given_labels = tf.argmax(self.targets, 1)
                    self.pred_labels = tf.argmax(real_logits, 1)
                    self._accuracy = tf.reduce_sum(tf.cast(self._correct_prediction, tf.float32)) / batch_size

        if is_training:
            with tf.device(gradient_device[3]):
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

        print "Calculating Graph is fully connected!!"

    def assign_lr(self, sess, lr_value):
        sess.run(tf.assign(self.lr, lr_value))

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

            num_LSTMs_0 = 1
            # num_LSTMs_1 = len(range(start_time[1], num_steps - win_size[1] + start_time[1], stride[1]))
            # num_LSTMs_2 = len(range(start_time[2], num_steps - win_size[2] + start_time[2], stride[2]))

        # self.scale_intput = scale_intput = tf.Variable(80.0, name="scale_intput")
        with tf.variable_scope('Type%d' % wi):
            print sw_general

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
                    self.softmax_w_0 = softmax_w_0 = tf.get_variable("softmax_w_0", [num_LSTMs_0 * num_steps * size, class_size])
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
        num_LSTMs_0 = 1
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
                            distance = inputs[:, time_step, :]
                        else:
                            distance = inputs[:, time_step, :]
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
            # concat_output_real_0 = tf.reduce_sum(input_0, 2)
            self.out_concat_output_real_0 = tf.reshape(input_0, [batch_size, num_LSTMs_0 * num_steps * size])

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
    learning_rate = 0.001
    learning_rate2 = 0.0005
    learning_rate3 = 0.0001
    learning_rate4 = 0.00005
    learning_rate5 = 0.00001
    learning_rate6 = 0.000005
    learning_rate7 = 0.000001
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
    max_max_epoch = 30000
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
    iters = 0
    accuracys = 0.0

    sumsums = 0.0
    p_ls = []
    g_ls = []
    #avg_time = 0.0
    cnt = 0
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
        cnt += 1
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
                    # feature_hcc[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                    #     = np.transpose(
                    #     np.dot(rot, feature[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                    #                                                                                                  new_origin[
                    #                                                                                                      1],
                    #                                                                                                  0]
                    feature_hcc[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T))

                first_index = True
            elif (np.linalg.norm(feature[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(feature.shape[2] / 3):
                    # feature_hcc[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                    #     = np.transpose(
                    #     np.dot(rot, feature[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                    #                                                                                                  new_origin[
                    #                                                                                                      1],
                    #                                                                                                  0]
                    feature_hcc[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T))
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

    ###################################################################################################################
    DATA_PATH = 'sklt_data_all'
    CAMERA_OR_SUBJECT = 1

    train_set = [2, 3]
    test_set = [1]

    train_sklt0, train_label0 = Action_input.read(train_set, DATA_PATH, CAMERA_OR_SUBJECT, config)
    test_sklt0, test_label0 = Action_input.read(test_set, DATA_PATH, CAMERA_OR_SUBJECT, eval_config)

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

    # MAX_LENGTH = 300

    print('MAX_LENGTH = ', MAX_LENGTH)

    train_sklt1 = feature_only_diff_0(train_sklt0, MAX_LENGTH, config)
    test_sklt1 = feature_only_diff_0(test_sklt0, MAX_LENGTH, eval_config)

    # feature_train = feature_only_diff_0(train_sklt0, MAX_LENGTH, config)
    # feature_test = feature_only_diff_0(test_sklt0, MAX_LENGTH, eval_config)

    del train_sklt0, test_sklt0

    feature_train = Human_Cognitive_Coordinate(train_sklt1)
    feature_test = Human_Cognitive_Coordinate(test_sklt1)

    del train_sklt1, test_sklt1

    # feature_train = Pose_Motion(train_sklt2)
    # feature_test = Pose_Motion(test_sklt2)

    AS_train_label = one_hot_labeling(train_label0, config)
    AS_test_label = one_hot_labeling(test_label0, eval_config)

    # del train_sklt2, test_sklt2, train_label0, test_label0
    ###################################################################################################################
    # # NTU skeleton data: ALL
    # np.save('./sklt_npy_subject_TPAMI/feature_train_subject_all', feature_train)
    # np.save('./sklt_npy_subject_TPAMI/AS_train_label_subject_all', AS_train_label)
    # np.save('./sklt_npy_subject_TPAMI/feature_test_subject_all', feature_test)
    # np.save('./sklt_npy_subject_TPAMI/AS_test_label_subject_all', AS_test_label)
    # print 'Skeleton data stored.'

    # # LOAD .npy: ALL
    # feature_train = np.load('./sklt_npy_subject_TPAMI/feature_train_{0}_all.npy'.format(view_subject))
    # AS_train_label = np.load('./sklt_npy_subject_TPAMI/AS_train_label_{0}_all.npy'.format(view_subject))
    # feature_test = np.load('./sklt_npy_subject_TPAMI/feature_test_{0}_all.npy'.format(view_subject))
    # AS_test_label = np.load('./sklt_npy_subject_TPAMI/AS_test_label_{0}_all.npy'.format(view_subject))
    # print 'Skeleton data (camera_all) restored.'

    # feature_train = feature_train[:, :, 0:75]
    # feature_test = feature_test[:, :, 0:75]

    print feature_train.shape, feature_test.shape
    print AS_train_label.shape, AS_test_label.shape

    config.batch_size = len(feature_train) / BatchDivider### batch_modifier
    eval_config.batch_size = len(feature_test)
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

        # csvWriter.writerow(['win_size', win_size[0], win_size[1], win_size[2], win_size[3], win_size[4], win_size[5]])
        # csvWriter.writerow(['win_size', win_size[0], win_size[1], win_size[2], win_size[3], win_size[4], win_size[5], win_size[6], win_size[7], win_size[8], win_size[9]])
        # csvWriter.writerow(['stride', stride[0], stride[1], stride[2], stride[3], stride[4], stride[5]])
        # csvWriter.writerow(['stride', stride[0], stride[1], stride[2], stride[3], stride[4], stride[5], stride[6], stride[7], stride[8], stride[9]])
        # csvWriter.writerow(
        #     ['start_time', start_time[0], start_time[1], start_time[2], start_time[3], start_time[4], start_time[5]])
        # csvWriter.writerow(
        #     ['start_time', start_time[0], start_time[1], start_time[2], start_time[3], start_time[4], start_time[5], start_time[6], start_time[7], start_time[8], start_time[9]])
        # csvWriter.writerow(['Assign', Top_Long_assign[1], Top_Medium_assign[1], Top_Short_assign[1]])
        csvWriter.writerow([])
        csvWriter.writerow(['nums of gpu :', getDevice()])
        csvWriter.writerow([])
        csvWriter.writerow(['LSTM_HCC'])

    #TODO=========================================================================================== BUILD GRAPH
    with tf.Graph().as_default(), tf.Session(config=sessConfig) as session:
        with tf.device('/cpu:0'):
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = MP_runner(is_training=True, config=config, labels=AS_train_label, win_sizes=win_size, strides = stride, start_times = start_time)
            print "\nTraining Model Established!\n"

            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = MP_runner(is_training=False, config=eval_config, labels=AS_test_label, win_sizes=win_size, strides = stride, start_times = start_time)

            print "\nTesting Model Established!!\n"

            # summary_writer = tf.train.SummaryWriter('/home/inwoong/MSR_logs', graph=session.graph)

            init = tf.initialize_all_variables() #TF ver 0.11
            #init = tf.global_variables_initializer() #TF ver 0.12
            session.run(init)

            saver = tf.train.Saver(tf.all_variables())

            #saver.restore(session, "./result_apply_170116/rNTU_view.ckpt-6000")
            print("Model restored.")

            stt_loop = time.time()
            print strftime("%Y%m%d_%H:%M:%S", localtime())

            csvWriter.writerow(['Time', 'Epoch #', 'Epoch Time', 'Train Accuracy', 'Train Cost'])
            for i in range(config.max_max_epoch):

                stt_lr = time.time()

                if i == 0:
                    print "First Learning Rate is assigned!!"
                    m.assign_lr(session, config.learning_rate)
                elif i == config.max_epoch1:
                    m.assign_lr(session, config.learning_rate2)
                elif i == config.max_epoch2:
                    m.assign_lr(session, config.learning_rate3)
                elif i == config.max_epoch3:
                    m.assign_lr(session, config.learning_rate4)
                elif i == config.max_epoch4: #6000
                    print "6000 Learning Rate is assigned!!"
                    m.assign_lr(session, config.learning_rate5)
                elif i == config.max_epoch5: #10,000
                    m.assign_lr(session, config.learning_rate6)
                elif i == config.max_epoch6: #10,000
                    m.assign_lr(session, config.learning_rate7)

                stt_epoch = time.time()

                if i == 0:
                    print "I'm Ready for First Epoch"
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
                    strtime =  strftime("%Y%m%d_%H:%M:%S", localtime())
                    print strtime
                    print("----------Epoch Time: %.3f, per Assign: %.3f, per Epoch: %.3f" % (
                    (end_loop - stt_loop), (stt_epoch - stt_lr), (end_epoch - stt_epoch)))
                    print("Epoch: %d Learning rate: %.6f Train Accuracy: %.4f" % (i, session.run(m.lr), train_accuracy))
                    # train_cost = train_cost * config.batch_size / len(feature_train)
                    sys.stdout.write("Train Cost: %.6f " % train_cost)

                    stt_loop = time.time()
                    print("\n")

                    csvWriter.writerow([strtime, i, (end_epoch - stt_epoch), train_accuracy, train_cost])

                if i % 100 == 0:
                    test_cost, test_accuracy, te_p_l, te_g_l = run_epoch(session,
                                                                                                                mtest,
                                                                                                                feature_test,
                                                                                                                AS_test_label,
                                                                                                                tf.no_op(), is_training=False)
                    print("Test Accuracy: %.5f\n" % (test_accuracy))
                    csvWriter.writerow(["Test Accuracy :", test_accuracy])

                    confusion_matrix = np.zeros([config.class_size, config.class_size + 1])
                    class_prob = np.zeros([config.class_size])
                    for j in range(len(te_g_l)):
                        confusion_matrix[te_g_l[j]][te_p_l[j]] += 1
                    for j in range(config.class_size):
                        class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:config.class_size])
                    # for j in range(config.class_size):
                    #     confusion_matrix[j][config.class_size] = class_prob[j]
                    #     print class_prob[j]*100

                    with open(folder_path + "/view-test-" + str(i) + ".csv", "wb") as csvfile:
                        csvwriter2 = csv.writer(csvfile)
                        for j in range(config.class_size):
                            csvwriter2.writerow(confusion_matrix[j])

    f.close()

                    # if i % 100 == 0 and i==-1:
                    #     stt_test = time.time()
                    #     test_cost_L, test_cost_M, test_cost_S, test_cost, test_accuracy, te_p_l, te_g_l = run_epoch(session,
                    #                                                                                                 mtest,
                    #                                                                                                 feature_test,
                    #                                                                                                 AS_test_label,
                    #                                                                                                 tf.no_op())
                    #     end_test = time.time()
                    #     print("Test Accuracy: %.5f" % (test_accuracy))
                    #
                    #     confusion_matrix = np.zeros([config.class_size, config.class_size + 1])
                    #     class_prob = np.zeros([config.class_size])
                    #     for j in range(len(te_g_l)):
                    #         confusion_matrix[te_g_l[j]][te_p_l[j]] += 1
                    #     for j in range(config.class_size):
                    #         class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:config.class_size])
                    #     for j in range(config.class_size):
                    #         confusion_matrix[j][config.class_size] = class_prob[j]
                    #         print class_prob[j] * 100
                    #
                    #     stt_csv = time.time()
                    #
                    #     with open('./view_mp/NTU_view-' + str(i) + '.csv', 'wb') as csvfile:
                    #         csvwriter = csv.writer(csvfile)
                    #         for j in range(config.class_size):
                    #             csvwriter.writerow(confusion_matrix[j])
                    #
                    #     end_csv = time.time()
                    #     print("Test Time: %.3f, print time: %.3f, csv time: %.3f" % ((end_test - stt_test), (stt_csv - end_test), (end_csv - stt_csv)))

if __name__ == "__main__":
    tf.app.run()