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

from PreProcesses import Human_Cognitive_Coordinate, bone_length, extract_AS, extract_data_label, extract_feature, count_zeros, feature_only_diff_0, one_hot_labeling

# from ops import batch_norm_rnn, batch_norm

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

win_size = [167]
stride = [100]
start_time = [1]

train_set = [int(sys.argv[1]), int(sys.argv[2])]
test_set1 = [int(sys.argv[3])]
test_set2 = [int(sys.argv[4])]

# train_set = [1, 2]
# test_set1 = [3]
# test_set2 = [4]

_now = datetime.now()
BatchDivider = 30
myKeepProb = 1.0
myHiddenSize = 50
view_subject = 'LSTM_SMF'
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
            sw_0 = '/gpu:0'
        else:
            sw_0 = '/cpu:0'

        with tf.device(sw_0):
            with tf.name_scope('%s_%d' % ('mGPU', 0)) as scope:
                self.mL = mL = TS_Long_LSTM_Top(is_training, config, labels, self.input_data, self.targets, size)
                logits_l, cross_entropy_l, state_0 = mL.get_inference()
                self._logits_0, self._cross_entropy_0, self._final_state = \
                    logits_0, cross_entropy_0, final_state = logits_l, cross_entropy_l, state_0

                self._cost_L = tf.reduce_sum(cross_entropy_l / 3) / batch_size

                if is_training:
                    self.tvars_L = tf.trainable_variables()
                    count = len(self.tvars_L)
                    print("Long : ", len(self.tvars_L))
                    # print(self.tvars_L)
                    self.grad_before_sum_L = tf.gradients(self._cost_L, self.tvars_L)
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
            with tf.device(gradient_device[3]):
                with tf.name_scope("train") as scope:
                    with tf.name_scope("Merging_Gradient"):
                        self.grad_after_sum = self.grad_before_sum_L
                        self._lr = tf.Variable(0.0, trainable=False)
                        self.grads, _ = tf.clip_by_global_norm(self.grad_after_sum, config.max_grad_norm)
                        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                        self.tvars = tf.trainable_variables()
                        with tf.name_scope("Applying-Gradient"):
                            # print("List : ", self.grads)
                            self._train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars))

        print("Calculating Graph is fully connected!!")

    def assign_lr(self, sess, lr_value):
        sess.run(tf.assign(self.lr, lr_value))

    def mp_init(self, sess):
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


class TS_Long_LSTM_Top(object):
    def __init__(self, is_training, config, labels, top_input_data, top_targets, top_hidden_size):
        if is_training:
            sw_0 = '/gpu:0'
        else:
            sw_0 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps  # 76
            self.class_size = class_size = config.class_size
            size = top_hidden_size  # 20

            self.input_data = top_input_data
            self.targets = top_targets

            num_LSTMs_0 = 1

        # self.scale_intput = scale_intput = tf.Variable(80.0, name="scale_intput")
        with tf.variable_scope("Long_Sliding"):
            print("Long_Sliding_Top")

            with tf.device(sw_0):
                with tf.name_scope('%s_%d' % ('Long0_GPU', device_index[0])) as scope:
                    with tf.variable_scope('l0'):
                        self.mL0 = mL0 = TS_LSTM_Long_0(is_training, config, self.input_data)
                        self._initial_state = mL0.initial_state
                        output_depthconcat_long_0 = mL0.get_depth_concat_output()
                        self.output_depthconcat_long_0 = output_depthconcat_long_0

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
        num_LSTMs_0 = 1
        print("num_LSTMs_0: ", num_LSTMs_0)
        for time_step in range(num_steps):
            for win_step in range(num_LSTMs_0):
                if time_step == 0:
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
                        distance = inputs[:, time_step, :] - inputs[:, time_step - start_time_0, :]
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
    learning_rate = 0.0001
    learning_rate2 = 0.00005
    learning_rate3 = 0.00001
    learning_rate4 = 0.000005
    learning_rate5 = 0.000001
    learning_rate6 = 0.0000005
    learning_rate7 = 0.0000001
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
    max_epoch2 = 2001
    max_epoch3 = 4001
    max_epoch4 = 6000
    max_epoch5 = 7000
    max_epoch6 = 8000
    max_max_epoch = 5001
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

    feature_train = feature_only_diff_0(train_sklt0, MAX_LENGTH, config)
    feature_test = feature_only_diff_0(test_sklt0, MAX_LENGTH, eval_config)
    feature_test2 = feature_only_diff_0(test_sklt02, MAX_LENGTH, eval_config2)


    # for i in range(len(train_sklt1)):
    #     np.save("temp_Data/%03d_train_sklt1.npy"%i,np.asarray(train_sklt1[i]) )

    del train_sklt0, test_sklt0, test_sklt02

    AS_train_label = one_hot_labeling(train_label0, config)
    AS_test_label = one_hot_labeling(test_label0, eval_config)
    AS_test_label2 = one_hot_labeling(test_label02, eval_config2)

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