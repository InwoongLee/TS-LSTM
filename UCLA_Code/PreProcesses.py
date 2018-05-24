import numpy as np

def one_hot_labeling(original_label, config):
    new_label = np.zeros([len(original_label), config.class_size])

    for batch_step in range(len(original_label)):
        # print original_label[batch_step]
        new_label[batch_step][original_label[batch_step]-1] = 1

    return new_label


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

def body_rotation(feature_train):
    ret1, _ = body_rotation_2(feature_train, feature_train)
    return ret1


def body_rotation_2(feature_train, feature_test):
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

                for nodeNo in range(np.int32(feature_train.shape[2] / 3)):
                    # feature_hc_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                    #     = np.transpose(
                    #     np.dot(rot, feature_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T))
                    feature_hc_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                     new_origin[
                                                                                                                         1],
                                                                                                                     0]

                first_index = True
            elif (np.linalg.norm(feature_train[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(np.int32(feature_train.shape[2] / 3)):
                    # feature_hc_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                    #     = np.transpose(
                    #     np.dot(rot, feature_train[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T))
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
                v1 = [0, 1, 0]
                u2 = feature_test[batchNo][frmNo][36:39] - feature_test[batchNo][frmNo][48:51]
                v2 = u2 - [np.inner(u2, v1) * v1[0], np.inner(u2, v1) * v1[1], np.inner(u2, v1) * v1[2]]
                v2 = v2 / np.linalg.norm(v2)
                v3 = np.cross(v2, v1)

                rot = np.linalg.inv([[v2[0], v1[0], v3[0]], [v2[1], v1[1], v3[1]], [v2[2], v1[2], v3[2]]])

                for nodeNo in range(np.int32(feature_test.shape[2] / 3)):
                    # print(feature_hc_test[batchNo][frmNo][3*nodeNo:3*(nodeNo+1)])
                    feature_hc_test[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)] \
                        = np.transpose(
                        np.dot(rot, feature_test[batchNo][frmNo][3 * nodeNo:3 * (nodeNo + 1)].T - new_origin.T)) + [0,
                                                                                                                    new_origin[
                                                                                                                        1],
                                                                                                                    0]
                first_index = True
            elif (np.linalg.norm(feature_test[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(np.int32(feature_test.shape[2] / 3)):
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


def feature_only_diff_0(data, maxValue, config):
    new_data = np.zeros([len(data), maxValue, config.feature_size])
    # print(len(data), len(data[0]))
    for batch_step in range(len(data)):
        # print(len(data[batch_step]), len(data[batch_step][0]))
        # print(len(new_data[batch_step][maxValue-len(data[batch_step]):maxValue]), len(new_data[batch_step][maxValue-len(data[batch_step]):maxValue][0]))
        new_data[batch_step][maxValue-len(data[batch_step]):maxValue] = data[batch_step]
        # print(len(data[batch_step]))

    return new_data

def Pose_Motion(feature):
    Feature_PM = np.zeros(feature.shape)
    for time_step in range(feature.shape[1]):
        if time_step > 0:
            Feature_PM[:, time_step, :] = feature[:, time_step, :] - feature[:, time_step - 1, :]
        else:
            Feature_PM[:, time_step, :] = feature[:, time_step, :]

    return Feature_PM

def feature_only_diff(data, maxValue, config):
    new_data = np.zeros([len(data), maxValue, config.input_size])

    for batch_step in range(len(data)):
        # print len(data[batch_step])
        # print(len(data[0]))
        new_data[batch_step][maxValue-len(data[batch_step]):maxValue] = data[batch_step]
        for time_step in range(maxValue):
            if np.sum(new_data[batch_step][time_step]) != 0:
                for ttime_step in range(time_step):
                    new_data[batch_step][ttime_step] = new_data[batch_step][time_step]
                break
            else:
                pass

    return new_data
