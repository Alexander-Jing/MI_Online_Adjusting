#data set class

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import os
import scipy.io as sio
from scipy import interpolate

class brain_dataset(Dataset):

    def __init__(self, instance_list, label_list):
        self.instance_list = instance_list
        self.instance_label = label_list
           
    def __getitem__(self, index):
        return self.instance_list[index], self.instance_label[index]
    
    def __len__(self):
        return len(self.instance_list)
    
    def __get_instance_label__(self):
        return self.instance_label
    
    def __get_instance_list__(self):
        return np.array(self.instance_list).shape
    
def read_subject_csv(path, select_feature_columns = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O',
       'CD_I_DO', 'CD_PHI_DO'], num_chunk_this_window_size = 2224, verbose=False):
    
    '''
    For binary classification: 0 vs 2
    '''
    
    instance_list = []
    instance_label = []
    
    # each subject csv file contain 2224 chunks (for window size 10 stride 3)
    subject_df = pd.read_csv(path) 
    assert np.max(subject_df.chunk.values) + 1 == num_chunk_this_window_size, '{} does not have {} chunks'.format(path, num_chunk_this_window_size) 
    
    subject_df = subject_df[select_feature_columns + ['chunk'] + ['label']] 

    #chunk id: 0 to 2223 (for window size 10 stride 3)
    for i in range(0, num_chunk_this_window_size):
        if verbose:
            print('current chunk: {}'.format(i))
        chunk_matrix = subject_df.iloc[:,:-2].loc[subject_df['chunk'] == i].values
        label_for_this_segment = subject_df[subject_df['chunk'] == i].label.values[0]
        
        instance_list.append(chunk_matrix)        
        instance_label.append(label_for_this_segment)


#
            
    instance_list = np.array(instance_list, dtype=np.float32) 
    instance_label = np.array(instance_label, dtype=np.int64)
    
#     print('Inside brain_data, this subject data size: {}'.format(instance_label.shape[0]), flush = True)
#     assert instance_label.shape[0] == 1112
    
    
    return instance_list, instance_label



def read_subject_csv_binary(path, select_feature_columns = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O',
       'CD_I_DO', 'CD_PHI_DO'], num_chunk_this_window_size = 2224, verbose=False):
    
    '''
    For binary classification: 0 vs 2
    '''
    
    instance_list = []
    instance_label = []
    
    # each subject csv file contain 2224 chunks (for window size 10 stride 3)
    subject_df = pd.read_csv(path) 
    assert np.max(subject_df.chunk.values) + 1 == num_chunk_this_window_size, '{} does not have {} chunks'.format(path, num_chunk_this_window_size) 
    
    subject_df = subject_df[select_feature_columns + ['chunk'] + ['label']] 

    #chunk id: 0 to 2223 (for window size 10 stride 3)
    for i in range(0, num_chunk_this_window_size):
        if verbose:
            print('current chunk: {}'.format(i))
        chunk_matrix = subject_df.iloc[:,:-2].loc[subject_df['chunk'] == i].values
        label_for_this_segment = subject_df[subject_df['chunk'] == i].label.values[0]
        
        if label_for_this_segment == 0:
            if verbose:
                print('label_for_this_segment is {}'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(label_for_this_segment)
        elif label_for_this_segment == 2:
            if verbose:
                print('label_for_this_segment is {}, map to class1'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(int(1))

#
            
    instance_list = np.array(instance_list, dtype=np.float32) 
    instance_label = np.array(instance_label, dtype=np.int64)
    
#     print('Inside brain_data, this subject data size: {}'.format(instance_label.shape[0]), flush = True)
#     assert instance_label.shape[0] == 1112
    
    
    return instance_list, instance_label



def read_subject_csv_binary_SelectWindowSize(path, select_feature_columns = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O',
       'CD_I_DO', 'CD_PHI_DO']):
    
    '''
    For binary classification: 0 vs 2
    '''
    
    instance_list = []
    instance_label = []
    
    # each subject csv file contain 608 chunks 
    subject_df = pd.read_csv(path) 
    assert np.max(subject_df.chunk.values) + 1 == 608, '{} SelectWindowSize testset does not have 608 chunks'.format(path) 
    
    subject_df = subject_df[select_feature_columns + ['chunk'] + ['label']] 

    #chunk id: 0 to 608
    for i in range(0, 608):
        chunk_matrix = subject_df.iloc[:,:-2].loc[subject_df['chunk'] == i].values
        
        assert len(list(set(subject_df[subject_df['chunk'] == i].label.values))) == 1, 'each chunk has only 1 label'
        label_for_this_segment = subject_df[subject_df['chunk'] == i].label.values[0]
        
        if label_for_this_segment == 0:
            print('label_for_this_segment is {}'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(label_for_this_segment)
        elif label_for_this_segment == 2:
            print('label_for_this_segment is {}, map to class1'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(int(1))

#
            
    instance_list = np.array(instance_list, dtype=np.float32) 
    instance_label = np.array(instance_label, dtype=np.int64)
    
#     print('Inside brain_data, this subject SelectWindowSize testset size: {}'.format(instance_label.shape[0]), flush = True)
#     assert instance_label.shape[0] == 304
    
    
    return instance_list, instance_label


def read_subject_csv_binary_chunk(path, select_feature_columns = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O',
       'CD_I_DO', 'CD_PHI_DO'], num_chunk_this_window_size = 2224, verbose=False):
    
    '''
    For binary classification: 0 vs 2
    '''
    
    instance_list = []
    instance_label = []
    
    # each subject csv file contain 2224 chunks (for window size 10 stride 3)
    subject_df = pd.read_csv(path) 
    assert np.max(subject_df.chunk.values) + 1 == num_chunk_this_window_size, '{} does not have {} chunks'.format(path, num_chunk_this_window_size) 
    
    subject_df = subject_df[select_feature_columns + ['chunk'] + ['label']] 

    #chunk id: 0 to 2223 (for window size 10 stride 3)
    for i in range(0, num_chunk_this_window_size):
        if verbose:
            print('current chunk: {}'.format(i))
        chunk_matrix = subject_df.iloc[:,:-2].loc[subject_df['chunk'] == i].values
        label_for_this_segment = subject_df[subject_df['chunk'] == i].label.values[0]
        chunk_for_this_segment = subject_df[subject_df['chunk'] == i].chunk.values[0]  # load the chunk numbers

        if label_for_this_segment == 0:
            if verbose:
                print('label_for_this_segment is {}'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append([label_for_this_segment, int(chunk_for_this_segment/372.0)])
        elif label_for_this_segment == 2:
            if verbose:
                print('label_for_this_segment is {}, map to class1'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append([int(1), int(chunk_for_this_segment/372.0)])

#
            
    instance_list = np.array(instance_list, dtype=np.float32) 
    instance_label = np.array(instance_label, dtype=np.int64)
    
#     print('Inside brain_data, this subject data size: {}'.format(instance_label.shape[0]), flush = True)
#     assert instance_label.shape[0] == 1112
    
    
    return instance_list, instance_label


def MixUp_expansion(prior_sub_feature_array, prior_sub_label_array, alpha = 0.75, expand=2):
    
    '''
    Mixing strategy1: mixing same chunk of different person to create synthetic person
                      randomly choose two person, sample lambda from beta distribution, use the same beta for each chunk
    '''
    # Make sure same number of subjects
    assert len(prior_sub_feature_array) == len(prior_sub_label_array)
    assert isinstance(prior_sub_feature_array, np.ndarray), 'input_images is not numpy array'
    assert isinstance(prior_sub_label_array, np.ndarray), 'input_labels is not numpy array'

    expanded_sub_feature_array = None
    expanded_sub_label_array = None
    
    num_sub = len(prior_sub_feature_array)
    
    for i in range(expand):
        # generate a different random lambda value for each subject
        lam = np.random.beta(alpha, alpha, (num_sub, 1, 1, 1))
        lam = np.maximum(lam, (1 - lam)) #ensure the created samples is closer to the first sample

        permutation_indices = np.random.permutation(num_sub)

        #linear interpolation of features
        synthetic_sub_feature_array = prior_sub_feature_array * lam + prior_sub_feature_array[permutation_indices] * (1 - lam)

        #linear interpolation of labels
        synthetic_sub_label_array = prior_sub_label_array * lam[:, :, 0, 0] + prior_sub_label_array[permutation_indices] * (1 - lam[:, :, 0, 0])  

        if expanded_sub_feature_array is None:
            expanded_sub_feature_array = synthetic_sub_feature_array
            expanded_sub_label_array = synthetic_sub_label_array
        else:     
            expanded_sub_feature_array = np.concatenate((expanded_sub_feature_array, synthetic_sub_feature_array))
            expanded_sub_label_array = np.concatenate((expanded_sub_label_array, synthetic_sub_label_array))
    
    return expanded_sub_feature_array, expanded_sub_label_array


def Offline_read_csv(folder_path, windows, proportion):
    """
    read the data from offline data collection 
    parameters:
        folder_path: the path of the offline collected data
        windows: the num of windows per class
        proportion: the proportion of the training and validation set,  proportion*windows for training , (1-proportion)*windows for validation
    returns:
        train_list, train_label: the np.array of the collected data and labels for training 
        val_list, val_label: the np.array of the collected data and the labels for training 
    """
    # initialize the training and validation list
    train_list = []
    train_label = []
    val_list = []
    val_label = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the filename follows the format class_i_window_j.csv
        if filename.startswith("class_") and filename.endswith(".csv"):
            # Extract the class identifier i and window identifier j from the filename
            class_id = int(filename.split("_")[1])
            window_id = int(filename.split("_")[3].split(".")[0])
            # Read the csv file using pandas
            data = pd.read_csv(os.path.join(folder_path, filename), header=None).values

            # Distribute the data to the training set or validation set based on the window identifier j
            # assert proportion * windows % 1 == 0, "The result of proportion * windows must be an integer."
            
            if window_id < int(proportion * windows):
                train_list.append(data)
                train_label.append(class_id)
            else:
                val_list.append(data)
                val_label.append(class_id)

    # Convert the lists to np.array
    train_list = np.array(train_list)
    train_label = np.array(train_label)
    val_list = np.array(val_list)
    val_label = np.array(val_label)

    return train_list, train_label, val_list, val_label


def Online_read_csv(folder_path, session, trial):
    """
    read the data from Online data collection 
    parameters:
        folder_path: the path of the online collected data
        session: the selected session 
        trial: the selected trial 
    returns:
        train_list, train_label: the np.array of the collected data and labels for training 
        scores: the np.array of the collected scores
    """
    train_list = []
    train_label = []
    scores = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Extract information from the filename
        parts = filename.split('_')
        label = int(parts[1])
        session_num = int(parts[3])
        trial_num = int(parts[5])

        # Check if session and trial match
        if session_num == session and trial_num == trial:
            # Read the data
            data = pd.read_csv(os.path.join(folder_path, filename), header=None).values
            # Extract the first n-1 rows of data
            data_sub = data[:-1]
            # Add to train_list
            train_list.append(data_sub)
            # Add to train_label
            train_label.append(label)
            # Extract and add score
            score = float( (parts[9]).split('.csv')[0] )
            scores.append(score)
            print('get csv file: ' + filename)
            """
            
            print('label: ' + str(label))
            print('session: ' + str(session_num))
            print('trial_num: ' + str(trial_num))
            print('score: ' + str(score))
            """
            
    return np.array(train_list), np.array(train_label), np.array(scores)


def Online_simulation_read_csv(folder_path, sub_file, trial_pre, proportion=0.8):
    # 读取.mat文件
    mat = sio.loadmat(os.path.join(folder_path, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
    
    # 提取数据
    task_data = mat['task_data'].reshape(-1, 40, 80)
    task_label = mat['task_label'].reshape(-1, 1)
    rest_data = mat['rest_data'].reshape(-1, 62, 800)

    # 提取类别为1和2的前trial_pre个trial的数据
    task_data_pre_1 = task_data[task_label[:, 0] == 1][:trial_pre]
    task_data_pre_2 = task_data[task_label[:, 0] == 2][:trial_pre]
    task_label_pre_1 = task_label[task_label[:, 0] == 1][:trial_pre]
    task_label_pre_2 = task_label[task_label[:, 0] == 2][:trial_pre]
    
    # 构成数据集
    sub_train_feature_array = np.concatenate((task_data_pre_1, task_data_pre_2, rest_data[:trial_pre]), axis=0)
    sub_train_label_array = np.concatenate((task_label_pre_1, task_label_pre_2, np.zeros((trial_pre, 1))), axis=0)
    
    # 剩下的数据
    task_data_rest_1 = task_data[task_label[:, 0] == 1][trial_pre:]
    task_data_rest_2 = task_data[task_label[:, 0] == 2][trial_pre:]
    task_label_rest_1 = task_label[task_label[:, 0] == 1][trial_pre:]
    task_label_rest_2 = task_label[task_label[:, 0] == 2][trial_pre:]
    rest_data_rest = rest_data[trial_pre:]
    
    sub_train_feature_array_1 = np.concatenate((task_data_rest_1, task_data_rest_2, rest_data_rest), axis=0)
    sub_train_label_array_1 = np.concatenate((task_label_rest_1, task_label_rest_2, np.zeros((len(task_data_rest_1) + len(task_data_rest_2), 1))), axis=0)
    
    # 将原来的训练集以proportion的比例拆分为新的训练集和验证集
    split_index = int(proportion * len(sub_train_feature_array))
    sub_val_feature_array = sub_train_feature_array[split_index:]
    sub_val_label_array = sub_train_label_array[split_index:]
    sub_train_feature_array = sub_train_feature_array[:split_index]
    sub_train_label_array = sub_train_label_array[:split_index]
    
    return sub_train_feature_array, sub_train_label_array, sub_val_feature_array, sub_val_label_array, \
        sub_train_feature_array_1, sub_train_label_array_1


def preprocess_eeg_data(eeg_data, channel_list, target_channel_list, max_scale, high=1, low=-1, length=4, old_freq=200, new_freq=256):
    # 选择目标通道
    target_indices = [channel_list.index(ch) for ch in target_channel_list if ch in channel_list]
    eeg_data = eeg_data[:, target_indices, :]
    eeg_data_processed = []

    for i in range(eeg_data.shape[0]):
        _eeg_data = eeg_data[i,:,:]
        # 归一化数据
        xmin = _eeg_data.min()
        xmax = _eeg_data.max()
        _eeg_data = (_eeg_data - xmin) / (xmax - xmin)
        _eeg_data -= 0.5
        _eeg_data += (high + low) / 2
        _eeg_data *= (high - low)
        
        # 添加scale行
        scale = 2 * (np.clip((_eeg_data.max() - _eeg_data.min()) / max_scale, 0, 1.0) - 0.5)
        _eeg_data = np.vstack((_eeg_data, np.full((1, _eeg_data.shape[1]), scale)))
        
        # 插值到新的采样频率
        if old_freq != new_freq:
            old_time = np.linspace(0, length, old_freq*length)
            new_time = np.linspace(0, length, new_freq*length)
            interpolator = interpolate.interp1d(old_time, _eeg_data, axis=1)
            _eeg_data = interpolator(new_time)

        eeg_data_processed.append(_eeg_data)
    
    eeg_data_processed = np.array(eeg_data_processed)

    return eeg_data_processed

def preprocess_eeg_data_Norm(eeg_data, channel_list, target_channel_list, max_scale, Norm=False, high=1, low=-1, length=4, old_freq=200, new_freq=256):
    # 选择目标通道
    target_indices = [channel_list.index(ch) for ch in target_channel_list if ch in channel_list]
    eeg_data = eeg_data[:, target_indices, :]
    eeg_data_processed = []
    eeg_data_max = []
    eeg_data_min = []

    for i in range(eeg_data.shape[0]):
        _eeg_data = eeg_data[i,:,:]
        
        if Norm:
            # 归一化数据
            xmin = _eeg_data.min()
            xmax = _eeg_data.max()
            eeg_data_max.append(xmax)
            eeg_data_min.append(xmin)

            _eeg_data = (_eeg_data - xmin) / (xmax - xmin)
            _eeg_data -= 0.5
            _eeg_data += (high + low) / 2
            _eeg_data *= (high - low)
            
            # 添加scale行
            scale = 2 * (np.clip((xmax - xmin) / max_scale, 0, 1.0) - 0.5)
            _eeg_data = np.vstack((_eeg_data, np.full((1, _eeg_data.shape[1]), scale)))
        
        # 插值到新的采样频率
        if old_freq != new_freq:
            old_time = np.linspace(0, length, old_freq*length)
            new_time = np.linspace(0, length, new_freq*length)
            interpolator = interpolate.interp1d(old_time, _eeg_data, axis=1)
            _eeg_data = interpolator(new_time)

        eeg_data_processed.append(_eeg_data)
    
    eeg_data_processed = np.array(eeg_data_processed)
    
    return eeg_data_processed, eeg_data_max, eeg_data_min

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    if axis >= data.ndim:
        raise ValueError(
            "Axis value is out of range. It should be within the range of input data dimensions")
    if stepsize < 1:
        raise ValueError("Stepsize should be >= 1.")
    if size > data.shape[axis]:
        raise ValueError("The size of the window should not exceed the size of selected axis.")
    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)
    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])
    strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    if copy:
        return strided.copy()
    else:
        return strided

def sliding_window_modified(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    if axis >= data.ndim:
        raise ValueError(
            "Axis value is out of range. It should be within the range of input data dimensions")
    if stepsize < 1:
        raise ValueError("Stepsize should be >= 1.")
    if size > data.shape[axis]:
        raise ValueError("The size of the window should not exceed the size of selected axis.")
    
    windowed_data = []
    for i in range(data.shape[0]):
        _data = data[i,:,:]
        window_num = int((_data.shape[-1]-size)/stepsize + 1)
        for j in range(window_num):
            _window = _data[:,j*stepsize:j*stepsize+size]
            windowed_data.append(_window)
    
    windowed_data = np.stack(windowed_data, axis=0)

    return windowed_data


CHANNEL_LIST = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', \
                'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', \
                'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',\
                'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', \
                'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', \
                'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', \
                'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', \
                'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2',
]

EEG_20_DIV_32 = [
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'FT7','FC3','FCZ','FC4','FT8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'TP7','CP3','CPZ','CP4','TP8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

def Online_simulation_read_csv_windows(folder_path, sub_file, trial_pre, max_scale=128-(-128), \
                                       channel_list=CHANNEL_LIST, target_channel_list=EEG_20_DIV_32, proportion=0.8,\
                                        window_size = 512, step_size = 256, batch_size_online=4,\
                                            pattern = [0, 1, 2,  0, 2, 1,  1, 0, 2,  1, 2, 0,  2, 0, 1,  2, 1, 0]):
    # 读取.mat文件
    mat = sio.loadmat(os.path.join(folder_path, 'sub-' + sub_file, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
    
    # 提取数据
    task_data = mat['task_data'].reshape(-1, 62, 800)
    task_label = mat['task_label'].reshape(-1, 1)
    rest_data = mat['rest_data'].reshape(-1, 62, 800)
    
    # 预处理数据
    task_data = preprocess_eeg_data(task_data, channel_list, target_channel_list, max_scale)
    rest_data = preprocess_eeg_data(rest_data, channel_list, target_channel_list, max_scale)
    
    # 提取类别为1和2的前trial_pre个trial的数据
    task_data_pre_1 = task_data[task_label[:, 0] == 1][:trial_pre]
    task_data_pre_2 = task_data[task_label[:, 0] == 2][:trial_pre]
    task_label_pre_1 = task_label[task_label[:, 0] == 1][:trial_pre]
    task_label_pre_2 = task_label[task_label[:, 0] == 2][:trial_pre]
    
    # 构成数据集
    sub_train_feature_array = np.concatenate((task_data_pre_1, task_data_pre_2, rest_data[:trial_pre]), axis=0)
    sub_train_label_array = np.concatenate((task_label_pre_1, task_label_pre_2, np.zeros((trial_pre, 1))), axis=0)
    
    # 剩下的数据
    task_data_rest_1 = task_data[task_label[:, 0] == 1][trial_pre:]
    task_data_rest_2 = task_data[task_label[:, 0] == 2][trial_pre:]
    task_label_rest_1 = task_label[task_label[:, 0] == 1][trial_pre:]
    task_label_rest_2 = task_label[task_label[:, 0] == 2][trial_pre:]
    rest_data_rest = rest_data[trial_pre:]
    
    sub_train_feature_array_1 = np.concatenate((task_data_rest_1, task_data_rest_2, rest_data_rest), axis=0)
    sub_train_label_array_1 = np.concatenate((task_label_rest_1, task_label_rest_2, np.zeros((len(rest_data_rest), 1))), axis=0)
    
    # 使用滑窗增广数据
    sub_train_feature_array = sliding_window(sub_train_feature_array, window_size, step_size)
    sub_train_feature_array = sub_train_feature_array.reshape(-1, sub_train_feature_array.shape[1], sub_train_feature_array.shape[3])

    sub_train_label_array = np.repeat(sub_train_label_array, sub_train_feature_array.shape[0] // len(sub_train_label_array))
    #sub_val_feature_array = sliding_window(sub_val_feature_array, window_size, step_size)
    #sub_val_label_array = np.repeat(sub_val_label_array, sub_val_feature_array.shape[0] // len(sub_val_label_array))
    sub_train_feature_array_1 = sliding_window(sub_train_feature_array_1, window_size, step_size)
    sub_train_feature_array_1 = sub_train_feature_array_1.reshape(-1, sub_train_feature_array_1.shape[1], sub_train_feature_array_1.shape[3])

    sub_train_label_array_1 = np.repeat(sub_train_label_array_1, sub_train_feature_array_1.shape[0] // len(sub_train_label_array_1))
    
    # 重新排列数据
    # 获取每个类别的索引
    indices_0 = np.where(sub_train_label_array_1 == 0)[0]
    indices_1 = np.where(sub_train_label_array_1 == 1)[0]
    indices_2 = np.where(sub_train_label_array_1 == 2)[0]

    # 创建新的索引数组
    new_indices = []
    
    # 按照顺序每次添加batch_size_online个样本用于在线更新
    for i in range(len(sub_train_label_array_1) // (len(pattern) * batch_size_online)):
        for j in pattern:
            if j == 0 and len(indices_0) >= batch_size_online:
                new_indices.extend(indices_0[:batch_size_online])
                indices_0 = indices_0[batch_size_online:]
            elif j == 1 and len(indices_1) >= batch_size_online:
                new_indices.extend(indices_1[:batch_size_online])
                indices_1 = indices_1[batch_size_online:]
            elif j == 2 and len(indices_2) >= batch_size_online:
                new_indices.extend(indices_2[:batch_size_online])
                indices_2 = indices_2[batch_size_online:]
    
    # 使用新的索引数组重新排列 sub_train_feature_array_1 和 sub_train_label_array_1
    sub_train_feature_array_1 = sub_train_feature_array_1[new_indices]
    sub_train_label_array_1 = sub_train_label_array_1[new_indices]

    # 将原来的训练集以proportion的比例拆分为新的训练集和验证集
    unique_labels = np.unique(sub_train_label_array)
    sub_train_feature_list = []
    sub_train_label_list = []
    sub_val_feature_list = []
    sub_val_label_list = []
    for label in unique_labels:
        indices = np.where(sub_train_label_array == label)[0]
        split_index = int(proportion * len(indices))
        sub_train_feature_list.append(sub_train_feature_array[indices[:split_index]])
        sub_train_label_list.append(sub_train_label_array[indices[:split_index]])
        sub_val_feature_list.append(sub_train_feature_array[indices[split_index:]])
        sub_val_label_list.append(sub_train_label_array[indices[split_index:]])
    sub_train_feature_array = np.concatenate(sub_train_feature_list, axis=0)
    sub_train_label_array = np.concatenate(sub_train_label_list, axis=0)
    sub_val_feature_array = np.concatenate(sub_val_feature_list, axis=0)
    sub_val_label_array = np.concatenate(sub_val_label_list, axis=0)

    return sub_train_feature_array, sub_train_label_array.astype(int), sub_val_feature_array, sub_val_label_array.astype(int), \
        sub_train_feature_array_1, sub_train_label_array_1.astype(int)

def Online_simulation_read_csv_windows_preprocess(folder_path, sub_file, trial_pre, preprocess=False, max_scale=128-(-128), \
                                       channel_list=CHANNEL_LIST, target_channel_list=EEG_20_DIV_32, proportion=0.8,\
                                        window_size = 512, step_size = 256, batch_size_online=4,\
                                            pattern = [0, 1, 2,  0, 2, 1,  1, 0, 2,  1, 2, 0,  2, 0, 1,  2, 1, 0]):
    # 读取.mat文件
    mat = sio.loadmat(os.path.join(folder_path, 'sub-' + sub_file, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
    
    # 提取数据
    task_data = mat['task_data'].reshape(-1, 62, 800)
    task_label = mat['task_label'].reshape(-1, 1)
    rest_data = mat['rest_data'].reshape(-1, 62, 800)
    
    # 预处理数据
    if preprocess:
        task_data = preprocess_eeg_data(task_data, channel_list, target_channel_list, max_scale)
        rest_data = preprocess_eeg_data(rest_data, channel_list, target_channel_list, max_scale)
    else:
        task_data,_,_ = preprocess_eeg_data_Norm(task_data, channel_list, target_channel_list, Norm=preprocess, max_scale=max_scale)
        rest_data,_,_ = preprocess_eeg_data_Norm(rest_data, channel_list, target_channel_list, Norm=preprocess, max_scale=max_scale)
    
    # 提取类别为1和2的前trial_pre个trial的数据
    task_data_pre_1 = task_data[task_label[:, 0] == 1][:trial_pre]
    task_data_pre_2 = task_data[task_label[:, 0] == 2][:trial_pre]
    task_label_pre_1 = task_label[task_label[:, 0] == 1][:trial_pre]
    task_label_pre_2 = task_label[task_label[:, 0] == 2][:trial_pre]
    
    # 构成数据集
    sub_train_feature_array = np.concatenate((task_data_pre_1, task_data_pre_2, rest_data[:trial_pre]), axis=0)
    sub_train_label_array = np.concatenate((task_label_pre_1, task_label_pre_2, np.zeros((trial_pre, 1))), axis=0)
    
    # 剩下的数据
    task_data_rest_1 = task_data[task_label[:, 0] == 1][trial_pre:]
    task_data_rest_2 = task_data[task_label[:, 0] == 2][trial_pre:]
    task_label_rest_1 = task_label[task_label[:, 0] == 1][trial_pre:]
    task_label_rest_2 = task_label[task_label[:, 0] == 2][trial_pre:]
    rest_data_rest = rest_data[trial_pre:]
    
    sub_train_feature_array_1 = np.concatenate((task_data_rest_1, task_data_rest_2, rest_data_rest), axis=0)
    sub_train_label_array_1 = np.concatenate((task_label_rest_1, task_label_rest_2, np.zeros((len(rest_data_rest), 1))), axis=0)
    
    # 使用滑窗增广数据
    sub_train_feature_array = sliding_window(sub_train_feature_array, window_size, step_size)
    sub_train_feature_array = sub_train_feature_array.reshape(-1, sub_train_feature_array.shape[1], sub_train_feature_array.shape[3])

    sub_train_label_array = np.repeat(sub_train_label_array, sub_train_feature_array.shape[0] // len(sub_train_label_array))
    #sub_val_feature_array = sliding_window(sub_val_feature_array, window_size, step_size)
    #sub_val_label_array = np.repeat(sub_val_label_array, sub_val_feature_array.shape[0] // len(sub_val_label_array))
    sub_train_feature_array_1 = sliding_window(sub_train_feature_array_1, window_size, step_size)
    sub_train_feature_array_1 = sub_train_feature_array_1.reshape(-1, sub_train_feature_array_1.shape[1], sub_train_feature_array_1.shape[3])

    sub_train_label_array_1 = np.repeat(sub_train_label_array_1, sub_train_feature_array_1.shape[0] // len(sub_train_label_array_1))
    
    # 重新排列数据
    # 获取每个类别的索引
    indices_0 = np.where(sub_train_label_array_1 == 0)[0]
    indices_1 = np.where(sub_train_label_array_1 == 1)[0]
    indices_2 = np.where(sub_train_label_array_1 == 2)[0]

    # 创建新的索引数组
    new_indices = []
    
    # 按照顺序每次添加batch_size_online个样本用于在线更新
    for i in range(len(sub_train_label_array_1) // (len(pattern) * batch_size_online)):
        for j in pattern:
            if j == 0 and len(indices_0) >= batch_size_online:
                new_indices.extend(indices_0[:batch_size_online])
                indices_0 = indices_0[batch_size_online:]
            elif j == 1 and len(indices_1) >= batch_size_online:
                new_indices.extend(indices_1[:batch_size_online])
                indices_1 = indices_1[batch_size_online:]
            elif j == 2 and len(indices_2) >= batch_size_online:
                new_indices.extend(indices_2[:batch_size_online])
                indices_2 = indices_2[batch_size_online:]
    
    # 使用新的索引数组重新排列 sub_train_feature_array_1 和 sub_train_label_array_1
    sub_train_feature_array_1 = sub_train_feature_array_1[new_indices]
    sub_train_label_array_1 = sub_train_label_array_1[new_indices]

    # 将原来的训练集以proportion的比例拆分为新的训练集和验证集
    unique_labels = np.unique(sub_train_label_array)
    sub_train_feature_list = []
    sub_train_label_list = []
    sub_val_feature_list = []
    sub_val_label_list = []
    for label in unique_labels:
        indices = np.where(sub_train_label_array == label)[0]
        split_index = int(proportion * len(indices))
        sub_train_feature_list.append(sub_train_feature_array[indices[:split_index]])
        sub_train_label_list.append(sub_train_label_array[indices[:split_index]])
        sub_val_feature_list.append(sub_train_feature_array[indices[split_index:]])
        sub_val_label_list.append(sub_train_label_array[indices[split_index:]])
    sub_train_feature_array = np.concatenate(sub_train_feature_list, axis=0)
    sub_train_label_array = np.concatenate(sub_train_label_list, axis=0)
    sub_val_feature_array = np.concatenate(sub_val_feature_list, axis=0)
    sub_val_label_array = np.concatenate(sub_val_label_list, axis=0)

    return sub_train_feature_array, sub_train_label_array.astype(int), sub_val_feature_array, sub_val_label_array.astype(int), \
        sub_train_feature_array_1, sub_train_label_array_1.astype(int)

def Online_simulation_read_csv_windows_preprocess_normalization(folder_path, sub_file, trial_pre, preprocess=True, max_scale=128-(-128), old_freq=200, new_freq=256,\
                                       channel_list=CHANNEL_LIST, target_channel_list=EEG_20_DIV_32, proportion=0.8,\
                                        window_size = 512, step_size = 256, window_per_trial = 3, batch_size_online=4, normalize=False,\
                                            pattern = [0, 1, 2,  0, 2, 1,  1, 0, 2,  1, 2, 0,  2, 0, 1,  2, 1, 0]):
    # 读取.mat文件
    mat = sio.loadmat(os.path.join(folder_path, 'sub-' + sub_file, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
    
    # 提取数据
    task_data = mat['task_data'].reshape(-1, 62, 800)
    task_label = mat['task_label'].reshape(-1, 1)
    rest_data = mat['rest_data'].reshape(-1, 62, 800)

    # 预处理数据
    task_data,eeg_data_max, eeg_data_min = preprocess_eeg_data_Norm(task_data, channel_list, target_channel_list, Norm=preprocess, max_scale=max_scale, old_freq=old_freq, new_freq=new_freq)
    rest_data,eeg_data_max_, eeg_data_min_ = preprocess_eeg_data_Norm(rest_data, channel_list, target_channel_list, Norm=preprocess, max_scale=max_scale, old_freq=old_freq, new_freq=new_freq)

    # 提取类别为1和2的前trial_pre个trial的数据
    task_data_pre_1 = task_data[task_label[:, 0] == 1][:trial_pre]
    task_data_pre_2 = task_data[task_label[:, 0] == 2][:trial_pre]
    task_label_pre_1 = task_label[task_label[:, 0] == 1][:trial_pre]
    task_label_pre_2 = task_label[task_label[:, 0] == 2][:trial_pre]
    
    # 构成数据集
    sub_train_feature_array = np.concatenate((task_data_pre_1, task_data_pre_2, rest_data[:trial_pre]), axis=0)
    sub_train_label_array = np.concatenate((task_label_pre_1, task_label_pre_2, np.zeros((trial_pre, 1))), axis=0)
    

    # 剩下的数据
    task_data_rest_1 = task_data[task_label[:, 0] == 1][trial_pre:]
    task_data_rest_2 = task_data[task_label[:, 0] == 2][trial_pre:]
    task_label_rest_1 = task_label[task_label[:, 0] == 1][trial_pre:]
    task_label_rest_2 = task_label[task_label[:, 0] == 2][trial_pre:]
    rest_data_rest = rest_data[trial_pre:]
    
    sub_train_feature_array_1 = np.concatenate((task_data_rest_1, task_data_rest_2, rest_data_rest), axis=0)
    sub_train_label_array_1 = np.concatenate((task_label_rest_1, task_label_rest_2, np.zeros((len(rest_data_rest), 1))), axis=0)

    # 使用滑窗增广数据
    sub_train_feature_array = sliding_window_modified(sub_train_feature_array, window_size, step_size)
    #sub_train_feature_array = sub_train_feature_array.reshape(-1, sub_train_feature_array.shape[1], sub_train_feature_array.shape[3])

    sub_train_label_array = np.repeat(sub_train_label_array, sub_train_feature_array.shape[0] // len(sub_train_label_array))
    #sub_val_feature_array = sliding_window(sub_val_feature_array, window_size, step_size)
    #sub_val_label_array = np.repeat(sub_val_label_array, sub_val_feature_array.shape[0] // len(sub_val_label_array))
    sub_train_feature_array_1 = sliding_window_modified(sub_train_feature_array_1, window_size, step_size)
    #sub_train_feature_array_1 = sub_train_feature_array_1.reshape(-1, sub_train_feature_array_1.shape[1], sub_train_feature_array_1.shape[3])

    sub_train_label_array_1 = np.repeat(sub_train_label_array_1, sub_train_feature_array_1.shape[0] // len(sub_train_label_array_1))

    sub_train_feature_array_backup = sub_train_feature_array.copy()
    sub_train_feature_array_1_backup = sub_train_feature_array_1.copy()

    # temporal normalization 
    if normalize:
        sub_train_feature_array = TemporalNormalization(sub_train_feature_array)
        sub_train_feature_array_1 = TemporalNormalization(sub_train_feature_array_1)

    # 重新排列数据
    # 获取每个类别的索引
    indices_0 = np.where(sub_train_label_array_1 == 0)[0]
    indices_1 = np.where(sub_train_label_array_1 == 1)[0]
    indices_2 = np.where(sub_train_label_array_1 == 2)[0]

    # 创建新的索引数组
    new_indices = []
    
    # 按照顺序每次添加batch_size_online个样本用于在线更新
    for i in range(len(sub_train_label_array_1) // (len(pattern) * batch_size_online)):
        for j in pattern:
            if j == 0 and len(indices_0) >= batch_size_online:
                new_indices.extend(indices_0[:batch_size_online])
                indices_0 = indices_0[batch_size_online:]
            elif j == 1 and len(indices_1) >= batch_size_online:
                new_indices.extend(indices_1[:batch_size_online])
                indices_1 = indices_1[batch_size_online:]
            elif j == 2 and len(indices_2) >= batch_size_online:
                new_indices.extend(indices_2[:batch_size_online])
                indices_2 = indices_2[batch_size_online:]
    
    # 使用新的索引数组重新排列 sub_train_feature_array_1 和 sub_train_label_array_1
    sub_train_feature_array_1 = sub_train_feature_array_1[new_indices]
    sub_train_label_array_1 = sub_train_label_array_1[new_indices]

    # 将原来的训练集以proportion的比例拆分为新的训练集和验证集
    unique_labels = np.unique(sub_train_label_array)
    sub_train_feature_list = []
    sub_train_label_list = []
    sub_val_feature_list = []
    sub_val_label_list = []
    for label in unique_labels:
        indices = np.where(sub_train_label_array == label)[0]
        split_index = int(proportion * len(indices))
        # because of the online windows, the situation that windows from the same trial both in training and validation set should be avoided
        split_index_trial = window_per_trial * (split_index // window_per_trial)  
        sub_train_feature_list.append(sub_train_feature_array[indices[:split_index_trial]])
        sub_train_label_list.append(sub_train_label_array[indices[:split_index_trial]])
        sub_val_feature_list.append(sub_train_feature_array[indices[split_index_trial:]])
        sub_val_label_list.append(sub_train_label_array[indices[split_index_trial:]])
    sub_train_feature_array = np.concatenate(sub_train_feature_list, axis=0)
    sub_train_label_array = np.concatenate(sub_train_label_list, axis=0)
    sub_val_feature_array = np.concatenate(sub_val_feature_list, axis=0)
    sub_val_label_array = np.concatenate(sub_val_label_list, axis=0)

    return sub_train_feature_array, sub_train_label_array.astype(int), sub_val_feature_array, sub_val_label_array.astype(int), \
        sub_train_feature_array_1, sub_train_label_array_1.astype(int)

def Online_simulation_read_csv_windows_preprocess_normalization_1(folder_path, sub_file, trial_pre, preprocess=True, max_scale=128-(-128), old_freq=200, new_freq=256,\
                                       channel_list=CHANNEL_LIST, target_channel_list=EEG_20_DIV_32, proportion=0.8,\
                                        window_size = 512, step_size = 256, batch_size_online=4, normalize=False,\
                                            pattern = [0, 1, 2,  0, 2, 1,  1, 0, 2,  1, 2, 0,  2, 0, 1,  2, 1, 0]):
    # 读取.mat文件
    mat = sio.loadmat(os.path.join(folder_path, 'sub-' + sub_file, 'eeg', 'sub-' + sub_file + '_task-motorimagery_eeg.mat'))
    
    # 提取数据
    task_data = mat['task_data'].reshape(-1, 62, 800)
    task_label = mat['task_label'].reshape(-1, 1)
    rest_data = mat['rest_data'].reshape(-1, 62, 800)

    # 预处理数据
    task_data = preprocess_eeg_data(task_data, channel_list, target_channel_list, max_scale)
    rest_data = preprocess_eeg_data(rest_data, channel_list, target_channel_list, max_scale)

    # 提取类别为1和2的前trial_pre个trial的数据
    task_data_pre_1 = task_data[task_label[:, 0] == 1][:trial_pre]
    task_data_pre_2 = task_data[task_label[:, 0] == 2][:trial_pre]
    task_label_pre_1 = task_label[task_label[:, 0] == 1][:trial_pre]
    task_label_pre_2 = task_label[task_label[:, 0] == 2][:trial_pre]
    
    # 构成数据集
    sub_train_feature_array = np.concatenate((task_data_pre_1, task_data_pre_2, rest_data[:trial_pre]), axis=0)
    sub_train_label_array = np.concatenate((task_label_pre_1, task_label_pre_2, np.zeros((trial_pre, 1))), axis=0)
    
    # 剩下的数据
    task_data_rest_1 = task_data[task_label[:, 0] == 1][trial_pre:]
    task_data_rest_2 = task_data[task_label[:, 0] == 2][trial_pre:]
    task_label_rest_1 = task_label[task_label[:, 0] == 1][trial_pre:]
    task_label_rest_2 = task_label[task_label[:, 0] == 2][trial_pre:]
    rest_data_rest = rest_data[trial_pre:]
    
    sub_train_feature_array_1 = np.concatenate((task_data_rest_1, task_data_rest_2, rest_data_rest), axis=0)
    sub_train_label_array_1 = np.concatenate((task_label_rest_1, task_label_rest_2, np.zeros((len(rest_data_rest), 1))), axis=0)
    
    # 使用滑窗增广数据
    sub_train_feature_array = sliding_window_modified(sub_train_feature_array, window_size, step_size)
    #sub_train_feature_array = sub_train_feature_array.reshape(-1, sub_train_feature_array.shape[1], sub_train_feature_array.shape[3])

    sub_train_label_array = np.repeat(sub_train_label_array, sub_train_feature_array.shape[0] // len(sub_train_label_array))
    #sub_val_feature_array = sliding_window(sub_val_feature_array, window_size, step_size)
    #sub_val_label_array = np.repeat(sub_val_label_array, sub_val_feature_array.shape[0] // len(sub_val_label_array))
    sub_train_feature_array_1 = sliding_window_modified(sub_train_feature_array_1, window_size, step_size)
    #sub_train_feature_array_1 = sub_train_feature_array_1.reshape(-1, sub_train_feature_array_1.shape[1], sub_train_feature_array_1.shape[3])

    sub_train_label_array_1 = np.repeat(sub_train_label_array_1, sub_train_feature_array_1.shape[0] // len(sub_train_label_array_1))
    
    # temporal normalization 
    if normalize:
        sub_train_feature_array = TemporalNormalization(sub_train_feature_array)
        sub_train_feature_array_1 = TemporalNormalization(sub_train_feature_array_1)

    # 重新排列数据
    # 获取每个类别的索引
    indices_0 = np.where(sub_train_label_array_1 == 0)[0]
    indices_1 = np.where(sub_train_label_array_1 == 1)[0]
    indices_2 = np.where(sub_train_label_array_1 == 2)[0]

    # 创建新的索引数组
    new_indices = []
    
    # 按照顺序每次添加batch_size_online个样本用于在线更新
    for i in range(len(sub_train_label_array_1) // (len(pattern) * batch_size_online)):
        for j in pattern:
            if j == 0 and len(indices_0) >= batch_size_online:
                new_indices.extend(indices_0[:batch_size_online])
                indices_0 = indices_0[batch_size_online:]
            elif j == 1 and len(indices_1) >= batch_size_online:
                new_indices.extend(indices_1[:batch_size_online])
                indices_1 = indices_1[batch_size_online:]
            elif j == 2 and len(indices_2) >= batch_size_online:
                new_indices.extend(indices_2[:batch_size_online])
                indices_2 = indices_2[batch_size_online:]
    
    # 使用新的索引数组重新排列 sub_train_feature_array_1 和 sub_train_label_array_1
    sub_train_feature_array_1 = sub_train_feature_array_1[new_indices]
    sub_train_label_array_1 = sub_train_label_array_1[new_indices]

    # 将原来的训练集以proportion的比例拆分为新的训练集和验证集
    unique_labels = np.unique(sub_train_label_array)
    sub_train_feature_list = []
    sub_train_label_list = []
    sub_val_feature_list = []
    sub_val_label_list = []
    for label in unique_labels:
        indices = np.where(sub_train_label_array == label)[0]
        split_index = int(proportion * len(indices))
        sub_train_feature_list.append(sub_train_feature_array[indices[:split_index]])
        sub_train_label_list.append(sub_train_label_array[indices[:split_index]])
        sub_val_feature_list.append(sub_train_feature_array[indices[split_index:]])
        sub_val_label_list.append(sub_train_label_array[indices[split_index:]])
    sub_train_feature_array = np.concatenate(sub_train_feature_list, axis=0)
    sub_train_label_array = np.concatenate(sub_train_label_list, axis=0)
    sub_val_feature_array = np.concatenate(sub_val_feature_list, axis=0)
    sub_val_label_array = np.concatenate(sub_val_label_list, axis=0)

    return sub_train_feature_array, sub_train_label_array.astype(int), sub_val_feature_array, sub_val_label_array.astype(int), \
        sub_train_feature_array_1, sub_train_label_array_1.astype(int)

def TemporalNormalization(sub_train_feature_array, axis=-1, copy=True):
    # using the temporal normalization method in paper:
    # Liu Y, Wu H, Wang J, et al. Non-stationary transformers: Exploring the stationarity in time series forecasting[J]. Advances in Neural Information Processing Systems, 2022, 35: 9881-9893.

    for i in range(sub_train_feature_array.shape[0]):
        _sub_train_feature_array = sub_train_feature_array[i,:,:].copy()
        # calculate the mean and std for the time slice
        _mean = np.mean(_sub_train_feature_array, axis=axis, keepdims=True)
        _std = np.std(_sub_train_feature_array, axis=axis, keepdims=True)
        # avoiding the std==0.0
        _std[_std==0] = 0.01
        # temporal normalization
        _sub_train_feature_array = (_sub_train_feature_array - _mean) / _std

        sub_train_feature_array[i, :,:] = _sub_train_feature_array
    
    return sub_train_feature_array

def WholeModelTrainingSampling(combined_feature_array, combined_label_array, unique_labels, update_wholeModel=8, batch_size_online=9, trial_pre=120, sampling_rate=3/4):
    # Split the updated training set into source and target sets
    sub_train_feature_update_source = []
    sub_train_label_update_source = []
    sub_train_feature_update_target = []
    sub_train_label_update_target = []
    focalloss_alpha = []  # preparing for the focalloss alpha
    if update_wholeModel != 8:
        _update_wholeModel = 8
    else:
        _update_wholeModel = update_wholeModel

    # for recording the whole sampled indices
    training_indices_all = []
    validation_indices_all = []

    # for each class, sample the data according to the sessions for training and validation set
    for label in unique_labels:
        indices = np.where(combined_label_array == label)[0]
        sampling_indices = []

        # sample 0.75 of data from caliberation set for training 
        caliberation_indices = indices[0:int(trial_pre)]
        caliberation_samples = np.random.choice(caliberation_indices, size=int(len(caliberation_indices)*sampling_rate), replace=False)
        sampling_indices.extend(caliberation_samples)

        # sample 0.75 from data of each session for training 
        session_indices = indices[int(trial_pre):]
        session_num = int(len(session_indices)/(batch_size_online*_update_wholeModel/2))  # in our experiment setting, only 4 trials of total 36 samples in one session
        for i in range(session_num):
            _samples_indices_session = np.random.choice(session_indices[int(i*(batch_size_online*_update_wholeModel/2)):int((i+1)*(batch_size_online*_update_wholeModel/2))],
                                                        size=int((batch_size_online*_update_wholeModel/2)*sampling_rate), replace=False)
            sampling_indices.extend(_samples_indices_session)
        
        # sample the last indices for validation set 
        validation_indices = list(set(indices) - set(sampling_indices))
        
        # form the validation (target) and training (source) set
        sub_train_feature_update_target.append(combined_feature_array[validation_indices])
        sub_train_label_update_target.append(combined_label_array[validation_indices])
        sub_train_feature_update_source.append(combined_feature_array[sampling_indices])
        sub_train_label_update_source.append(combined_label_array[sampling_indices])

        # record the indices
        training_indices_all.extend(sampling_indices)
        validation_indices_all.extend(validation_indices)

    sub_train_feature_update_source = np.concatenate(sub_train_feature_update_source, axis=0)
    sub_train_label_update_source = np.concatenate(sub_train_label_update_source, axis=0)
    sub_train_feature_update_target = np.concatenate(sub_train_feature_update_target, axis=0)
    sub_train_label_update_target = np.concatenate(sub_train_label_update_target, axis=0)
    
    return sub_train_feature_update_source, sub_train_label_update_source, sub_train_feature_update_target, sub_train_label_update_target


def WholeModelTrainingSampling_NewSession(combined_feature_array, combined_label_array, unique_labels, update_wholeModel=8, batch_size_online=9, trial_pre=120, sampling_rate=3/4, ExcludeLatestSession=True):
    # Split the updated training set into source and target sets
    sub_train_feature_update_source = []
    sub_train_label_update_source = []
    sub_train_feature_update_target = []
    sub_train_label_update_target = []
    focalloss_alpha = []  # preparing for the focalloss alpha
    if update_wholeModel != 8:
        _update_wholeModel = 8
    else:
        _update_wholeModel = update_wholeModel

    # for recording the whole sampled indices
    training_indices_all = []
    validation_indices_all = []

    # for each class, sample the data according to the sessions for training and validation set
    for label in unique_labels:
        indices = np.where(combined_label_array == label)[0]
        sampling_indices = []

        # sample 0.75 of data from caliberation set for training 
        caliberation_indices = indices[0:int(trial_pre)]
        caliberation_samples = np.random.choice(caliberation_indices, size=int(len(caliberation_indices)*sampling_rate), replace=False)
        sampling_indices.extend(caliberation_samples)

        # sample 0.75 from data of each session for training 
        if ExcludeLatestSession:
            session_indices = indices[int(trial_pre): -int(_update_wholeModel/2)*batch_size_online]  # we only select the old data sessions as the training sessions
        else:
            session_indices = indices[int(trial_pre):]
        
        session_num = int(len(session_indices)/(batch_size_online*_update_wholeModel/2))  # in our experiment setting, only 4 trials of total 36 samples in one session
        for i in range(session_num):
            _samples_indices_session = np.random.choice(session_indices[int(i*(batch_size_online*_update_wholeModel/2)):int((i+1)*(batch_size_online*_update_wholeModel/2))],
                                                        size=int((batch_size_online*_update_wholeModel/2)*sampling_rate), replace=False)
            sampling_indices.extend(_samples_indices_session)
        
        # sample the last indices for validation set 
        validation_indices = list(set(indices) - set(sampling_indices))
        
        # form the validation (target) and training (source) set
        sub_train_feature_update_target.append(combined_feature_array[validation_indices])
        sub_train_label_update_target.append(combined_label_array[validation_indices])
        sub_train_feature_update_source.append(combined_feature_array[sampling_indices])
        sub_train_label_update_source.append(combined_label_array[sampling_indices])

        # record the indices
        training_indices_all.extend(sampling_indices)
        validation_indices_all.extend(validation_indices)

    sub_train_feature_update_source = np.concatenate(sub_train_feature_update_source, axis=0)
    sub_train_label_update_source = np.concatenate(sub_train_label_update_source, axis=0)
    sub_train_feature_update_target = np.concatenate(sub_train_feature_update_target, axis=0)
    sub_train_label_update_target = np.concatenate(sub_train_label_update_target, axis=0)
    
    return sub_train_feature_update_source, sub_train_label_update_source, sub_train_feature_update_target, sub_train_label_update_target

        