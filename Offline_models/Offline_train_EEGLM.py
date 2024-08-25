import os
import sys
import numpy as np
import torch
import torch.nn as nn

import time
import argparse
import re
import shutil

from easydict import EasyDict as edict
from tqdm import trange

from helpers.models import EEGNetTest, ResEncoderfinetune, ConvEncoderResBN, ConvEncoderCls, ConvEncoderClsFea, ConvEncoder3_ClsFeaTL
from helpers.brain_data import Offline_read_csv, brain_dataset, preprocess_eeg_data
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, train_one_epoch_fea, eval_model_fea, \
    eval_model, eval_model_confusion_matrix, eval_model_confusion_matrix_fea, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time
from helpers.utils import Offline_write_performance_info_FixedTrainValSplit, Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix
from helpers.utils import str2bool, save_best_validation_class_accuracy_offline
from Offline_synthesizing_results.synthesize_hypersearch_for_a_subject import synthesize_hypersearch, synthesize_hypersearch_confusionMatrix

#for personal model, save the test prediction of each cv fold
def Offline_train_classifierLM(args_dict):
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    sub_name_offline = args_dict.sub_name_offline
    sub_name_online = args_dict.sub_name_online
    Offline_folder_path = args_dict.Offline_folder_path
    Online_folder_path = args_dict.Online_folder_path
    windows_num = args_dict.windows_num
    proportion = args_dict.proportion
    result_save_rootdir = args_dict.Offline_result_save_rootdir
    Online_result_save_rootdir = args_dict.Online_result_save_rootdir
    restore_file = args_dict.restore_file
    n_epoch_offline = args_dict.n_epoch_offline
    batch_size = args_dict.batch_size
    unfreeze_encoder_offline = args_dict.unfreeze_encoder_offline
    use_pretrain = args_dict.use_pretrain
    
    
    sub_train_feature_array, sub_train_label_array, sub_val_feature_array, sub_val_label_array = Offline_read_csv(Offline_folder_path, windows_num, proportion)
    

    #dataset object
    group_train_set = brain_dataset(sub_train_feature_array, sub_train_label_array)
    group_val_set = brain_dataset(sub_val_feature_array, sub_val_label_array)

    #dataloader object
    cv_train_batch_size = batch_size
    cv_val_batch_size = batch_size
    sub_cv_train_loader = torch.utils.data.DataLoader(group_train_set, batch_size=cv_train_batch_size, shuffle=True) 
    sub_cv_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False)
    print("data prepared")
    
    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        #device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
        

    #cross validation:
    lrs = [0.001, 0.01, 0.1]
    dropouts = [0.0, 0.25, 0.5, 0.75]

    start_time = time.time()
    
    for lr in lrs:
        for dropout in dropouts:
            experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
            print(experiment_name)
            #derived arg
            result_save_subjectdir = os.path.join(result_save_rootdir, sub_name_offline, experiment_name)
            result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
            result_save_subject_predictionsdir = os.path.join(result_save_subjectdir, 'predictions')
            result_save_subject_resultanalysisdir = os.path.join(result_save_subjectdir, 'result_analysis')
            result_save_subject_trainingcurvedir = os.path.join(result_save_subjectdir, 'trainingcurve')

            makedir_if_not_exist(result_save_subjectdir)
            makedir_if_not_exist(result_save_subject_checkpointdir)
            makedir_if_not_exist(result_save_subject_predictionsdir)
            makedir_if_not_exist(result_save_subject_resultanalysisdir)
            makedir_if_not_exist(result_save_subject_trainingcurvedir)
            
            result_save_dict = dict()
            
            #create model
            if use_pretrain:
                encoder_to_use = ConvEncoderResBN(in_features=32, encoder_h=512)
                encoder_to_use_output = ConvEncoderClsFea(output_h=128, dropout=dropout)
            else:
                encoder_to_use_output = ConvEncoder3_ClsFeaTL(in_features=32, output_h=128, dropout=dropout)

            # reload weights from restore_file is specified
            if restore_file != 'None' and use_pretrain:
                #restore_path = os.path.join(os.path.join(result_save_subject_checkpointdir, restore_file))
                restore_path = restore_file
                print('loading checkpoint: {}'.format(restore_path))
                encoder_to_use.load(filename=restore_path)  # We load and freeze the model parameters
                encoder_to_use.freeze_features(unfreeze=unfreeze_encoder_offline)
            
            print("use pretrain models: {}".format(use_pretrain))
            if use_pretrain:
                model_to_use = ResEncoderfinetune(encoder=encoder_to_use, encoder_output=encoder_to_use_output).double()
            else:
                model_to_use = encoder_to_use_output.double()

            model = model_to_use.to(device)

            #create criterion and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) #the authors used Adam instead of SGD
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
            #training loop
            best_val_accuracy = 0.0
            is_best = False

            epoch_train_loss = []
            epoch_train_accuracy = []
            epoch_validation_accuracy = []

            for epoch in trange(n_epoch_offline, desc='1-fold cross validation'):
                average_loss_this_epoch = train_one_epoch_fea(model, optimizer, criterion, sub_cv_train_loader, device)
                val_accuracy, _, _, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_cv_val_loader, device)
                train_accuracy, _, _ , _ = eval_model_fea(model, sub_cv_train_loader, device)

                epoch_train_loss.append(average_loss_this_epoch)
                epoch_train_accuracy.append(train_accuracy)
                epoch_validation_accuracy.append(val_accuracy)

                #update is_best flag, only when the accuracies of two classes of motor imagery are larger than random choice
                if accuracy_per_class[1] > 0.33 or accuracy_per_class[2] > 0.33:
                    is_best = val_accuracy >= best_val_accuracy

                if is_best:
                    best_val_accuracy = val_accuracy

                    #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.statedict'))
                    if use_pretrain:
                        encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder.pt'))
                        encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))
                    else:
                        encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))
                    
                    result_save_dict['bestepoch_val_accuracy'] = val_accuracy
                    for cls_i in range(accuracy_per_class.shape[0]):
                        result_save_dict['class_accuracy_' + str(cls_i)] = accuracy_per_class[cls_i]

            #save training curve 
            save_training_curves_FixedTrainValSplit('training_curve.png', result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)

            #save the model at last epoch
            #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'last_model.statedict'))
            if use_pretrain:
                encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder.pt'))
                encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
            else:
                encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
            
            #save result_save_dict
            save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
            
            #write performance to txt file
            Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix(model.state_dict(), result_save_subject_resultanalysisdir, result_save_dict)
    
    end_time = time.time()
    total_time = end_time - start_time
    write_program_time(os.path.join(result_save_rootdir, sub_name_offline), total_time)    

"""
if __name__ == "__main__":
    
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--gpu_idx', default=0, type=int, help="gpu idx")
    parser.add_argument('--sub_name', default='Jyt', type=str, help='name of the subject')
    parser.add_argument('--windows_num', default=149, type=int, help='number of windows')
    
    parser.add_argument('--Offline_folder_path', default='./Offline_DataCollected', help="Offline folder to the dataset")
    parser.add_argument('--Offline_result_save_rootdir', default='./Offline_experiments', help="Directory containing the experiment models")
    parser.add_argument('--restore_file', default='None', help="xxx.statedict")
    parser.add_argument('--proportion', default=0.8, type=float, help='proportion of the training set of the whole dataset')
    parser.add_argument('--n_epoch', default=100, type=int, help="number of epoch")
    parser.add_argument('--n_epoch_offline', default=100, type=int, help="number of epoch")
    parser.add_argument('--n_epoch_online', default=100, type=int, help="number of epoch")
    parser.add_argument('--batch_size', default=64, type=int, help="number of batch size")
    parser.add_argument('--Online_folder_path', default='./Online_DataCollected', help="Online folder to the dataset")
    parser.add_argument('--Online_result_save_rootdir', default='./Online_experiments', help="Directory containing the experiment models")
    parser.add_argument('--batch_size_online', default=4, type=int, help="number of batch size for online updating")
    parser.add_argument('--best_validation_path', default='lr0.001_dropout0.5', type=str, help="path of the best validation performance model")
    parser.add_argument('--unfreeze_encoder_offline', default=False, type=str2bool, help="whether to unfreeze the encoder params during offline training process")
    parser.add_argument('--unfreeze_encoder_online', default=False, type=str2bool, help="whether to unfreeze the encoder params during online training process")
    parser.add_argument('--use_pretrain', default=False, type=str2bool, help="whether to use the pretrain models")

    parser.add_argument('--ip', default='172.18.22.21', type=str, help='the IP address')
    parser.add_argument('--port', default=8888, type=int, help='the port')
    parser.add_argument('--mode', default='Online', type=str, help='choice of working mode: Offline or Online')
    args = parser.parse_args()
    
    seed = args.seed
    gpu_idx = args.gpu_idx
    sub_name = args.sub_name
    Offline_folder_path  = args.Offline_folder_path
    Online_folder_path = args.Online_folder_path
    windows_num = args.windows_num
    proportion = args.proportion
    Offline_result_save_rootdir = args.Offline_result_save_rootdir
    Online_result_save_rootdir = args.Online_result_save_rootdir
    restore_file = args.restore_file
    n_epoch_offline = args.n_epoch_offline
    n_epoch_online = args.n_epoch_online
    batch_size = args.batch_size
    ip = args.ip
    port = args.port
    mode = args.mode
    batch_size_online = args.batch_size_online
    best_validation_path = args.best_validation_path
    unfreeze_encoder_offline = args.unfreeze_encoder_offline
    unfreeze_encoder_online = args.unfreeze_encoder_online
    use_pretrain = args.use_pretrain
    
    # save_folder = './Online_DataCollected' + str(sub_name)
    #sanity check:
    print('gpu_idx: {}, type: {}'.format(gpu_idx, type(gpu_idx)))
    print('sub_name: {}, type: {}'.format(sub_name, type(sub_name)))
    print('Offline_folder_path: {}, type: {}'.format(os.path.join(Offline_folder_path, sub_name), type(Offline_folder_path)))
    print('Online_folder_path: {}, type: {}'.format(os.path.join(Online_folder_path, sub_name), type(Offline_folder_path)))
    print('windows_num: {}, type: {}'.format(windows_num, type(windows_num)))
    print('proportion: {}, type: {}'.format(proportion, type(proportion)))
    print('Offline_result_save_rootdir: {}, type: {}'.format(Offline_result_save_rootdir, type(Offline_result_save_rootdir)))
    print('restore_file: {} type: {}'.format(restore_file, type(restore_file)))
    print('n_epoch: {} type: {}'.format(n_epoch_offline, type(n_epoch_offline)))
    print('batch size: {} type: {}'.format(batch_size, type(batch_size)))
   
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.sub_name = sub_name
    args_dict.Offline_folder_path = os.path.join(Offline_folder_path, sub_name)
    args_dict.Online_folder_path = os.path.join(Online_folder_path, sub_name)   
    args_dict.windows_num = windows_num
    args_dict.proportion = proportion
    args_dict.Offline_result_save_rootdir = Offline_result_save_rootdir
    args_dict.Online_result_save_rootdir = Online_result_save_rootdir
    args_dict.restore_file = restore_file
    args_dict.n_epoch_offline = n_epoch_offline
    args_dict.n_epoch_online = n_epoch_online
    args_dict.batch_size = batch_size
    args_dict.ip = ip
    args_dict.port = port
    args_dict.mode = mode
    args_dict.batch_size_online = batch_size_online
    args_dict.best_validation_path = best_validation_path
    args_dict.unfreeze_encoder_offline = unfreeze_encoder_offline
    args_dict.unfreeze_encoder_online = unfreeze_encoder_online
    args_dict.use_pretrain = use_pretrain
    
    seed_everything(seed)

    # 离线训练模型
    Offline_train_classifierLM(args_dict)
    # 搜索离线训练模型超参数
    experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
    summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
    if not os.path.exists(summary_save_dir):
        os.makedirs(summary_save_dir)    
    best_validation_class_accuracy, best_validation_path = \
            synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
    # 存储best_validation_class_accuracy到summary_save_dir文件夹下面best_validation_class_accuracy.csv
    save_best_validation_class_accuracy_offline(best_validation_class_accuracy, summary_save_dir)
"""