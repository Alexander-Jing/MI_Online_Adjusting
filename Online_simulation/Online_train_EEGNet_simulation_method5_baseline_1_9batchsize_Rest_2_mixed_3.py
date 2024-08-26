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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import EEGNetFea, ConvEncoderResBN, ConvEncoder3ResBN, ConvEncoderCls, ConvEncoderClsFea, ResEncoderfinetune, ConvEncoder3_ClsFeaTL, ConvEncoder_OutputClsFeaTL, ConvEncoder_OutputClsHeavyFeaTL
from helpers.brain_data import Offline_read_csv, brain_dataset, Online_read_csv, Online_simulation_read_csv, Online_simulation_read_csv_windows_preprocess_normalization
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, \
    save_pickle, train_one_epoch, train_one_epoch_fea, train_update, eval_model, eval_model_fea, train_one_epoch_MMD, save_training_curves_FixedTrainValSplit, \
        write_performance_info_FixedTrainValSplit, write_program_time, eval_model_confusion_matrix_fea, train_one_epoch_MMDavg, write_inference_time
from helpers.utils import Offline_write_performance_info_FixedTrainValSplit, Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix, \
    eval_model_confusion_matrix, accuracy_iteration_plot,str2bool,accuracy_save2csv, train_one_epoch_MMD_Weights, compute_total_accuracy_per_class,\
    accuracy_perclass_save2csv, accuracy_perclass_iteration_plot, eval_model_fea_exemplars, eval_model_fea_exemplars_distillation, train_one_epoch_fea_distillation,\
        eval_model_fea_exemplars_distillation_label, eval_model_fea_exemplars_distillation_datafea_logitlabel, train_one_epoch_logit_distillation, train_one_epoch_label_distillation, train_one_epoch_logitlabel_distillation, eval_model_fea_loss
from Offline_synthesizing_results.synthesize_hypersearch_for_a_subject import synthesize_hypersearch_confusionMatrix
from Online_simulation_synthesizing.Online_simulation_synthesizing_subjects import Online_simulation_synthesizing_results, Online_simulation_synthesizing_results_comparison,\
      Online_simulation_synthesizing_results_linear, Online_simulation_synthesizing_results_comparison_linear, Online_simulation_synthesizing_results_linear_perclass, Online_simulation_synthesizing_results_2cls_linear,\
      Online_simulation_synthesizing_results_comparison_linear_2cls, Online_simulation_synthesizing_results_polynomial, Online_simulation_synthesizing_results_comparison_polynomial


#for personal model, save the test prediction of each cv fold
def Offline_EEGNet_simulation(args_dict):
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    sub_name = args_dict.sub_name
    Offline_folder_path = args_dict.Offline_folder_path
    trial_pre = args_dict.trial_pre
    Online_folder_path = args_dict.Online_folder_path
    windows_num = args_dict.windows_num
    proportion = args_dict.proportion
    Offline_result_save_rootdir = args_dict.Offline_result_save_rootdir
    Online_result_save_rootdir = args_dict.Online_result_save_rootdir
    restore_file = args_dict.restore_file
    n_epoch_offline = args_dict.n_epoch_offline
    batch_size = args_dict.batch_size
    unfreeze_encoder_offline = args_dict.unfreeze_encoder_offline
    unfreeze_encoder_online = args_dict.unfreeze_encoder_online
    
    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        #device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
    
    sub_train_feature_array, sub_train_label_array, sub_val_feature_array, sub_val_label_array, \
        sub_train_feature_array_1, sub_train_label_array_1 = Online_simulation_read_csv_windows_preprocess_normalization(folder_path=Offline_folder_path, sub_file=sub_name, trial_pre=40, proportion=proportion)
        
    #dataset object
    group_train_set = brain_dataset(sub_train_feature_array, sub_train_label_array)
    group_val_set = brain_dataset(sub_val_feature_array, sub_val_label_array)

    #dataloader object
    cv_train_batch_size = batch_size
    cv_val_batch_size = batch_size
    sub_cv_train_loader = torch.utils.data.DataLoader(group_train_set, batch_size=cv_train_batch_size, shuffle=True) 
    sub_cv_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False)
    print("data prepared")

    """
    The offline simulation part, using the data from MI2 as the simulation dataset 
    MI2: https://doi.org/10.7910/DVN/RBN3XG
    """
    #cross validation:
    lrs = [0.001, 0.01, 0.1]
    dropouts = [0.0, 0.25, 0.5, 0.75]

    start_time = time.time()
    
    for lr in lrs:
        for dropout in dropouts:
            experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
            print(experiment_name)
            #derived arg
            result_save_subjectdir = os.path.join(Offline_result_save_rootdir, sub_name, experiment_name)
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
            #encoder_to_use = ConvEncoder3ResBN(in_features=30, encoder_h=128, enc_width=((3,3),(3,3),(3,3)), enc_downsample=((1,1),(1,1),(1,1)), dropout=dropout)
            #encoder_to_use_output = ConvEncoder_OutputClsFeaTL(in_features=128, output_h=128, width=((3,3),), stride=((1,1),), num_features_for_classification=int(15*64), dropout=dropout)
            model = EEGNetFea(feature_size=30, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)


            # reload weights from restore_file is specified
            if restore_file != 'None':
                #restore_path = os.path.join(os.path.join(result_save_subject_checkpointdir, restore_file))
                restore_path = restore_file
                print('loading checkpoint: {}'.format(restore_path))

            model = model.to(device)

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
                if accuracy_per_class[1] > 0.33 and accuracy_per_class[2] > 0.33:
                    is_best = val_accuracy >= best_val_accuracy

                if is_best:
                    best_val_accuracy = val_accuracy

                    torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                    #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder.pt'))
                    #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))

                    result_save_dict['bestepoch_val_accuracy'] = val_accuracy
                    for cls_i in range(accuracy_per_class.shape[0]):
                        result_save_dict['class_accuracy_' + str(cls_i)] = accuracy_per_class[cls_i]

            #save training curve 
            save_training_curves_FixedTrainValSplit('training_curve.png', result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)

            #save the model at last epoch
            #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'last_model.statedict'))
            #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder.pt'))
            #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
            
            #save result_save_dict
            save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
            
            #write performance to txt file
            Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix(model.state_dict(), result_save_subject_resultanalysisdir, result_save_dict)
    
    end_time = time.time()
    total_time = end_time - start_time
    write_program_time(os.path.join(Offline_result_save_rootdir, sub_name), total_time)


def Online_updating_EEGNet_simulation(args_dict):
    """
    Online simulation part
    The online simulation part using the new weighting method to update the model
    """  
    #parse args:
    gpu_idx = args_dict.gpu_idx
    sub_name = args_dict.sub_name
    Offline_folder_path = args_dict.Offline_folder_path
    trial_pre = args_dict.trial_pre
    Online_folder_path = args_dict.Online_folder_path
    windows_num = args_dict.windows_num
    proportion = args_dict.proportion
    Offline_result_save_rootdir = args_dict.Offline_result_save_rootdir
    Online_result_save_rootdir = args_dict.Online_result_save_rootdir
    restore_file = args_dict.restore_file
    n_epoch_online = args_dict.n_epoch_online
    batch_size = args_dict.batch_size
    batch_size_online = args_dict.batch_size_online    # batch_size_online = 4
    trial_nums = args_dict.trial_nums    # trial_nums = 40
    unfreeze_encoder_offline = args_dict.unfreeze_encoder_offline
    unfreeze_encoder_online = args_dict.unfreeze_encoder_online
    accuracy_per_class_init = args_dict.accuracy_per_class_init
    update_trial = args_dict.update_trial
    alpha_distill = args_dict.alpha_distill
    update_wholeModel = args_dict.update_wholeModel

    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        #device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
    
    sub_train_feature_array, sub_train_label_array, sub_val_feature_array, sub_val_label_array, \
        sub_train_feature_array_1, sub_train_label_array_1 = Online_simulation_read_csv_windows_preprocess_normalization(folder_path=Offline_folder_path, sub_file=sub_name, trial_pre=50, \
                                                                                                proportion=proportion, batch_size_online=batch_size_online, \
                                                                                                    pattern=[1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 
                                                                                                               2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 
                                                                                                               2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 
                                                                                                               2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 
                                                                                                               1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 
                                                                                                               2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 
                                                                                                               1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 
                                                                                                               2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0])
    
     
    match = re.search(r"lr(\d+\.\d+)_dropout(\d+\.\d+)", restore_file)
    if match:
        lr = float(match.group(1))
        dropout = float(match.group(2))
        print(f"lr={lr}, dropout={dropout}")
    else:
        print("No match found.")
    
    #create model
    model = EEGNetFea(feature_size=30, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)

    experiment_name = 'lr{}_dropout{}'.format(0.0, dropout)#experiment name: used for indicating hyper setting
    print(experiment_name)

    #reload weights from restore_file is specified
    if restore_file != 'None':
        # move the best model from the offline experiments results
        Offline_path_encoder = os.path.join(Offline_result_save_rootdir, sub_name, restore_file, 'checkpoint', 'best_model.pt')  # using the name online_model.statedict for all the online manipulations
        
        makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name, experiment_name, 'checkpoint'))
        restore_path_encoder = os.path.join(Online_result_save_rootdir, sub_name, experiment_name, 'checkpoint', 'best_model.pt')  # using the name online_model.statedict for all the online manipulations

        # Copy Offline path to restore path
        shutil.copy(Offline_path_encoder, restore_path_encoder)
        print('Successfully copied {} to {}'.format(Offline_path_encoder, restore_path_encoder))

        # load the model
        model.load_state_dict(torch.load(restore_path_encoder))  
        print('loading checkpoint encoder: {}'.format(restore_path_encoder))
    
    # online simulation
    _combined_feature_array = np.concatenate((sub_train_feature_array, sub_val_feature_array), axis=0)
    _combined_label_array = np.concatenate((sub_train_label_array, sub_val_label_array), axis=0)
    # to avoid the situation that data out of offline training are used as the online training set earlier, we get the trial_pre num of the dataset
    unique_labels = np.unique(_combined_label_array)
    combined_feature_array = []
    combined_label_array = []
    for label in unique_labels:
        indices = np.where(_combined_label_array == label)[0]
        selected_indices = indices[:trial_pre]
        sub_feature = _combined_feature_array[selected_indices]
        sub_label = _combined_label_array[selected_indices]
        combined_feature_array.append(sub_feature)
        combined_label_array.append(sub_label)
    
    combined_feature_array = np.concatenate(combined_feature_array, axis=0)
    combined_label_array = np.concatenate(combined_label_array, axis=0)

    #derived arg
    result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name, experiment_name)
    result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
    result_save_subject_predictionsdir = os.path.join(result_save_subjectdir, 'predictions')
    result_save_subject_resultanalysisdir = os.path.join(result_save_subjectdir, 'result_analysis')
    result_save_subject_trainingcurvedir = os.path.join(result_save_subjectdir, 'trainingcurve')

    makedir_if_not_exist(result_save_subjectdir)
    makedir_if_not_exist(result_save_subject_checkpointdir)
    makedir_if_not_exist(result_save_subject_predictionsdir)
    makedir_if_not_exist(result_save_subject_resultanalysisdir)
    makedir_if_not_exist(result_save_subject_trainingcurvedir)

    predict_accuracies = []
    class_predictions_arrays = []
    labels_arrays = []

    accuracies_per_class = []
    accuracy_per_class_iters = []

    _n_epoch_online = n_epoch_online
    accuracies_per_class_iterations = []
    accuracies_per_class_iterations.append([0, 0])
    accuracies_per_class_iterations.append([1, 0])
    accuracies_per_class_iterations.append([2, 0])

    
    # this is the method in paper: 
    # Lin C Y, Lu C F, Jao C W, et al. Toward consistency between humans and classifiers: Improved performance of a real-time brain–computer interface using a mutual learning system[J]. Expert Systems with Applications, 2023, 226: 120205.
    # using the method for the simulation experiment

    for trial_idx in range(trial_nums):
        # generate the new data, simulating the online experiment 
        sub_train_feature_batches = sub_train_feature_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online, :, :]
        sub_train_label_batches = sub_train_label_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online]

        # combine the datasets with the new data
        combined_feature_array = np.concatenate((combined_feature_array, sub_train_feature_batches), axis=0)
        combined_label_array = np.concatenate((combined_label_array, sub_train_label_batches), axis=0)  
        
        # set the instance class for updating 
        train_label_now_ = np.unique(sub_train_label_batches)
        train_label_exemplars = train_label_now_%2 + 1  # if label is 1, generate label of 2, else if label is 2, generate label of 1
        
        # form the training set and the validation set
        unique_labels = np.unique(combined_label_array)
        sub_train_feature_update = []
        sub_train_label_update = []
        sub_train_feature_update_val = []
        sub_train_label_update_val = []
        for label in unique_labels:
            # form the validation set, in Lin's original work, they use the 4 latest EEG segments as the validation data for each category
            if label != 0:
                indices = np.where(combined_label_array == label)[0]
                selected_indices = indices[-4:]
            else:
                indices = np.where(combined_label_array == label)[0]
                #selected_indices = np.random.choice(indices, 4, replace=False)
                selected_indices = indices[-4:]

            sub_train_feature_update_val.append(combined_feature_array[selected_indices])
            sub_train_label_update_val.append(combined_label_array[selected_indices])
            # form the training set, which is the data in the last trial
            if label == train_label_now_:
                selected_indices = indices[-batch_size_online:-4]
                sub_train_feature_update.append(combined_feature_array[selected_indices])
                sub_train_label_update.append(combined_label_array[selected_indices])
        
        sub_train_feature_update = np.concatenate(sub_train_feature_update, axis=0)
        sub_train_label_update = np.concatenate(sub_train_label_update, axis=0)
        sub_train_feature_update_val = np.concatenate(sub_train_feature_update_val, axis=0)
        sub_train_label_update_val = np.concatenate(sub_train_label_update_val, axis=0)

        # form the online test set 
        _sub_updating_predict = brain_dataset(sub_train_feature_batches, sub_train_label_batches)
        sub_updating_predict = torch.utils.data.DataLoader(_sub_updating_predict, batch_size=sub_train_feature_batches.shape[0], shuffle=False)
        # form the online training set
        sub_online_train_set = brain_dataset(sub_train_feature_update, sub_train_label_update)
        sub_online_val_set = brain_dataset(sub_train_feature_update_val, sub_train_label_update_val)
        sub_online_train_loader = torch.utils.data.DataLoader(sub_online_train_set, batch_size=10, shuffle=True)
        sub_online_val_loader = torch.utils.data.DataLoader(sub_online_val_set, batch_size=10, shuffle=True)
        
        print("********** Online simulation trial: {} ***********".format(trial_idx))
        start_time_infer = time.time()
        # load the best model
        if (trial_idx+1) > update_trial:
            model.load_state_dict(torch.load(os.path.join(result_save_subject_checkpointdir, 'best_model.pt')))  

        model = model.to(device)
        # online testing of the MI class
        ground_truth_label = np.unique(sub_train_label_batches)
        print("ground truth label:{}".format(ground_truth_label))
        predict_accu, class_predictions_array, labels_array, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_updating_predict, device)
        # recording the corresponding accuracy of each class
        accuracies_per_class_iterations.append([ground_truth_label[0], predict_accu/100])
        predict_accuracies.append(predict_accu)
        #accuracies_per_class.append(accuracy_per_class)
        class_predictions_arrays.extend(class_predictions_array.tolist())
        labels_arrays.extend(labels_array.tolist())
        
        stop_time_infer = time.time()
        time_infer = stop_time_infer - start_time_infer
        write_inference_time(os.path.join(Online_result_save_rootdir, sub_name), time_infer)

        print("predict accuracy: {}".format(predict_accu))
        print("predict accuracy per class: {}".format(accuracy_per_class))

        if (trial_idx + 1) % update_trial == 0:           
            print("******* Updating the model trial: {} ************".format(trial_idx))
            start_time = time.time()
            experiment_name = 'lr{}_dropout{}'.format(0.0, dropout)#experiment name: used for indicating hyper setting
            # print(experiment_name)
            #derived arg
            result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name, experiment_name)
            result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')

            makedir_if_not_exist(result_save_subjectdir)
            makedir_if_not_exist(result_save_subject_checkpointdir)

            if (trial_idx + 1) % (update_trial) == 0: 
                accuracy_per_class_iter = compute_total_accuracy_per_class(accuracies_per_class_iterations)
                accuracy_per_class_iters.append(accuracy_per_class_iter)
                print(accuracy_per_class_iter)
            
            # the loss function
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            # Initialize learning rate bounds
            lower_bound_lr = 0
            upper_bound_lr = 0.01

            # Initialize best loss
            best_loss = float('inf')

            # Calculate the loss for the initial learning rate bounds
            for lr in [lower_bound_lr, upper_bound_lr]:
                optimizer.param_groups[0]['lr'] = lr

                # Training loop
                is_best = False
                best_val_accuracy = 0.0
                best_val_loss = 0.0
                epoch_train_loss = []
                epoch_train_accuracy = []
                epoch_validation_accuracy = []

                result_save_dict = dict()

                _n_epoch_online = 4

                for epoch in trange(_n_epoch_online, desc='online classification update'):
                    optimizer.param_groups[0]['lr'] = lr

                    average_loss_this_epoch = train_one_epoch_fea(model, optimizer, criterion, sub_online_train_loader, device)
                    val_accuracy, _, _ , _, loss_avg_val = eval_model_fea_loss(model, sub_online_val_loader, criterion, device)
                    train_accuracy, _, _ , _, loss_avg_train = eval_model_fea_loss(model, sub_online_train_loader, criterion, device)

                    epoch_train_loss.append(average_loss_this_epoch)
                    epoch_train_accuracy.append(train_accuracy)
                    epoch_validation_accuracy.append(val_accuracy)

                    #update is_best flag, only when the accuracies of two classes of motor imagery are larger than random choice
                    is_best = val_accuracy >= best_val_accuracy

                    if is_best:
                        print("best_val_accuracy: {}".format(val_accuracy))
                        best_val_accuracy = val_accuracy
                        best_val_loss = loss_avg_val
                        if best_val_loss <= best_loss:
                            torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                            #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder.pt'))
                            #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))
                            best_loss = best_val_loss
                        result_save_dict['bestepoch_val_accuracy'] = val_accuracy
            
            print("Initial best loss: {}".format(best_loss))

            # Use bisection method to find optimal learning rate
            # Referring to the bisection optimization method in convex optimization 
            # Referring to https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
            for iter_num in range(4):  # number of iterations in the bisection method is set 4 in Lin's method
                lr = (lower_bound_lr + upper_bound_lr) / 2

                # Training loop
                is_best = False
                best_val_accuracy = 0.0
                best_val_loss = 0.0
                epoch_train_loss = []
                epoch_train_accuracy = []
                epoch_validation_accuracy = []

                result_save_dict = dict()

                _n_epoch_online = 4

                for epoch in trange(_n_epoch_online, desc='online classification update'):
                    optimizer.param_groups[0]['lr'] = lr

                    average_loss_this_epoch = train_one_epoch_fea(model, optimizer, criterion, sub_online_train_loader, device)
                    val_accuracy, _, _ , _, loss_avg_val = eval_model_fea_loss(model, sub_online_val_loader, criterion, device)
                    train_accuracy, _, _ , _, loss_avg_train = eval_model_fea_loss(model, sub_online_train_loader, criterion, device)

                    epoch_train_loss.append(average_loss_this_epoch)
                    epoch_train_accuracy.append(train_accuracy)
                    epoch_validation_accuracy.append(val_accuracy)

                    #update is_best flag, only when the accuracies of two classes of motor imagery are larger than random choice
                    is_best = val_accuracy >= best_val_accuracy

                    if is_best:
                        print("best_val_accuracy: {}".format(val_accuracy))
                        best_val_accuracy = val_accuracy
                        best_val_loss = loss_avg_val
                        if best_val_loss <= best_loss:
                            torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                            #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder.pt'))
                            #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))

                        result_save_dict['bestepoch_val_accuracy'] = val_accuracy

                # If validation loss is high, decrease learning rate
                if best_val_loss > best_loss:  # adjust this threshold as needed
                    upper_bound_lr = lr
                else:
                    lower_bound_lr = lr
                    best_loss = best_val_loss
                print("Iteration{},  best loss: {}".format(iter_num, best_loss))

            #save training curve 
            save_training_curves_FixedTrainValSplit('training_curve.png', result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)

            #save the model at last epoch
            #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'last_model.statedict'))
            #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder.pt'))
            #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
            
            #save result_save_dict
            save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
            
            #write performance to txt file
            Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix(model.state_dict(), result_save_subject_resultanalysisdir, result_save_dict)
            end_time = time.time()
            total_time = end_time - start_time
            write_program_time(os.path.join(Online_result_save_rootdir, sub_name), total_time)
    
    accuracy_save2csv(predict_accuracies, result_save_subjectdir, filename='predict_accuracies.csv', columns=['Accuracy'])
    accuracy_save2csv(class_predictions_arrays, result_save_subjectdir, filename='class_predictions_arrays.csv', columns=['class_predictions_arrays'])
    accuracy_save2csv(labels_arrays, result_save_subjectdir, filename='labels_arrays.csv', columns=['labels_arrays'])

    accuracy_iteration_plot(predict_accuracies, result_save_subjectdir)
    accuracy_perclass_save2csv(accuracy_per_class_iters, result_save_subjectdir)
    accuracy_perclass_iteration_plot(accuracy_per_class_iters, result_save_subjectdir)



    


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
    parser.add_argument('--n_epoch_offline', default=100, type=int, help="number of epoch")
    parser.add_argument('--n_epoch_online', default=100, type=int, help="number of epoch")
    parser.add_argument('--batch_size', default=64, type=int, help="number of batch size")
    parser.add_argument('--Online_folder_path', default='./Online_DataCollected', help="Online folder to the dataset")
    parser.add_argument('--Online_result_save_rootdir', default='./Online_experiments', help="Directory containing the experiment models")
    parser.add_argument('--batch_size_online', default=4, type=int, help="number of batch size for online updating")
    parser.add_argument('--trial_pre', default=120, type=int, help="number of samples each class for offline training")
    parser.add_argument('--trial_nums', default=40, type=int, help="number of trails of samples for online updating")
    parser.add_argument('--update_trial', default=15, type=int, help="number of trails for instant updating")
    parser.add_argument('--update_wholeModel', default=15, type=int, help="number of trails for longer updating")
    parser.add_argument('--alpha_distill', default=0.5, type=float, help="alpha of the distillation and cls loss func")
    parser.add_argument('--best_validation_path', default='lr0.001_dropout0.5', type=str, help="path of the best validation performance model")
    parser.add_argument('--unfreeze_encoder_offline', default=False, type=str2bool, help="whether to unfreeze the encoder params during offline training process")
    parser.add_argument('--unfreeze_encoder_online', default=False, type=str2bool, help="whether to unfreeze the encoder params during online training process")

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
    trial_pre = args.trial_pre
    trial_nums = args.trial_nums
    best_validation_path = args.best_validation_path
    unfreeze_encoder_offline = args.unfreeze_encoder_offline
    unfreeze_encoder_online = args.unfreeze_encoder_online
    update_trial = args.update_trial
    alpha_distill = args.alpha_distill
    update_wholeModel = args.update_wholeModel

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
    print('n_epoch_offline: {} type: {}'.format(n_epoch_offline, type(n_epoch_offline)))
    print('batch size: {} type: {}'.format(batch_size, type(batch_size)))
   
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.sub_name = sub_name
    args_dict.Offline_folder_path = Offline_folder_path
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
    args_dict.trial_pre = trial_pre
    args_dict.trial_nums = trial_nums
    args_dict.best_validation_path = best_validation_path
    args_dict.unfreeze_encoder_offline = unfreeze_encoder_offline
    args_dict.unfreeze_encoder_online = unfreeze_encoder_online
    args_dict.accuracy_per_class_init = []
    args_dict.update_trial = update_trial
    args_dict.alpha_distill = alpha_distill
    args_dict.update_wholeModel = update_wholeModel

    seed_everything(seed)
    if mode == 'offline':
        # 离线训练模型
        Offline_EEGNet_simulation(args_dict)
        # 搜索离线训练模型超参数
        experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
        summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
        if not os.path.exists(summary_save_dir):
            os.makedirs(summary_save_dir)    
        best_validation_class_accuracy, best_validation_path = \
            synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
    
    if mode == 'online':
        # 搜索离线训练模型超参数
        experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
        summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
        if not os.path.exists(summary_save_dir):
            os.makedirs(summary_save_dir)    
        best_validation_class_accuracy, best_validation_path = \
            synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
        
        args_dict.accuracy_per_class_init = best_validation_class_accuracy
        args_dict.restore_file = best_validation_path
        Online_updating_EEGNet_simulation(args_dict)
    
    if mode == 'hybrid':
        # 离线训练模型
        Offline_EEGNet_simulation(args_dict)
        # 搜索离线训练模型超参数
        experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name)
        summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
        if not os.path.exists(summary_save_dir):
            os.makedirs(summary_save_dir)    
        best_validation_class_accuracy, best_validation_path = \
            synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
        
        args_dict.accuracy_per_class_init = best_validation_class_accuracy
        args_dict.restore_file = best_validation_path
        Online_updating_EEGNet_simulation(args_dict)

    if mode == 'synthesizing':
        Online_simulation_synthesizing_results_linear(Online_result_save_rootdir=Online_result_save_rootdir)
        Online_simulation_synthesizing_results_polynomial(Online_result_save_rootdir=Online_result_save_rootdir)
        #Online_simulation_synthesizing_results_2cls_linear(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_linear_perclass(Online_result_save_rootdir)

    if mode == 'comparison':
        methods = ['baseline1_encoder3_noupdate_noRest_val_6', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_11', 'method23_encoder3_val_classval_pretrainlight_unfreeze_inctremental_noRest_fixedepoch_val_6','method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1']  #'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1'
        Online_simulation_synthesizing_results_comparison_linear(Online_result_save_rootdir, methods)
        Online_simulation_synthesizing_results_comparison_polynomial(Online_result_save_rootdir, methods)
        #Online_simulation_synthesizing_results_comparison_linear_2cls(Online_result_save_rootdir, methods)