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
import torch.optim
from tqdm import trange
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import EEGNetFea, ConvEncoderResBN, ConvEncoder3ResBN, ConvEncoderCls, ConvEncoderClsFea, ResEncoderfinetune, ConvEncoder3_ClsFeaTL, ConvEncoder_OutputClsFeaTL, ConvEncoder_OutputClsHeavyFeaTL
from helpers.brain_data import Offline_read_csv, brain_dataset, Online_read_csv, Online_simulation_read_csv, Online_simulation_read_csv_windows_preprocess_normalization
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, \
    save_pickle, train_one_epoch, train_one_epoch_fea, train_update, eval_model, eval_model_fea, train_one_epoch_MMD, save_training_curves_FixedTrainValSplit, \
        write_performance_info_FixedTrainValSplit, write_program_time, eval_model_confusion_matrix_fea, train_one_epoch_MMDavg, write_inference_time
from helpers.utils import Offline_write_performance_info_FixedTrainValSplit, Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix, \
    eval_model_confusion_matrix, accuracy_iteration_plot,str2bool,accuracy_save2csv, train_one_epoch_MMD_Weights, compute_total_accuracy_per_class,\
    accuracy_perclass_save2csv, accuracy_perclass_iteration_plot, eval_model_fea_exemplars, eval_model_fea_exemplars_distillation, train_one_epoch_fea_distillation, train_one_epoch_fealogitlabel_distillation, \
        eval_model_fea_exemplars_distillation_label, eval_model_fea_exemplars_distillation_datafea_logitlabel, train_one_epoch_logit_distillation, train_one_epoch_label_distillation, train_one_epoch_logitlabel_distillation, train_one_epoch_fea_MMDContrastive, train_one_epoch_fea_MMDContrastive_targetcls_iter, \
            MultiClassFocalLoss, MultiClassNpFocalLoss, PolyLoss, write_exemplar_time, eval_model_fea_classPrototypes
from Offline_synthesizing_results.synthesize_hypersearch_for_a_subject import synthesize_hypersearch_confusionMatrix
from Online_simulation_synthesizing.Online_simulation_synthesizing_subjects import Online_simulation_synthesizing_results, Online_simulation_synthesizing_results_comparison,\
      Online_simulation_synthesizing_results_linear, Online_simulation_synthesizing_results_comparison_linear, Online_simulation_synthesizing_results_linear_perclass, Online_simulation_synthesizing_results_2cls_linear,\
      Online_simulation_synthesizing_results_comparison_linear_2cls, Online_simulation_synthesizing_results_polynomial, Online_simulation_synthesizing_results_comparison_polynomial, Online_simulation_synthesizing_results_comparison_polynomial_optimized, Online_simulation_synthesizing_results_polynomial_avg, Online_simulation_synthesizing_results_polynomial_avgF1,Online_simulation_synthesizing_results_comparison_polynomial_optimized_perclass


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
        sub_train_feature_array_1, sub_train_label_array_1 = Online_simulation_read_csv_windows_preprocess_normalization(folder_path=Offline_folder_path, sub_file=sub_name, trial_pre=40, preprocess=False, proportion=proportion)
        
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
            model = EEGNetFea(feature_size=29, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)
            
            # reload weights from restore_file is specified
            if restore_file != 'None':
                #restore_path = os.path.join(os.path.join(result_save_subject_checkpointdir, restore_file))
                restore_path = restore_file
                print('loading checkpoint: {}'.format(restore_path))
            
            model = model.to(device)

            #create criterion and optimizer
            criterion = nn.CrossEntropyLoss()
            #criterion = MultiClassFocalLoss(device=device, alpha=[0.33,0.33,0.33])
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
    para_m = args_dict.para_m
    cons_rate = args_dict.cons_rate

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
        sub_train_feature_array_1, sub_train_label_array_1 = Online_simulation_read_csv_windows_preprocess_normalization(folder_path=Offline_folder_path, sub_file=sub_name, trial_pre=50, preprocess=False,\
                                                                                                proportion=proportion, batch_size_online=batch_size_online, \
                                                                                                    pattern=  [1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 
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
    model = EEGNetFea(feature_size=29, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)
            
    #reload weights from restore_file is specified
    if restore_file != 'None':
        # move the best model from the offline experiments results
        Offline_path_encoder = os.path.join(Offline_result_save_rootdir, sub_name, restore_file, 'checkpoint', 'best_model.pt')  # using the name online_model.statedict for all the online manipulations
        
        makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name, restore_file, 'checkpoint'))
        restore_path_encoder = os.path.join(Online_result_save_rootdir, sub_name, restore_file, 'checkpoint', 'best_model.pt')  # using the name online_model.statedict for all the online manipulations

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

    experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
    print(experiment_name)
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

    # accuracies_per_class = []
    accuracy_per_class_iters = []

    _n_epoch_online = n_epoch_online
    accuracies_per_class_iterations = []
    accuracies_per_class_iterations.append([0, 0])
    accuracies_per_class_iterations.append([1, 0])
    accuracies_per_class_iterations.append([2, 0])
    accuracies_per_class_iterations_Rest = []
    accuracies_per_class_iterations_Rest.append([0, accuracy_per_class_init[0]])  # saving the existing initial caliberation accuracy for class 0
    
    #best_val_accuracy = 40.0
    #best_train_accuracy = 40.0

    for trial_idx in range(trial_nums):
        # generate the new data, simulating the online experiment 
        sub_train_feature_batches = sub_train_feature_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online, :, :]
        sub_train_label_batches = sub_train_label_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online]

        # when the model begin to update in the new task, try to save the output of the old model for the old tasks so that the distillation method can be used 
        if (trial_idx + 1) % update_wholeModel == 1:
            print("********** Online mean-of-exemplars generation trial: {} ***********".format(trial_idx))
            start_time_exemplar = time.time()
            # load the model
            if trial_idx > 0: 
                model.load_state_dict(torch.load(os.path.join(result_save_subject_checkpointdir, 'best_model.pt')))  

            model = model.to(device)
            
            unique_labels = np.unique(combined_label_array)
            # generate the exemplar of each class, including the 0(rest), 1,2(MI)
            for label in unique_labels:
                # for old classes, generating the exemplars class
                if label == 0.0:
                    indices = np.where(combined_label_array == label)[0]
                    selected_indices_exemplars = indices
                    sub_train_feature_exemplars = combined_feature_array[selected_indices_exemplars]
                    sub_train_label_exemplars = combined_label_array[selected_indices_exemplars]
                    _sub_exemplars = brain_dataset(sub_train_feature_exemplars, sub_train_label_exemplars)
                    sub_exemplars = torch.utils.data.DataLoader(_sub_exemplars, batch_size=sub_train_feature_exemplars.shape[0], shuffle=False)
                    Rest_output_data_exemplars, Rest_output_feas_exemplars, Rest_output_logits_exemplars, Rest_output_label_exemplars = eval_model_fea_exemplars_distillation_datafea_logitlabel(model, sub_exemplars, device, trial_pre)

                if label == 1.0:
                    indices = np.where(combined_label_array == label)[0]
                    selected_indices_exemplars = indices
                    sub_train_feature_exemplars = combined_feature_array[selected_indices_exemplars]
                    sub_train_label_exemplars = combined_label_array[selected_indices_exemplars]
                    _sub_exemplars = brain_dataset(sub_train_feature_exemplars, sub_train_label_exemplars)
                    sub_exemplars = torch.utils.data.DataLoader(_sub_exemplars, batch_size=sub_train_feature_exemplars.shape[0], shuffle=False)
                    MI1_output_data_exemplars, MI1_output_feas_exemplars, MI1_output_logits_exemplars, MI1_output_label_exemplars = eval_model_fea_exemplars_distillation_datafea_logitlabel(model, sub_exemplars, device, trial_pre)
                     
                if label == 2.0:
                    indices = np.where(combined_label_array == label)[0]
                    selected_indices_exemplars = indices
                    sub_train_feature_exemplars = combined_feature_array[selected_indices_exemplars]
                    sub_train_label_exemplars = combined_label_array[selected_indices_exemplars]
                    _sub_exemplars = brain_dataset(sub_train_feature_exemplars, sub_train_label_exemplars)
                    sub_exemplars = torch.utils.data.DataLoader(_sub_exemplars, batch_size=sub_train_feature_exemplars.shape[0], shuffle=False)
                    MI2_output_data_exemplars, MI2_output_feas_exemplars, MI2_output_logits_exemplars, MI2_output_label_exemplars = eval_model_fea_exemplars_distillation_datafea_logitlabel(model, sub_exemplars, device, trial_pre)
            end_time_exemplar = time.time()
            total_time_exemplar = end_time_exemplar - start_time_exemplar
            write_exemplar_time(os.path.join(Online_result_save_rootdir, sub_name), total_time_exemplar)        
        # set the instance class for updating 
        train_label_now_ = np.unique(sub_train_label_batches)
        train_label_exemplars = train_label_now_%2 + 1  # if label is 1, generate label of 2, else if label is 2, generate label of 1
            
        old_data_exmeplars = []
        old_feas_exemplars = []
        old_logits_exemplars = []
        old_labels_exemplars = []
        new_feas_exemplars = []
        new_labels_exemplars = []

        # form the exemplar sets
        if train_label_now_ == 0.0:
            old_data_exmeplars.append(MI1_output_data_exemplars)
            old_data_exmeplars.append(MI2_output_data_exemplars)
            old_feas_exemplars.append(MI1_output_feas_exemplars)
            old_feas_exemplars.append(MI2_output_feas_exemplars)
            old_logits_exemplars.append(MI1_output_logits_exemplars)
            old_logits_exemplars.append(MI2_output_logits_exemplars)
            old_labels_exemplars.append(MI1_output_label_exemplars)
            old_labels_exemplars.append(MI2_output_label_exemplars)
            new_feas_exemplars.append(Rest_output_feas_exemplars)
            new_labels_exemplars.append(Rest_output_label_exemplars)
        
        if train_label_now_ == 1.0:
            old_data_exmeplars.append(Rest_output_data_exemplars)
            old_data_exmeplars.append(MI2_output_data_exemplars)
            old_feas_exemplars.append(Rest_output_feas_exemplars)
            old_feas_exemplars.append(MI2_output_feas_exemplars)
            old_logits_exemplars.append(Rest_output_logits_exemplars)
            old_logits_exemplars.append(MI2_output_logits_exemplars)
            old_labels_exemplars.append(Rest_output_label_exemplars)
            old_labels_exemplars.append(MI2_output_label_exemplars)
            new_feas_exemplars.append(MI1_output_feas_exemplars)
            new_labels_exemplars.append(MI1_output_label_exemplars)
        
        if train_label_now_ == 2.0:
            old_data_exmeplars.append(Rest_output_data_exemplars)
            old_data_exmeplars.append(MI1_output_data_exemplars)
            old_feas_exemplars.append(Rest_output_feas_exemplars)
            old_feas_exemplars.append(MI1_output_feas_exemplars)
            old_logits_exemplars.append(Rest_output_logits_exemplars)
            old_logits_exemplars.append(MI1_output_logits_exemplars)
            old_labels_exemplars.append(Rest_output_label_exemplars)
            old_labels_exemplars.append(MI1_output_label_exemplars)
            new_feas_exemplars.append(MI2_output_feas_exemplars)
            new_labels_exemplars.append(MI2_output_label_exemplars)

            
        # generate the old data loader
        sub_oldclass_data_distill = np.concatenate(old_data_exmeplars, axis=0)        
        sub_oldclass_fea_distill = np.concatenate(old_feas_exemplars, axis=0)
        sub_oldclass_logits_distill = np.concatenate(old_logits_exemplars, axis=0)
        sub_oldclass_labels = np.concatenate(old_labels_exemplars, axis=0)
        sub_oldclass_datafea_distill = brain_dataset(sub_oldclass_data_distill, sub_oldclass_fea_distill)
        sub_oldclass_datalogits_distill = brain_dataset(sub_oldclass_data_distill, sub_oldclass_logits_distill)
        sub_oldclass_datalabels_distill = brain_dataset(sub_oldclass_data_distill, sub_oldclass_labels)
        sub_oldclass_datafea_distill_loader = torch.utils.data.DataLoader(sub_oldclass_datafea_distill, batch_size=batch_size, shuffle=True)
        sub_oldclass_datalogits_distill_loader = torch.utils.data.DataLoader(sub_oldclass_datalogits_distill, batch_size=batch_size, shuffle=True)
        sub_oldclass_datalabels_distill_loader = torch.utils.data.DataLoader(sub_oldclass_datalabels_distill, batch_size=batch_size, shuffle=True)
        
        # generate the new data loader
        sub_newclass_fea_distill = np.concatenate(new_feas_exemplars, axis=0)
        sub_newclass_labels_distill = np.concatenate(new_labels_exemplars, axis=0)
        sub_newclass_fealabel_distill = brain_dataset(sub_newclass_fea_distill, sub_newclass_labels_distill)
        sub_newclass_fealabel_distill_loader = torch.utils.data.DataLoader(sub_newclass_fealabel_distill, batch_size=batch_size, shuffle=True)

        # combine the datasets with the new data
        combined_feature_array = np.concatenate((combined_feature_array, sub_train_feature_batches), axis=0)
        combined_label_array = np.concatenate((combined_label_array, sub_train_label_batches), axis=0)  
        
        # if time for updating, updating the whole model
        if (trial_idx + 1) % update_wholeModel == 0:
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

            for label in unique_labels:
                indices = np.where(combined_label_array == label)[0]
                
                if label != 0:
                    #print("MI label: ", label)
                    target_indices = indices[-int(_update_wholeModel/2)*batch_size_online:]
                else:
                    #print("Rest label: ", label)
                    # choose int(_update_wholeModel/2)*batch_size_online samples from the newest data
                    target_indices = indices[-int(_update_wholeModel/2)*batch_size_online:]
                    
                source_indices = list(set(indices) - set(target_indices))
                
                sub_train_feature_update_target.append(combined_feature_array[target_indices])
                sub_train_label_update_target.append(combined_label_array[target_indices])
                sub_train_feature_update_source.append(combined_feature_array[source_indices])
                sub_train_label_update_source.append(combined_label_array[source_indices])
            sub_train_feature_update_source = np.concatenate(sub_train_feature_update_source, axis=0)
            sub_train_label_update_source = np.concatenate(sub_train_label_update_source, axis=0)
            sub_train_feature_update_target = np.concatenate(sub_train_feature_update_target, axis=0)
            sub_train_label_update_target = np.concatenate(sub_train_label_update_target, axis=0)
            
            # Form the new training sets
            source_train_set = brain_dataset(sub_train_feature_update_source, sub_train_label_update_source)
            target_train_set = brain_dataset(sub_train_feature_update_target, sub_train_label_update_target)
            source_train_loader = torch.utils.data.DataLoader(source_train_set, batch_size=batch_size, shuffle=True)
            target_train_loader = torch.utils.data.DataLoader(target_train_set, batch_size=batch_size, shuffle=True)


        # form the new data training set if updat_trial
        if (trial_idx+1) % update_trial == 0:
            sub_newdata_data_update = []
            sub_newdata_label_update = []
            indices = np.where(combined_label_array == train_label_now_)
            #selected_indices = indices[0][-update_trial * batch_size_online:]
            selected_indices = indices[0][-trial_pre:]
            sub_newdata_data_update.append(combined_feature_array[selected_indices])
            sub_newdata_label_update.append(combined_label_array[selected_indices])
            sub_newdata_data_update = np.concatenate(sub_newdata_data_update, axis=0)
            sub_newdata_label_update = np.concatenate(sub_newdata_label_update, axis=0)
            sub_newdata_datalabel = brain_dataset(sub_newdata_data_update, sub_newdata_label_update)
            sub_newdata_datalabel_loader = torch.utils.data.DataLoader(sub_newdata_datalabel, batch_size=batch_size, shuffle=True)


        # the loss function
        criterion = nn.CrossEntropyLoss()

        # form the online test set 
        _sub_updating_predict = brain_dataset(sub_train_feature_batches, sub_train_label_batches)
        sub_updating_predict = torch.utils.data.DataLoader(_sub_updating_predict, batch_size=sub_train_feature_batches.shape[0], shuffle=False)

        print("********** Online simulation trial: {} ***********".format(trial_idx))
        start_time_infer = time.time()
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
        class_predictions_arrays.extend(class_predictions_array.tolist())
        labels_arrays.extend(labels_array.tolist())
        #accuracies_per_class.append(accuracy_per_class)
        # specially recording the corresponding accuracy of class 0 for further validation 
        if ground_truth_label[0] == 0.0:
            accuracies_per_class_iterations_Rest.append([ground_truth_label[0], predict_accu/100])
        
        stop_time_infer = time.time()
        time_infer = stop_time_infer - start_time_infer
        write_inference_time(os.path.join(Online_result_save_rootdir, sub_name), time_infer)

        print("predict accuracy: {}".format(predict_accu))
        print("predict accuracy per class: {}".format(accuracy_per_class))

        if (trial_idx + 1) % update_trial == 0:           
            print("******* Updating the model trial: {} ************".format(trial_idx))
            start_time = time.time()
            experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
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
                accuracy_per_class_iter_Rest = compute_total_accuracy_per_class(accuracies_per_class_iterations_Rest)
            
            #training loop
            # set the best validation accuracy
            if train_label_now_[0] != 0:
                best_val_accuracy = 0.8 * 100 * accuracy_per_class_iter[int(train_label_now_[0])]
                best_train_accuracy = 0.8 * 100 * (accuracy_per_class_iter[int(train_label_exemplars[0])] + accuracy_per_class_iter_Rest[0]\
                                                                )/2
            else:
                best_val_accuracy = 0.8 * 100 * accuracy_per_class_iter_Rest[0]
                best_train_accuracy = 0.8 * 100 * (accuracy_per_class_iter[1] + accuracy_per_class_iter[2]\
                                                                )/2

            is_best = False

            epoch_train_loss = []
            epoch_train_accuracy = []
            epoch_validation_accuracy = []
            
            result_save_dict = dict()
            
            _n_epoch_online = n_epoch_online

            criterion = nn.CrossEntropyLoss()
            #optimizer = torch.optim.Adam(model.encoder_output.encoder.Encoder_Cls.parameters(), lr=lr)  # only update the cls part
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            #optimizer = torch.optim.AdamW(model.parameters(), lr=0.1*lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)  # using adamw as the optimizer
            print("cls updating")
            
            for epoch in trange(_n_epoch_online, desc='online classification update'):
                
                #average_loss_this_epoch = train_one_epoch_label_distillation(model, optimizer, criterion, sub_newdata_datalabel_loader, sub_oldclass_datalabels_distill_loader, device, T=2.0, alpha=alpha_distill)
                average_loss_this_epoch = train_one_epoch_logitlabel_distillation(model, optimizer, criterion, sub_newdata_datalabel_loader, sub_oldclass_datalogits_distill_loader, sub_oldclass_datalabels_distill_loader, device, T=2.0, alpha=alpha_distill)
                #average_loss_this_epoch = train_one_epoch_logit_distillation(model, optimizer, criterion, sub_newdata_datalabel_loader, sub_oldclass_datalogits_distill_loader, device, T=2, alpha=alpha_distill)
                #average_loss_this_epoch = train_one_epoch_fealogitlabel_distillation(model, optimizer, criterion, sub_newdata_datalabel_loader, sub_oldclass_datalogits_distill_loader, sub_oldclass_datalabels_distill_loader,\
                #                                                                     sub_oldclass_datafea_distill_loader, sub_newclass_fealabel_distill_loader, sub_newdata_datalabel_loader, device, T=2.0, alpha=alpha_distill)
                val_accuracy, _, _ , _ = eval_model_fea(model, sub_newdata_datalabel_loader, device)
                #val_accuracy, _, _, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_newdata_datalabel_loader, device)
                train_accuracy, _, _ , _ = eval_model_fea(model, sub_oldclass_datalabels_distill_loader, device)
                
                epoch_train_loss.append(average_loss_this_epoch)
                epoch_train_accuracy.append(train_accuracy)
                epoch_validation_accuracy.append(val_accuracy)

                #update is_best flag, only when the accuracies of two classes of motor imagery are larger than random choice
                is_best = (val_accuracy >= best_val_accuracy) and (train_accuracy >= best_train_accuracy)
                #is_best = (train_accuracy >= best_train_accuracy)
                #is_best = (train_accuracy>=50*(accuracy_per_class_iter[int(train_label_exemplars[0])] + accuracy_per_class_iter[0]\
                #                                                            )/2) and (val_accuracy>=50*accuracy_per_class_iter[int(train_label_now_[0])])
                
                if is_best:
                    print("best_val_accuracy: {}".format(val_accuracy))
                    print("best_train_accuracy: {}".format(train_accuracy))
                    best_val_accuracy = val_accuracy
                    best_train_accuracy = train_accuracy

                    # using the momentum updating method
                    original_state_dict = torch.load(os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                    current_state_dict = model.state_dict()  # load the last model and the current model parameters
                    new_state_dict = {}
                    for key in current_state_dict.keys():
                        new_state_dict[key] = para_m * original_state_dict[key] + (1-para_m) * current_state_dict[key]  # updating the model in a momentum way
                    
                    torch.save(new_state_dict, os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                    
                    #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                    #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder.pt'))
                    #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))

                    result_save_dict['bestepoch_val_accuracy'] = val_accuracy
                    """for cls_i in range(accuracy_per_class.shape[0]):
                        result_save_dict['class_accuracy_' + str(cls_i)] = accuracy_per_class[cls_i]"""
                
                #if (best_val_accuracy >= 80.0) and (best_train_accuracy >= 80.0):
                #    break

            # updating the whole model
            if (trial_idx+1) % update_wholeModel == 0:
                print("******* Updating the whole model trial: {} ************".format(trial_idx))
                #criterion = MultiClassNpFocalLoss(device=device, alpha=[0.45,0.3,0.25])
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                whole_model_is_best = False
                whole_model_best_val_accuracy = 0
                
                _n_epoch_online = n_epoch_online * 2
                
                
                for epoch in trange(_n_epoch_online, desc='online classification update whole model'):
                    # initially calculate the memory bank for source and target data in each epoch 
                    memoryBank_source, memoryBank_target = eval_model_fea_classPrototypes(model, source_train_loader, target_train_loader, device, classes=3)    
                
                    #average_loss_this_epoch = train_one_epoch_fea(model, optimizer, criterion, source_train_loader, device)
                    average_loss_this_epoch = train_one_epoch_fea_MMDContrastive_targetcls_iter(model, optimizer, criterion, source_train_loader, target_train_loader, memoryBank_source, memoryBank_target, device, cons_beta=cons_rate)
                    whole_model_val_accuracy, _, _, _, _, whole_model_accuracy_per_class = eval_model_confusion_matrix_fea(model, target_train_loader, device)
                    whole_model_train_accuracy, _, _ , _ = eval_model_fea(model, source_train_loader, device)
                    
                    epoch_train_loss.append(average_loss_this_epoch)
                    epoch_train_accuracy.append(whole_model_train_accuracy)
                    epoch_validation_accuracy.append(whole_model_val_accuracy)

                    
                    whole_model_is_best = (whole_model_val_accuracy >= whole_model_best_val_accuracy)
                    
                    if whole_model_is_best:
                        #whole_model_is_best = False

                        print("whole model best_val_accuracy: {}".format(whole_model_val_accuracy))
                        #print("whole model best_train_accuracy: {}".format(whole_model_train_accuracy))
                        whole_model_best_val_accuracy = whole_model_val_accuracy
                        #best_train_accuracy = train_accuracy

                        torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.pt'))
                        #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder.pt'))
                        #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))

                        result_save_dict['bestepoch_val_accuracy'] = whole_model_val_accuracy
                    

            
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

def Online_data_painiting(args_dict):
    """
    Online simulation part
    this function is used only to show the data 
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
        sub_train_feature_array_1, sub_train_label_array_1 = Online_simulation_read_csv_windows_preprocess_normalization(folder_path=Offline_folder_path, sub_file=sub_name, trial_pre=50, preprocess=False,\
                                                                                                proportion=proportion, batch_size_online=batch_size_online, \
                                                                                                    pattern=  [1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 
                                                                                                               2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 
                                                                                                               2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 
                                                                                                               2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 
                                                                                                               1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 
                                                                                                               2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 
                                                                                                               1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 
                                                                                                               2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0])
    trial_idx = 3
    sub_train_feature_batches = sub_train_feature_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online, :, :]
    sub_train_label_batches = sub_train_label_array_1[trial_idx * batch_size_online : (trial_idx + 1) * batch_size_online]
    data2paint = sub_train_feature_batches[0,11:18,:]
    plt.figure(figsize=(16, 9), frameon=False)
    for i in range(data2paint.shape[0]):
        offset = 2*i
        plt.plot(data2paint[i,:] + offset, color='#838482', linewidth=4.0)
    
    # 隐藏刻度标签但保留刻度线
    plt.tick_params(axis='both', which='both', length=6, labelleft=False, labelbottom=False)
    # 减少图像白边
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{Online_result_save_rootdir}/data_sample.png')
    plt.savefig(f'{Online_result_save_rootdir}/data_sample.pdf')
    plt.close()

    


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
    parser.add_argument('--para_m', default=0.99, type=float, help="hyper parameter for momentum updating")
    parser.add_argument('--cons_rate', default=0.01, type=float, help="hyper parameter for constractive loss")
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
    para_m = args.para_m
    cons_rate = args.cons_rate

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
    args_dict.para_m = para_m
    args_dict.cons_rate = cons_rate

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
        #Online_simulation_synthesizing_results_polynomial(Online_result_save_rootdir=Online_result_save_rootdir)
        Online_simulation_synthesizing_results_polynomial_avg(Online_result_save_rootdir=Online_result_save_rootdir)
        #Online_simulation_synthesizing_results_2cls_linear(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_linear_perclass(Online_result_save_rootdir)
        Online_simulation_synthesizing_results_polynomial_avgF1(Online_result_save_rootdir)

    if mode == 'comparison':
        methods = ['baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1_new', 'method5_EEGNet_baseline_1_9batchsize_Rest_2_mixed_3_new', 'method5_EEGNet_baseline_2_7_9batchsize_Rest_2_mixed_3_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_new' ,'method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new']
        methods_perclass = ['method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1_new','method5_EEGNet_baseline_1_9batchsize_Rest_2_mixed_3_new', 'method5_EEGNet_baseline_2_7_9batchsize_Rest_2_mixed_3_new','method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new', 'method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new']
        #methods = ['baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_1', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_1_3', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_2_2', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_2']
        #methods = ['baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_1', 'method5_encoder3_pretrainlight_baseline_1_9batchsize_Rest_2_mixed_2','method5_encoder3_pretrainlight_baseline_2_4_9batchsize_Rest_2_mixed_1', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_2']  #'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1'
        #methods = ['baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_1','method5_encoder3_pretrainlight_baseline_2_4_9batchsize_Rest_2_mixed_1']  #'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1'
        #methods = ['method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_1_4', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_3', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_2_2', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_6_mixed_3']  #'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_6_ablation_1'
        Online_simulation_synthesizing_results_comparison_linear(Online_result_save_rootdir, methods)
        Online_simulation_synthesizing_results_comparison_polynomial(Online_result_save_rootdir, methods)
        Online_simulation_synthesizing_results_comparison_polynomial_optimized(Online_result_save_rootdir, methods)
        Online_simulation_synthesizing_results_comparison_polynomial_optimized_perclass(Online_result_save_rootdir, methods_perclass)
        #Online_simulation_synthesizing_results_comparison_linear_2cls(Online_result_save_rootdir, methods)
    
    if mode == 'paint':
        Online_data_painiting(args_dict)