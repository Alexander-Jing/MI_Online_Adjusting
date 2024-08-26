#!/bin/bash

# 你的程序的路径
program_path="Online_simulation/Online_train_EEGNet_simulation_baseline1_noupdate_noRest_val_6_9batchsize_Rest_mixed_2.py"

# 你的工作目录的路径
workspace_folder="/home/jyt/workspace/MI_Online_Adjusting"

# 改变当前工作目录
cd $workspace_folder

# 循环运行你的程序
for i in $(seq 1 25)
do
  sub_name=$(printf "%03d" $i)
  python3 $program_path \
    --seed 42 \
    --gpu_idx 0 \
    --sub_name $sub_name \
    --Offline_folder_path "/home/jyt/workspace/transfer_models/datasets_MI/hand_elbow/derivatives" \
    --windows_num 120 \
    --trial_pre 120 \
    --proportion 0.75 \
    --Offline_result_save_rootdir "Offline_simulation_experiments/method2_EEGNet_val_classval_pretrainlight_unfreeze_new" \
    --Online_folder_path "Online_DataCollected" \
    --Online_result_save_rootdir "Online_simulation_experiments/baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new" \
    --restore_file "None" \
    --n_epoch_offline 16 \
    --n_epoch_online 1 \
    --batch_size 12 \
    --mode "online" \
    --batch_size_online 9 \
    --trial_nums 96 \
    --best_validation_path "lr0.01_dropout0.5" \
    --unfreeze_encoder_offline "True" \
    --unfreeze_encoder_online "True" \
    --alpha_distill  0.60 \
    --update_trial 1 \

done