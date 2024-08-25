#!/bin/bash

#export PYTHONPATH="${PYTHONPATH}:/home/jyt/workspace/MI_Online"
#cd /home/jyt/workspace/MI_Online/Offline_models/
#python Offline_train_EEGNet.py --seed 42 --gpu_idx 0 --sub_name 'Jyt' --folder_path '/home/jyt/workspace/MI_Online/DataCollected_Jyt' --windows_num 149 --proportion 0.8 --result_save_rootdir '/home/jyt/workspace/MI_Online/Offline_experiments' --restore_file 'None' --n_epoch 64


export PYTHONPATH="${PYTHONPATH}/home/jyt/workspace/MI_Online"
cd ${PYTHONPATH}/Offline_models/
python Offline_train_EEGNet.py --seed 42 --gpu_idx 0 --sub_name 'Jyt' --folder_path ${PYTHONPATH}/DataCollected_Jyt --windows_num 149 --proportion 0.8 --result_save_rootdir ${PYTHONPATH}/Offline_experiments --restore_file 'None' --n_epoch 64
