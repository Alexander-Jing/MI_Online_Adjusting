import os
import numpy as np
import csv
import argparse

def extract_experiment_setting(experiment_name):
    
    print('Passed in experiment_name is {}'.format(experiment_name), flush = True)
    
    hyper_parameter_dict = {}
    
    #hyperparameter to extract
    lr = experiment_name.split('lr')[-1].split('_')[0]
    dropout = experiment_name.split('dropout')[-1]
    
    #record to dict
    hyper_parameter_dict['lr'] = lr
    hyper_parameter_dict['dropout'] = dropout  
    
    #print values
    header = ' checking experiment '.center(100, '-')
    print(header)
    print('lr: {}; dropout: {}'.format(lr, dropout))
    
    print('\n')
    
    return hyper_parameter_dict

def extract_experiment_performance(experiment_dir, experiment_name):
    
    
    performance_file_fullpath = os.path.join(experiment_dir, experiment_name, 'result_analysis/performance.txt')
    returned_file = None
    
    with open(performance_file_fullpath, 'r') as f: #only read mode, do not modify
        returned_file = f.read()
        
        validation_accuracy = round(float(returned_file.split('highest validation accuracy: ')[1].split('\n')[0]), 3)
        
        print('validation_accuracy: {}'.format(validation_accuracy))
        
        
    return returned_file, validation_accuracy

def extract_experiment_performance_confusionMatrix(experiment_dir, experiment_name):
    
    
    performance_file_fullpath = os.path.join(experiment_dir, experiment_name, 'result_analysis/performance.txt')
    returned_file = None
    class_accuracies = []

    with open(performance_file_fullpath, 'r') as f: #only read mode, do not modify
        returned_file = f.read()
        lines = returned_file.split('\n')
        validation_accuracy = round(float(returned_file.split('highest validation accuracy: ')[1].split('\n')[0]), 3)
        for line in lines:
            # 检查该行是否包含 'class_accuracy_'
            if 'class_accuracy_' in line:
                # 提取精度
                accuracy = float(line.split(':')[1])
                # 将结果添加到列表中
                class_accuracies.append(accuracy)
        
        print('validation_accuracy: {}'.format(validation_accuracy))
        
        
    return returned_file, validation_accuracy, class_accuracies

def synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir):
    
    experiments = os.listdir(experiment_dir)
    incomplete_experiment_writer = open(os.path.join(summary_save_dir, 'incomplete_experiment_list.txt'), 'w')
    summary_filename = os.path.join(summary_save_dir, 'hypersearch_summary.csv')
    
    with open(summary_filename, mode='w') as csv_file:
        
        fieldnames = ['validation_accuracy', 'lr', 'dropout',  'performance_string', 'experiment_folder', 'status']
        fileEmpty = os.stat(summary_filename).st_size==0
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fileEmpty:
            writer.writeheader()
        
        best_validation_accuracy = 0.0
        best_validation_class_accuracy = []
        best_validation_path = ''
        best_validation_writer = open(os.path.join(summary_save_dir, 'best_validation_model.txt'), 'w')
        
        for experiment_name in experiments:
            if experiment_name !='hypersearch_summary':
                experiment_folder = os.path.join(experiment_dir, experiment_name)
                
                experiment_summary = extract_experiment_setting(experiment_name)
                
                try:
                    returned_file, validation_accuracy, class_accurcies = extract_experiment_performance_confusionMatrix(experiment_dir, experiment_name)
                    print('Able to extract performance', flush = True)
                    
                    is_best = validation_accuracy >= best_validation_accuracy
                    if is_best:
                        best_validation_accuracy = validation_accuracy
                        best_validation_path = experiment_name
                        best_validation_class_accuracy = class_accurcies
                    
                    experiment_summary.update(validation_accuracy=validation_accuracy, performance_string=returned_file, experiment_folder=experiment_folder, status='Completed')
                    print('Able to update experiment_summary\n\n')
                    
                
                except:
                    print(' NOT ABLE TO PROCESS {} \n\n'.format(experiment_dir + '/' + experiment_name).center(100, '-'), flush=True)
                    
                    incomplete_experiment_writer.write(f"{experiment_name}\n\n")
                    experiment_summary.update(validation_accuracy='NA', performance_string='NA', experiment_folder=experiment_folder, status='Incompleted')
                    
                writer.writerow(experiment_summary)
        
        print('best_validation_path: ' + os.path.join(experiment_dir, best_validation_path, 'checkpoint'))
        print('best_validation_accuracy: ' + str(best_validation_accuracy))
        print('best_validation_class_accuracy: ')
        print(best_validation_class_accuracy)

        best_validation_writer.write(f"{best_validation_path}\n\n")
        best_validation_writer.write(f"{best_validation_accuracy}\n\n")
        
        incomplete_experiment_writer.close()

    return best_validation_class_accuracy, best_validation_path
    
        

def synthesize_hypersearch(experiment_dir, summary_save_dir):
    
    experiments = os.listdir(experiment_dir)
    incomplete_experiment_writer = open(os.path.join(summary_save_dir, 'incomplete_experiment_list.txt'), 'w')
    summary_filename = os.path.join(summary_save_dir, 'hypersearch_summary.csv')
    
    with open(summary_filename, mode='w') as csv_file:
        
        fieldnames = ['validation_accuracy', 'lr', 'dropout',  'performance_string', 'experiment_folder', 'status']
        fileEmpty = os.stat(summary_filename).st_size==0
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fileEmpty:
            writer.writeheader()
        
        best_validation_accuracy = 0.0
        best_validation_path = ''
        best_validation_writer = open(os.path.join(summary_save_dir, 'best_validation_model.txt'), 'w')
        
        for experiment_name in experiments:
            if experiment_name !='hypersearch_summary':
                experiment_folder = os.path.join(experiment_dir, experiment_name)
                
                experiment_summary = extract_experiment_setting(experiment_name)
                
                try:
                    returned_file, validation_accuracy = extract_experiment_performance(experiment_dir, experiment_name)
                    print('Able to extract performance', flush = True)
                    
                    is_best = validation_accuracy >= best_validation_accuracy
                    if is_best:
                        best_validation_accuracy = validation_accuracy
                        best_validation_path = experiment_name

                    experiment_summary.update(validation_accuracy=validation_accuracy, performance_string=returned_file, experiment_folder=experiment_folder, status='Completed')
                    print('Able to update experiment_summary\n\n')
                    
                
                except:
                    print(' NOT ABLE TO PROCESS {} \n\n'.format(experiment_dir + '/' + experiment_name).center(100, '-'), flush=True)
                    
                    incomplete_experiment_writer.write(f"{experiment_name}\n\n")
                    experiment_summary.update(validation_accuracy='NA', performance_string='NA', experiment_folder=experiment_folder, status='Incompleted')
                    
                writer.writerow(experiment_summary)
        
        print('best_validation_path: ' + os.path.join(experiment_dir, best_validation_path, 'checkpoint'))
        print('best_validation_accuracy: ' + str(best_validation_accuracy))
        best_validation_writer.write(f"{best_validation_path}\n\n")
        best_validation_writer.write(f"{best_validation_accuracy}\n\n")
        
        incomplete_experiment_writer.close()       

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='synthesizing hyperparameter search results')
    parser.add_argument('--experiment_dir')
    
    #parse args
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    assert os.path.exists(experiment_dir),'The passed in experiment_dir {} does not exist'.format(experiment_dir)
    
    summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
    
    if not os.path.exists(summary_save_dir):
        os.makedirs(summary_save_dir)
    
    synthesize_hypersearch(experiment_dir, summary_save_dir)
    
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    