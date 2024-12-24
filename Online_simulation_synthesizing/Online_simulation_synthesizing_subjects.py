import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score

def Online_simulation_synthesizing_results(Online_result_save_rootdir):
   # 初始化一个空的列表来存储所有被试的数据
    all_data = []

    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                data = np.loadtxt(f'{subject_folder}/{file}/predict_accuracies.csv', delimiter=',',skiprows=1)

                # 将数据添加到all_data列表中
                all_data.append(data)

    # 使用pandas DataFrame来存储数据
    all_data_df = pd.DataFrame(np.column_stack(all_data))

    # 计算每一行的均值和方差
    means = all_data_df.mean(axis=1)
    std_devs = all_data_df.std(axis=1)

    # 将all_data_df，means和std_devs存储到一个CSV文件中
    all_data_df.to_csv(f'{Online_result_save_rootdir}/all_data.csv', index_label='Iteration')
    means.to_csv(f'{Online_result_save_rootdir}/means.csv', header=['Mean'])
    std_devs.to_csv(f'{Online_result_save_rootdir}/std_devs.csv', header=['Std Dev'])

    # 创建一个新的figure
    sns.set()

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 绘制均值折线图
    
    sns.lineplot(data=means, label='Mean')

    # 添加方差
    plt.fill_between(range(len(means)), means-std_devs , means+std_devs , color='b', alpha=.1)

    # 添加图例
    plt.legend()

    # 保存图形到data_folder文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects.png')

def Online_simulation_synthesizing_results_comparison(Online_result_save_rootdir, methods):
    # 设置seaborn样式
    sns.set()

    # 遍历所有的方法
    for method in methods:
        # 读取means.csv和std_devs.csv文件
        means = pd.read_csv(f'{Online_result_save_rootdir}/{method}/means.csv', index_col=0)
        std_devs = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_devs.csv', index_col=0)

        # 绘制均值折线图
        sns.lineplot(data=means['Mean'], label=f'{method} Mean')

        # 添加标准差
        plt.fill_between(range(len(means)), means['Mean']-std_devs['Std Dev'], means['Mean']+std_devs['Std Dev'], alpha=.1)

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration methods comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 添加图例
    plt.legend()

    # 保存图形到Online_result_save_rootdir文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_comparison.png')

def linear_func(x, a, b):
    return a * x + b

def Online_simulation_synthesizing_results_linear(Online_result_save_rootdir):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []

    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                data = np.loadtxt(f'{subject_folder}/{file}/predict_accuracies.csv', delimiter=',',skiprows=1)

                # 将数据添加到all_data列表中
                all_data.append(data)

    # 使用pandas DataFrame来存储数据
    all_data_df = pd.DataFrame(np.column_stack(all_data))

    # 计算每一行的均值和标准差
    means = all_data_df.mean(axis=1)
    std_devs = all_data_df.std(axis=1)

    # 将all_data_df，means和std_devs存储到一个CSV文件中
    all_data_df.to_csv(f'{Online_result_save_rootdir}/all_data.csv', index_label='Iteration')
    means.to_csv(f'{Online_result_save_rootdir}/means.csv', header=['Mean'])
    std_devs.to_csv(f'{Online_result_save_rootdir}/std_devs.csv', header=['Std Dev'])

    # 使用最小二乘法拟合一次函数
    popt, pcov = curve_fit(linear_func, means.index, means)

    # 创建一个新的figure
    sns.set()

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 绘制均值折线图
    sns.lineplot(data=means, label='Mean')

    # 绘制拟合的一次函数
    plt.plot(means.index, linear_func(means.index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}')

    # 添加标准差
    plt.fill_between(range(len(means)), means-std_devs , means+std_devs , color='b', alpha=.1)

    # 添加图例
    plt.legend()

    # 保存图形到data_folder文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_linear.png')
    plt.close()

# 定义一个3次多项式函数
def polynomial_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d
"""
# 定义一个函数来合成在线模拟结果
def Online_simulation_synthesizing_results_polynomial(Online_result_save_rootdir):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []

    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                data = np.loadtxt(f'{subject_folder}/{file}/predict_accuracies.csv', delimiter=',',skiprows=1)

                # 将数据添加到all_data列表中
                all_data.append(data)

    # 使用pandas DataFrame来存储数据
    all_data_df = pd.DataFrame(np.column_stack(all_data))

    # 计算每一行的均值和标准差
    means = all_data_df.mean(axis=1)
    std_devs = all_data_df.std(axis=1)

    # 将all_data_df，means和std_devs存储到一个CSV文件中
    all_data_df.to_csv(f'{Online_result_save_rootdir}/all_data.csv', index_label='Iteration')
    means.to_csv(f'{Online_result_save_rootdir}/means.csv', header=['Mean'])
    std_devs.to_csv(f'{Online_result_save_rootdir}/std_devs.csv', header=['Std Dev'])

    # 使用最小二乘法拟合3次多项式函数
    popt, pcov = curve_fit(polynomial_func, means.index, means)

    # 创建一个新的figure
    sns.set()

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 绘制均值折线图
    sns.lineplot(data=means, label='Mean')

    # 绘制拟合的3次多项式函数
    plt.plot(means.index, polynomial_func(means.index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, d={popt[3]:.3f}')

    # 添加标准差
    plt.fill_between(range(len(means)), means-std_devs , means+std_devs , color='b', alpha=.1)

    # 添加图例
    plt.legend()

    # 保存图形到data_folder文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_polynomial.png')
    plt.close()

"""
def Online_simulation_synthesizing_results_polynomial(Online_result_save_rootdir, random_acc=33.0):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []

    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                data = np.loadtxt(f'{subject_folder}/{file}/predict_accuracies.csv', delimiter=',',skiprows=1)

                # 将数据添加到all_data列表中
                all_data.append(data)

    # 使用pandas DataFrame来存储数据
    all_data_df = pd.DataFrame(np.column_stack(all_data))

    # 计算每一行的均值和标准差
    means = all_data_df.mean(axis=1).rolling(window=4, min_periods=1).mean()  # 使用滑动窗口平均来平滑均值曲线
    std_devs = all_data_df.std(axis=1)

    # 将all_data_df，means和std_devs存储到一个CSV文件中
    all_data_df.to_csv(f'{Online_result_save_rootdir}/all_data.csv', index_label='Iteration')
    means.to_csv(f'{Online_result_save_rootdir}/means.csv', header=['Mean'])
    std_devs.to_csv(f'{Online_result_save_rootdir}/std_devs.csv', header=['Std Dev'])

    # 使用最小二乘法拟合3次多项式函数
    popt, pcov = curve_fit(polynomial_func, means.index, means)

    # 创建一个新的figure
    sns.set()

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 绘制均值折线图
    sns.lineplot(data=means, label='Mean')

    # 绘制拟合的3次多项式函数
    plt.plot(means.index, polynomial_func(means.index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, d={popt[3]:.3f}')

    # 添加标准差
    plt.fill_between(range(len(means)), means-std_devs , means+std_devs , color='b', alpha=.1)

    # 绘制y=random_acc的虚线
    plt.axhline(y=random_acc, color='g', linestyle='--', label=f'Random accuracy: {random_acc}')

    # 添加图例
    plt.legend()

    # 保存图形到data_folder文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_polynomial.png')
    plt.close()

def Online_simulation_synthesizing_results_polynomial_avg(Online_result_save_rootdir, random_acc=33.0, data_session_avg=24):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []
    all_data_avg = []

    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                data = np.loadtxt(f'{subject_folder}/{file}/predict_accuracies.csv', delimiter=',',skiprows=1)

                # 计算前data_session_avg个数据的均值
                data_session_avg_former = np.mean(data[:data_session_avg])
                # 计算后data_session_avg个数据的均值
                data_session_avg_latter = np.mean(data[-data_session_avg:])

                # 将计算出的均值添加到all_data_avg列表中
                all_data_avg.append([data_session_avg_former, data_session_avg_latter])

                # 将数据添加到all_data列表中
                all_data.append(data)

    # 使用pandas DataFrame来存储数据
    all_data_df = pd.DataFrame(np.column_stack(all_data))
    all_data_avg_df = pd.DataFrame(all_data_avg, columns=['data_session_avg_former', 'data_session_avg_latter'])

    # 计算每一行的均值和标准差
    #_means = all_data_df.mean(axis=1).rolling(window=4, min_periods=1)
    means = all_data_df.mean(axis=1).rolling(window=4, min_periods=1).mean()  # 使用滑动窗口平均来平滑均值曲线
    std_devs = all_data_df.std(axis=1)

    # 计算all_data_avg中所有data_session_avg_former和data_session_avg_latter的均值和标准差
    mean_former = all_data_avg_df['data_session_avg_former'].mean()
    std_dev_former = all_data_avg_df['data_session_avg_former'].std()
    mean_latter = all_data_avg_df['data_session_avg_latter'].mean()
    std_dev_latter = all_data_avg_df['data_session_avg_latter'].std()

    # 使用配对t检验分析显著性
    t_statistic, p_value = ttest_rel(all_data_avg_df['data_session_avg_former'], all_data_avg_df['data_session_avg_latter'])

    # 打印显著性结果
    print(f"t-statistic: {t_statistic}, p-value: {p_value}")
    if p_value < 0.05:
        print("The difference is statistically significant.")
    else:
        print("The difference is not statistically significant.")

    # 将均值、标准差和显著性结果存储在'Online_result_save_rootdir/all_data_avg_results.csv'里面
    results_df = pd.DataFrame({
        'Mean': [mean_former, mean_latter],
        'Std Dev': [std_dev_former, std_dev_latter],
        't-statistic': [t_statistic, t_statistic],
        'p-value': [p_value, p_value]
    }, index=['data_session_avg_former', 'data_session_avg_latter'])
    results_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg_results.csv')

    # 将all_data_df，means和std_devs存储到一个CSV文件中
    all_data_df.to_csv(f'{Online_result_save_rootdir}/all_data.csv', index_label='Iteration')
    means.to_csv(f'{Online_result_save_rootdir}/means.csv', header=['Mean'])
    std_devs.to_csv(f'{Online_result_save_rootdir}/std_devs.csv', header=['Std Dev'])
    #all_data_avg_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg.csv', index_label='Iteration')

    # 使用最小二乘法拟合3次多项式函数
    popt, pcov = curve_fit(polynomial_func, means.index, means)

    # 创建一个新的figure
    sns.set()

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 绘制均值折线图
    sns.lineplot(data=means, label='Mean')

    # 绘制拟合的3次多项式函数
    plt.plot(means.index, polynomial_func(means.index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, d={popt[3]:.3f}')

    # 添加标准差
    plt.fill_between(range(len(means)), means-std_devs , means+std_devs , color='b', alpha=.1)

    # 绘制y=random_acc的虚线
    plt.axhline(y=random_acc, color='g', linestyle='--', label=f'Random accuracy: {random_acc}')

    # 添加图例
    plt.legend()

    # 保存图形到data_folder文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_polynomial.png')
    plt.close()

def Online_simulation_synthesizing_results_polynomial_avg_1(Online_result_save_rootdir, random_acc=33.0, data_session_avg=24):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []
    all_data_avg = []

    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 使用numpy读取predict_accuracies.csv文件
        data = np.loadtxt(f'{subject_folder}/predict_accuracies.csv', delimiter=',',skiprows=1)

        # 计算前data_session_avg个数据的均值
        data_session_avg_former = np.mean(data[:data_session_avg])
        # 计算后data_session_avg个数据的均值
        data_session_avg_latter = np.mean(data[-data_session_avg:])

        # 将计算出的均值添加到all_data_avg列表中
        all_data_avg.append([data_session_avg_former, data_session_avg_latter])

        # 将数据添加到all_data列表中
        all_data.append(data)

    # 使用pandas DataFrame来存储数据
    all_data_df = pd.DataFrame(np.column_stack(all_data))
    all_data_avg_df = pd.DataFrame(all_data_avg, columns=['data_session_avg_former', 'data_session_avg_latter'])

    # 计算每一行的均值和标准差
    means = all_data_df.mean(axis=1).rolling(window=4, min_periods=1).mean()  # 使用滑动窗口平均来平滑均值曲线
    std_devs = all_data_df.std(axis=1)

    # 计算all_data_avg中所有data_session_avg_former和data_session_avg_latter的均值和标准差
    mean_former = all_data_avg_df['data_session_avg_former'].mean()
    std_dev_former = all_data_avg_df['data_session_avg_former'].std()
    mean_latter = all_data_avg_df['data_session_avg_latter'].mean()
    std_dev_latter = all_data_avg_df['data_session_avg_latter'].std()

    # 使用配对t检验分析显著性
    t_statistic, p_value = ttest_rel(all_data_avg_df['data_session_avg_former'], all_data_avg_df['data_session_avg_latter'])

    # 打印显著性结果
    print(f"t-statistic: {t_statistic}, p-value: {p_value}")
    if p_value < 0.05:
        print("The difference is statistically significant.")
    else:
        print("The difference is not statistically significant.")

    # 将均值、标准差和显著性结果存储在'Online_result_save_rootdir/all_data_avg_results.csv'里面
    results_df = pd.DataFrame({
        'Mean': [mean_former, mean_latter],
        'Std Dev': [std_dev_former, std_dev_latter],
        't-statistic': [t_statistic, t_statistic],
        'p-value': [p_value, p_value]
    }, index=['data_session_avg_former', 'data_session_avg_latter'])
    results_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg_results.csv')

    # 将all_data_df，means和std_devs存储到一个CSV文件中
    all_data_df.to_csv(f'{Online_result_save_rootdir}/all_data.csv', index_label='Iteration')
    means.to_csv(f'{Online_result_save_rootdir}/means.csv', header=['Mean'])
    std_devs.to_csv(f'{Online_result_save_rootdir}/std_devs.csv', header=['Std Dev'])
    #all_data_avg_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg.csv', index_label='Iteration')

    # 使用最小二乘法拟合3次多项式函数
    popt, pcov = curve_fit(polynomial_func, means.index, means)

    # 创建一个新的figure
    sns.set()

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 绘制均值折线图
    sns.lineplot(data=means, label='Mean')

    # 绘制拟合的3次多项式函数
    plt.plot(means.index, polynomial_func(means.index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, d={popt[3]:.3f}')

    # 添加标准差
    plt.fill_between(range(len(means)), means-std_devs , means+std_devs , color='b', alpha=.1)

    # 绘制y=random_acc的虚线
    plt.axhline(y=random_acc, color='g', linestyle='--', label=f'Random accuracy: {random_acc}')

    # 添加图例
    plt.legend()

    # 保存图形到data_folder文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_polynomial.png')
    plt.close()

def Online_simulation_synthesizing_results_polynomial_avgF1(Online_result_save_rootdir, random_acc=33.0, data_session_avg=24*9, pattern=[0,0,0,0]):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []
    all_data_avg = []
    pattern=[1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0]
    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                predicts = np.loadtxt(f'{subject_folder}/{file}/class_predictions_arrays.csv', delimiter=',',skiprows=1)
                labels = np.loadtxt(f'{subject_folder}/{file}/labels_arrays.csv', delimiter=',',skiprows=1)
                
                # 计算前data_session_avg个数据的均值
                predicts_session_former = predicts[:data_session_avg]
                labels_session_former = labels[:data_session_avg]
                # 计算后data_session_avg个数据的均值
                predicts_session_latter = predicts[-data_session_avg:]
                labels_session_latter = labels[-data_session_avg:]

                accuracy_former = accuracy_score(labels_session_former, predicts_session_former)
                F1score_former = f1_score(labels_session_former, predicts_session_former, average='macro')

                accuracy_latter = accuracy_score(labels_session_latter, predicts_session_latter)
                F1score_latter = f1_score(labels_session_latter, predicts_session_latter, average='macro')
                # 将计算出的均值添加到all_data_avg列表中
                all_data_avg.append([accuracy_former, F1score_former, accuracy_latter, F1score_latter])

    # 使用pandas DataFrame来存储数据
    all_data_avg_df = pd.DataFrame(all_data_avg, columns=['accuracy_former', 'F1score_former', 'accuracy_latter', 'F1score_latter'])

    # 计算all_data_avg中所有data_session_avg_former和data_session_avg_latter的均值和标准差
    acc_mean_former = all_data_avg_df['accuracy_former'].mean()
    acc_std_dev_former = all_data_avg_df['accuracy_former'].std()
    f1_mean_former = all_data_avg_df['F1score_former'].mean()
    f1_std_dev_former = all_data_avg_df['F1score_former'].std()

    acc_mean_latter = all_data_avg_df['accuracy_latter'].mean()
    acc_std_dev_latter = all_data_avg_df['accuracy_latter'].std()
    f1_mean_latter = all_data_avg_df['F1score_latter'].mean()
    f1_std_dev_latter = all_data_avg_df['F1score_latter'].std()
    

    # 使用配对t检验分析显著性
    acc_t_statistic, acc_p_value = ttest_rel(all_data_avg_df['accuracy_former'], all_data_avg_df['accuracy_latter'])
    f1_t_statistic, f1_p_value = ttest_rel(all_data_avg_df['F1score_former'], all_data_avg_df['F1score_latter'])


    # 打印显著性结果
    print(f"t-statistic: {acc_t_statistic}, p-value: {acc_p_value}")
    if acc_p_value < 0.05:
        print("The difference in acc is statistically significant.")
    else:
        print("The difference in acc is not statistically significant.")

    print(f"t-statistic: {f1_t_statistic}, p-value: {f1_p_value}")
    if f1_p_value < 0.05:
        print("The difference in f1 is statistically significant.")
    else:
        print("The difference in f1 is not statistically significant.")
    
    # 将均值、标准差和显著性结果存储在'Online_result_save_rootdir/all_data_avg_results.csv'里面
    results_df = pd.DataFrame({
        'Mean': [acc_mean_former, acc_mean_latter, f1_mean_former, f1_mean_latter],
        'Std Dev': [acc_std_dev_former, acc_std_dev_latter, f1_std_dev_former, f1_std_dev_latter],
        't-statistic': [acc_t_statistic, acc_t_statistic, f1_t_statistic, f1_t_statistic],
        'p-value': [acc_p_value, acc_p_value, f1_p_value, f1_p_value]
    }, index=['acc_avg_former', 'acc_avg_latter', 'f1_avg_former', 'f1_avg_latter'])
    results_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg_results.csv')
    
    all_data_avg_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg.csv', index_label='sub')

def Online_simulation_synthesizing_results_polynomial_avgF1_noRest(Online_result_save_rootdir, noRest=True, random_acc=33.0, data_session_avg=24*9, pattern=[0,0,0,0]):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []
    all_data_avg = []
    pattern=[1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0]
    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                predicts = np.loadtxt(f'{subject_folder}/{file}/class_predictions_arrays.csv', delimiter=',',skiprows=1)
                labels = np.loadtxt(f'{subject_folder}/{file}/labels_arrays.csv', delimiter=',',skiprows=1)
                if noRest:
                    # 过滤掉类别为0的情况
                    mask_former = (labels[:data_session_avg] != 0.)
                    predicts_session_former = predicts[:data_session_avg][mask_former]
                    labels_session_former = labels[:data_session_avg][mask_former]
                    
                    mask_latter = (labels[-data_session_avg:] != 0.)
                    predicts_session_latter = predicts[-data_session_avg:][mask_latter]
                    labels_session_latter = labels[-data_session_avg:][mask_latter]
                    #print("exclude the Rest class")
                else:
                    # 计算前data_session_avg个数据的均值
                    predicts_session_former = predicts[:data_session_avg]
                    labels_session_former = labels[:data_session_avg]
                    # 计算后data_session_avg个数据的均值
                    predicts_session_latter = predicts[-data_session_avg:]
                    labels_session_latter = labels[-data_session_avg:]

                accuracy_former = accuracy_score(labels_session_former, predicts_session_former)
                F1score_former = f1_score(labels_session_former, predicts_session_former, average='macro')

                accuracy_latter = accuracy_score(labels_session_latter, predicts_session_latter)
                F1score_latter = f1_score(labels_session_latter, predicts_session_latter, average='macro')
                # 将计算出的均值添加到all_data_avg列表中
                all_data_avg.append([accuracy_former, F1score_former, accuracy_latter, F1score_latter])

    # 使用pandas DataFrame来存储数据
    all_data_avg_df = pd.DataFrame(all_data_avg, columns=['accuracy_former', 'F1score_former', 'accuracy_latter', 'F1score_latter'])

    # 计算all_data_avg中所有data_session_avg_former和data_session_avg_latter的均值和标准差
    acc_mean_former = all_data_avg_df['accuracy_former'].mean()
    acc_std_dev_former = all_data_avg_df['accuracy_former'].std()
    f1_mean_former = all_data_avg_df['F1score_former'].mean()
    f1_std_dev_former = all_data_avg_df['F1score_former'].std()

    acc_mean_latter = all_data_avg_df['accuracy_latter'].mean()
    acc_std_dev_latter = all_data_avg_df['accuracy_latter'].std()
    f1_mean_latter = all_data_avg_df['F1score_latter'].mean()
    f1_std_dev_latter = all_data_avg_df['F1score_latter'].std()
    

    # 使用配对t检验分析显著性
    acc_t_statistic, acc_p_value = ttest_rel(all_data_avg_df['accuracy_former'], all_data_avg_df['accuracy_latter'])
    f1_t_statistic, f1_p_value = ttest_rel(all_data_avg_df['F1score_former'], all_data_avg_df['F1score_latter'])


    # 打印显著性结果
    print(f"t-statistic: {acc_t_statistic}, p-value: {acc_p_value}")
    if acc_p_value < 0.05:
        print("The difference of MI classes without Rest in acc is statistically significant.")
    else:
        print("The difference of MI classes without Rest in acc is not statistically significant.")

    print(f"t-statistic: {f1_t_statistic}, p-value: {f1_p_value}")
    if f1_p_value < 0.05:
        print("The difference of MI classes without Rest in f1 is statistically significant.")
    else:
        print("The difference of MI classes without Rest in f1 is not statistically significant.")
    
    # 将均值、标准差和显著性结果存储在'Online_result_save_rootdir/all_data_avg_results.csv'里面
    results_df = pd.DataFrame({
        'Mean': [acc_mean_former, acc_mean_latter, f1_mean_former, f1_mean_latter],
        'Std Dev': [acc_std_dev_former, acc_std_dev_latter, f1_std_dev_former, f1_std_dev_latter],
        't-statistic': [acc_t_statistic, acc_t_statistic, f1_t_statistic, f1_t_statistic],
        'p-value': [acc_p_value, acc_p_value, f1_p_value, f1_p_value]
    }, index=['acc_avg_former', 'acc_avg_latter', 'f1_avg_former', 'f1_avg_latter'])
    results_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg_results_noRest.csv')

def Online_simulation_synthesizing_results_polynomial_avgF1_Rest(Online_result_save_rootdir, Rest=True, random_acc=33.0, data_session_avg=24*9, pattern=[0,0,0,0]):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []
    all_data_avg = []
    pattern=[1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0]
    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                predicts = np.loadtxt(f'{subject_folder}/{file}/class_predictions_arrays.csv', delimiter=',',skiprows=1)
                labels = np.loadtxt(f'{subject_folder}/{file}/labels_arrays.csv', delimiter=',',skiprows=1)
                if Rest:
                    # 类别为0的情况
                    mask_former = (labels[:data_session_avg] == 0.)
                    predicts_session_former = predicts[:data_session_avg][mask_former]
                    labels_session_former = labels[:data_session_avg][mask_former]
                    
                    mask_latter = (labels[-data_session_avg:] == 0.)
                    predicts_session_latter = predicts[-data_session_avg:][mask_latter]
                    labels_session_latter = labels[-data_session_avg:][mask_latter]
                    #print("exclude the Rest class")
                else:
                    # 计算前data_session_avg个数据的均值
                    predicts_session_former = predicts[:data_session_avg]
                    labels_session_former = labels[:data_session_avg]
                    # 计算后data_session_avg个数据的均值
                    predicts_session_latter = predicts[-data_session_avg:]
                    labels_session_latter = labels[-data_session_avg:]

                accuracy_former = accuracy_score(labels_session_former, predicts_session_former)
                F1score_former = f1_score(labels_session_former, predicts_session_former, average='macro')

                accuracy_latter = accuracy_score(labels_session_latter, predicts_session_latter)
                F1score_latter = f1_score(labels_session_latter, predicts_session_latter, average='macro')
                # 将计算出的均值添加到all_data_avg列表中
                all_data_avg.append([accuracy_former, F1score_former, accuracy_latter, F1score_latter])

    # 使用pandas DataFrame来存储数据
    all_data_avg_df = pd.DataFrame(all_data_avg, columns=['accuracy_former', 'F1score_former', 'accuracy_latter', 'F1score_latter'])

    # 计算all_data_avg中所有data_session_avg_former和data_session_avg_latter的均值和标准差
    acc_mean_former = all_data_avg_df['accuracy_former'].mean()
    acc_std_dev_former = all_data_avg_df['accuracy_former'].std()
    f1_mean_former = all_data_avg_df['F1score_former'].mean()
    f1_std_dev_former = all_data_avg_df['F1score_former'].std()

    acc_mean_latter = all_data_avg_df['accuracy_latter'].mean()
    acc_std_dev_latter = all_data_avg_df['accuracy_latter'].std()
    f1_mean_latter = all_data_avg_df['F1score_latter'].mean()
    f1_std_dev_latter = all_data_avg_df['F1score_latter'].std()
    

    # 使用配对t检验分析显著性
    acc_t_statistic, acc_p_value = ttest_rel(all_data_avg_df['accuracy_former'], all_data_avg_df['accuracy_latter'])
    f1_t_statistic, f1_p_value = ttest_rel(all_data_avg_df['F1score_former'], all_data_avg_df['F1score_latter'])


    # 打印显著性结果
    print(f"t-statistic: {acc_t_statistic}, p-value: {acc_p_value}")
    if acc_p_value < 0.05:
        print("The difference of MI classes in Rest in acc is statistically significant.")
    else:
        print("The difference of MI classes in Rest in acc is not statistically significant.")

    print(f"t-statistic: {f1_t_statistic}, p-value: {f1_p_value}")
    if f1_p_value < 0.05:
        print("The difference of MI classes in Rest in f1 is statistically significant.")
    else:
        print("The difference of MI classes in Rest in f1 is not statistically significant.")
    
    # 将均值、标准差和显著性结果存储在'Online_result_save_rootdir/all_data_avg_results.csv'里面
    results_df = pd.DataFrame({
        'Mean': [acc_mean_former, acc_mean_latter, f1_mean_former, f1_mean_latter],
        'Std Dev': [acc_std_dev_former, acc_std_dev_latter, f1_std_dev_former, f1_std_dev_latter],
        't-statistic': [acc_t_statistic, acc_t_statistic, f1_t_statistic, f1_t_statistic],
        'p-value': [acc_p_value, acc_p_value, f1_p_value, f1_p_value]
    }, index=['acc_avg_former', 'acc_avg_latter', 'f1_avg_former', 'f1_avg_latter'])
    results_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg_results_Rest.csv')


def Online_simulation_synthesizing_results_polynomial_avgF1_noRest_1(Online_result_save_rootdir, noRest=True, random_acc=33.0, data_session_avg=24*9, pattern=[0,0,0,0]):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []
    all_data_avg = []
    pattern=[1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0]
    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                predicts = np.loadtxt(f'{subject_folder}/class_predictions_arrays.csv', delimiter=',',skiprows=1)
                labels = np.loadtxt(f'{subject_folder}/labels_arrays.csv', delimiter=',',skiprows=1)
                if noRest:
                    # 过滤掉类别为0的情况
                    mask_former = (labels[:data_session_avg] != 0.)
                    predicts_session_former = predicts[:data_session_avg][mask_former]
                    labels_session_former = labels[:data_session_avg][mask_former]
                    
                    mask_latter = (labels[-data_session_avg:] != 0.)
                    predicts_session_latter = predicts[-data_session_avg:][mask_latter]
                    labels_session_latter = labels[-data_session_avg:][mask_latter]
                    #print("exclude the Rest class")
                else:
                    # 计算前data_session_avg个数据的均值
                    predicts_session_former = predicts[:data_session_avg]
                    labels_session_former = labels[:data_session_avg]
                    # 计算后data_session_avg个数据的均值
                    predicts_session_latter = predicts[-data_session_avg:]
                    labels_session_latter = labels[-data_session_avg:]

                accuracy_former = accuracy_score(labels_session_former, predicts_session_former)
                F1score_former = f1_score(labels_session_former, predicts_session_former, average='macro')

                accuracy_latter = accuracy_score(labels_session_latter, predicts_session_latter)
                F1score_latter = f1_score(labels_session_latter, predicts_session_latter, average='macro')
                # 将计算出的均值添加到all_data_avg列表中
                all_data_avg.append([accuracy_former, F1score_former, accuracy_latter, F1score_latter])

    # 使用pandas DataFrame来存储数据
    all_data_avg_df = pd.DataFrame(all_data_avg, columns=['accuracy_former', 'F1score_former', 'accuracy_latter', 'F1score_latter'])

    # 计算all_data_avg中所有data_session_avg_former和data_session_avg_latter的均值和标准差
    acc_mean_former = all_data_avg_df['accuracy_former'].mean()
    acc_std_dev_former = all_data_avg_df['accuracy_former'].std()
    f1_mean_former = all_data_avg_df['F1score_former'].mean()
    f1_std_dev_former = all_data_avg_df['F1score_former'].std()

    acc_mean_latter = all_data_avg_df['accuracy_latter'].mean()
    acc_std_dev_latter = all_data_avg_df['accuracy_latter'].std()
    f1_mean_latter = all_data_avg_df['F1score_latter'].mean()
    f1_std_dev_latter = all_data_avg_df['F1score_latter'].std()
    

    # 使用配对t检验分析显著性
    acc_t_statistic, acc_p_value = ttest_rel(all_data_avg_df['accuracy_former'], all_data_avg_df['accuracy_latter'])
    f1_t_statistic, f1_p_value = ttest_rel(all_data_avg_df['F1score_former'], all_data_avg_df['F1score_latter'])


    # 打印显著性结果
    print(f"t-statistic: {acc_t_statistic}, p-value: {acc_p_value}")
    if acc_p_value < 0.05:
        print("The difference of MI classes without Rest in acc is statistically significant.")
    else:
        print("The difference of MI classes without Rest in acc is not statistically significant.")

    print(f"t-statistic: {f1_t_statistic}, p-value: {f1_p_value}")
    if f1_p_value < 0.05:
        print("The difference of MI classes without Rest in f1 is statistically significant.")
    else:
        print("The difference of MI classes without Rest in f1 is not statistically significant.")
    
    # 将均值、标准差和显著性结果存储在'Online_result_save_rootdir/all_data_avg_results.csv'里面
    results_df = pd.DataFrame({
        'Mean': [acc_mean_former, acc_mean_latter, f1_mean_former, f1_mean_latter],
        'Std Dev': [acc_std_dev_former, acc_std_dev_latter, f1_std_dev_former, f1_std_dev_latter],
        't-statistic': [acc_t_statistic, acc_t_statistic, f1_t_statistic, f1_t_statistic],
        'p-value': [acc_p_value, acc_p_value, f1_p_value, f1_p_value]
    }, index=['acc_avg_former', 'acc_avg_latter', 'f1_avg_former', 'f1_avg_latter'])
    results_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg_results_noRest.csv')
    

def Online_simulation_synthesizing_results_polynomial_avgF1_1(Online_result_save_rootdir, random_acc=33.0, data_session_avg=24*9, pattern=[0,0,0,0]):
    # 初始化一个空的列表来存储所有被试的数据
    all_data = []
    all_data_avg = []
    pattern=[1, 2, 1, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0]
    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 使用numpy读取predict_accuracies.csv文件
        predicts = np.loadtxt(f'{subject_folder}/class_predictions_arrays.csv', delimiter=',',skiprows=1)
        labels = np.loadtxt(f'{subject_folder}/labels_arrays.csv', delimiter=',',skiprows=1)
        
        # 计算前data_session_avg个数据的均值
        predicts_session_former = predicts[:data_session_avg]
        labels_session_former = labels[:data_session_avg]
        # 计算后data_session_avg个数据的均值
        predicts_session_latter = predicts[-data_session_avg:]
        labels_session_latter = labels[-data_session_avg:]

        accuracy_former = accuracy_score(labels_session_former, predicts_session_former)
        F1score_former = f1_score(labels_session_former, predicts_session_former, average='macro')

        accuracy_latter = accuracy_score(labels_session_latter, predicts_session_latter)
        F1score_latter = f1_score(labels_session_latter, predicts_session_latter, average='macro')
        # 将计算出的均值添加到all_data_avg列表中
        all_data_avg.append([accuracy_former, F1score_former, accuracy_latter, F1score_latter])

    # 使用pandas DataFrame来存储数据
    all_data_avg_df = pd.DataFrame(all_data_avg, columns=['accuracy_former', 'F1score_former', 'accuracy_latter', 'F1score_latter'])

    # 计算all_data_avg中所有data_session_avg_former和data_session_avg_latter的均值和标准差
    acc_mean_former = all_data_avg_df['accuracy_former'].mean()
    acc_std_dev_former = all_data_avg_df['accuracy_former'].std()
    f1_mean_former = all_data_avg_df['F1score_former'].mean()
    f1_std_dev_former = all_data_avg_df['F1score_former'].std()

    acc_mean_latter = all_data_avg_df['accuracy_latter'].mean()
    acc_std_dev_latter = all_data_avg_df['accuracy_latter'].std()
    f1_mean_latter = all_data_avg_df['F1score_latter'].mean()
    f1_std_dev_latter = all_data_avg_df['F1score_latter'].std()
    

    # 使用配对t检验分析显著性
    acc_t_statistic, acc_p_value = ttest_rel(all_data_avg_df['accuracy_former'], all_data_avg_df['accuracy_latter'])
    f1_t_statistic, f1_p_value = ttest_rel(all_data_avg_df['F1score_former'], all_data_avg_df['F1score_latter'])


    # 打印显著性结果
    print(f"t-statistic: {acc_t_statistic}, p-value: {acc_p_value}")
    if acc_p_value < 0.05:
        print("The difference in acc is statistically significant.")
    else:
        print("The difference in acc is not statistically significant.")

    print(f"t-statistic: {f1_t_statistic}, p-value: {f1_p_value}")
    if f1_p_value < 0.05:
        print("The difference in f1 is statistically significant.")
    else:
        print("The difference in f1 is not statistically significant.")
    
    # 将均值、标准差和显著性结果存储在'Online_result_save_rootdir/all_data_avg_results.csv'里面
    results_df = pd.DataFrame({
        'Mean': [acc_mean_former, acc_mean_latter, f1_mean_former, f1_mean_latter],
        'Std Dev': [acc_std_dev_former, acc_std_dev_latter, f1_std_dev_former, f1_std_dev_latter],
        't-statistic': [acc_t_statistic, acc_t_statistic, f1_t_statistic, f1_t_statistic],
        'p-value': [acc_p_value, acc_p_value, f1_p_value, f1_p_value]
    }, index=['acc_avg_former', 'acc_avg_latter', 'f1_avg_former', 'f1_avg_latter'])
    results_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg_results.csv')
    
    all_data_avg_df.to_csv(f'{Online_result_save_rootdir}/all_data_avg.csv', index_label='sub')

def Online_simulation_synthesizing_results_2cls_linear(Online_result_save_rootdir):
    # 初始化两个空的列表来存储所有被试的数据
    all_data_1 = []
    all_data_2 = []

    # 遍历所有被试
    for i in range(1, 26):
        # 格式化被试的文件夹名
        subject_folder = f'{Online_result_save_rootdir}/0{i:02d}'

        # 在被试的文件夹中找到所有以'lr'开头的文件
        for file in os.listdir(subject_folder):
            if file.startswith('lr'):
                # 使用numpy读取predict_accuracies.csv文件
                data = np.loadtxt(f'{subject_folder}/{file}/predict_accuracies.csv', delimiter=',',skiprows=1)

                # 将数据添加到all_data列表中
                all_data_1.append(np.concatenate((data[1:30], data[61:90])))
                all_data_2.append(np.concatenate((data[31:60], data[91:120])))

    # 使用pandas DataFrame来存储数据
    all_data_df_1 = pd.DataFrame(np.column_stack(all_data_1))
    all_data_df_2 = pd.DataFrame(np.column_stack(all_data_2))

    # 计算每一行的均值和标准差
    means_1 = all_data_df_1.mean(axis=1)
    std_devs_1 = all_data_df_1.std(axis=1)
    means_2 = all_data_df_2.mean(axis=1)
    std_devs_2 = all_data_df_2.std(axis=1)

    # 将all_data_df，means和std_devs存储到一个CSV文件中
    all_data_df_1.to_csv(f'{Online_result_save_rootdir}/all_data_1.csv', index_label='Iteration')
    means_1.to_csv(f'{Online_result_save_rootdir}/means_1.csv', header=['Mean'])
    std_devs_1.to_csv(f'{Online_result_save_rootdir}/std_devs_1.csv', header=['Std Dev'])
    all_data_df_2.to_csv(f'{Online_result_save_rootdir}/all_data_2.csv', index_label='Iteration')
    means_2.to_csv(f'{Online_result_save_rootdir}/means_2.csv', header=['Mean'])
    std_devs_2.to_csv(f'{Online_result_save_rootdir}/std_devs_2.csv', header=['Std Dev'])

    # 使用最小二乘法拟合一次函数
    popt_1, pcov_1 = curve_fit(linear_func, means_1.index, means_1)
    popt_2, pcov_2 = curve_fit(linear_func, means_2.index, means_2)

    # 创建一个新的figure
    fig, axs = plt.subplots(2, figsize=(16, 11))

    # 添加标题和轴标签
    axs[0].set_title('Average accuracy every iteration (0-30 and 60-90)')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Accuracy(%)')

    # 绘制均值折线图
    sns.lineplot(data=means_1, label='Mean', ax=axs[0])

    # 绘制拟合的一次函数
    axs[0].plot(means_1.index, linear_func(means_1.index, *popt_1), 'r-', label=f'fit: a={popt_1[0]:.3f}, b={popt_1[1]:.3f}')

    # 添加标准差
    axs[0].fill_between(range(len(means_1)), means_1-std_devs_1 , means_1+std_devs_1 , color='b', alpha=.1)

    # 添加图例
    axs[0].legend()

    # 添加标题和轴标签
    axs[1].set_title('Average accuracy every iteration (30-60 and 90-120)')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Accuracy(%)')

    # 绘制均值折线图
    sns.lineplot(data=means_2, label='Mean', ax=axs[1])

    # 绘制拟合的一次函数
    axs[1].plot(means_2.index, linear_func(means_2.index, *popt_2), 'r-', label=f'fit: a={popt_2[0]:.3f}, b={popt_2[1]:.3f}')

    # 添加标准差
    axs[1].fill_between(range(len(means_2)), means_2-std_devs_2 , means_2+std_devs_2 , color='b', alpha=.1)

    # 添加图例
    axs[1].legend()

    # 保存图形到data_folder文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_2cls.png')
    plt.close()

def Online_simulation_synthesizing_results_linear_perclass(Online_result_save_rootdir, random_acc=0.33):
    # 初始化一个列表来存储所有被试的数据
    all_subjects_data = []
    #_all_subjects_data_test = []

    # 遍历每一个被试的文件夹
    for i in range(1, 26):
        subject_dir = os.path.join(Online_result_save_rootdir, f'0{i:02}')
        
        # 寻找以"lr"开头的文件夹
        for folder in os.listdir(subject_dir):
            if folder.startswith('lr'):
                csv_file = os.path.join(subject_dir, folder, 'predict_perclass_accuracies.csv')

                # 读取.csv文件并添加到列表中
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, header=None)
                    all_subjects_data.append(df)
                    #_all_subjects_data_test.append(df.iloc[5])

    # 计算均值和标准差
    """
    _all_subjects_data_test_df = pd.DataFrame(_all_subjects_data_test)
    _mean_values = _all_subjects_data_test_df.mean()
    _std_values = _all_subjects_data_test_df.std()
    """
    mean_df = pd.concat(all_subjects_data).groupby(level=0).mean()
    std_df = pd.concat(all_subjects_data).groupby(level=0).std()

    # 创建一个新的figure
    sns.set()
    for i in range(mean_df.shape[1]):
        # 使用最小二乘法拟合一次函数
        #popt, pcov = curve_fit(linear_func, mean_df[i].index, mean_df[i])
        #plt.plot(mean_df[i].index, linear_func(mean_df[i].index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}')
        sns.lineplot(data=mean_df[i], label=f'Class {i}')
        plt.fill_between(range(len(mean_df[i])), mean_df[i]-std_df[i], mean_df[i]+std_df[i], alpha=0.2)
    
    plt.axhline(y=random_acc, color='g', linestyle='--', label=f'Random accuracy: {random_acc}')

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    
    plt.title('Mean Accuracy per Class over Iterations with Standard Deviation')
    plt.legend()
    plt.savefig(os.path.join(Online_result_save_rootdir, 'mean_accuracies_perclass_iterations_plot.png'))
    plt.close()
    
    # 保存均值和标准差
    mean_df.to_csv(os.path.join(Online_result_save_rootdir, 'mean_perclass.csv'), index=False)
    std_df.to_csv(os.path.join(Online_result_save_rootdir, 'std_perclass.csv'), index=False)

def Online_simulation_synthesizing_results_linear_perclass_1(Online_result_save_rootdir, random_acc=0.33):
    # 初始化一个列表来存储所有被试的数据
    all_subjects_data = []

    # 遍历每一个被试的文件夹
    for i in range(1, 26):
        subject_dir = os.path.join(Online_result_save_rootdir, f'0{i:02}')
        
        
        csv_file = os.path.join(subject_dir, 'predict_perclass_accuracies.csv')

        # 读取.csv文件并添加到列表中
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, header=None)
            all_subjects_data.append(df)

    # 计算均值和标准差
    mean_df = pd.concat(all_subjects_data).groupby(level=0).mean()
    std_df = pd.concat(all_subjects_data).groupby(level=0).std()

    # 创建一个新的figure
    sns.set()
    for i in range(mean_df.shape[1]):
        # 使用最小二乘法拟合一次函数
        #popt, pcov = curve_fit(linear_func, mean_df[i].index, mean_df[i])
        #plt.plot(mean_df[i].index, linear_func(mean_df[i].index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}')
        sns.lineplot(data=mean_df[i], label=f'Class {i}')
        plt.fill_between(range(len(mean_df[i])), mean_df[i]-std_df[i], mean_df[i]+std_df[i], alpha=0.2)
    
    plt.axhline(y=random_acc, color='g', linestyle='--', label=f'Random accuracy: {random_acc}')

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    
    plt.title('Mean Accuracy per Class over Iterations with Standard Deviation')
    plt.legend()
    plt.savefig(os.path.join(Online_result_save_rootdir, 'mean_accuracies_perclass_iterations_plot.png'))
    plt.close()
    
    # 保存均值和标准差
    mean_df.to_csv(os.path.join(Online_result_save_rootdir, 'mean_perclass.csv'), index=False)
    std_df.to_csv(os.path.join(Online_result_save_rootdir, 'std_perclass.csv'), index=False)


def Online_simulation_synthesizing_results_comparison_linear(Online_result_save_rootdir, methods):
    
    # 设置图形大小
    plt.figure(figsize=(16, 9))

    # 设置seaborn样式
    sns.set()

    # 遍历所有的方法
    for method in methods:
        # 读取means.csv和std_devs.csv文件
        means = pd.read_csv(f'{Online_result_save_rootdir}/{method}/means.csv', index_col=0)
        std_devs = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_devs.csv', index_col=0)

        # 使用最小二乘法拟合一次函数
        popt, pcov = curve_fit(linear_func, means.index, means['Mean'])

        # 绘制均值折线图
        sns.lineplot(data=means['Mean'], label=f'{method} Mean')

        # 绘制拟合的一次函数
        plt.plot(means.index, linear_func(means.index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}')

        # 添加标准差
        plt.fill_between(range(len(means)), means['Mean']-std_devs['Std Dev'], means['Mean']+std_devs['Std Dev'], alpha=.1)

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration methods comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 添加图例
    plt.legend()

    # 保存图形到Online_result_save_rootdir文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_comparison.png')


def Online_simulation_synthesizing_results_comparison_linear_2cls(Online_result_save_rootdir, methods):
    # 创建一个新的figure
    fig, axs = plt.subplots(2, figsize=(16, 16))

    # 设置seaborn样式
    sns.set()

    # 遍历所有的方法
    for method in methods:
        # 读取means_1.csv和std_devs_1.csv文件
        means_1 = pd.read_csv(f'{Online_result_save_rootdir}/{method}/means_1.csv', index_col=0)
        std_devs_1 = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_devs_1.csv', index_col=0)

        # 使用最小二乘法拟合一次函数
        popt_1, pcov_1 = curve_fit(linear_func, means_1.index, means_1['Mean'])

        # 绘制均值折线图
        sns.lineplot(data=means_1['Mean'], label=f'{method} Mean', ax=axs[0])

        # 绘制拟合的一次函数
        axs[0].plot(means_1.index, linear_func(means_1.index, *popt_1), 'r-', label=f'fit: a={popt_1[0]:.3f}, b={popt_1[1]:.3f}')

        # 添加标准差
        axs[0].fill_between(range(len(means_1)), means_1['Mean']-std_devs_1['Std Dev'], means_1['Mean']+std_devs_1['Std Dev'], alpha=.1)

        # 读取means_2.csv和std_devs_2.csv文件
        means_2 = pd.read_csv(f'{Online_result_save_rootdir}/{method}/means_2.csv', index_col=0)
        std_devs_2 = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_devs_2.csv', index_col=0)

        # 使用最小二乘法拟合一次函数
        popt_2, pcov_2 = curve_fit(linear_func, means_2.index, means_2['Mean'])

        # 绘制均值折线图
        sns.lineplot(data=means_2['Mean'], label=f'{method} Mean', ax=axs[1])

        # 绘制拟合的一次函数
        axs[1].plot(means_2.index, linear_func(means_2.index, *popt_2), 'r-', label=f'fit: a={popt_2[0]:.3f}, b={popt_2[1]:.3f}')

        # 添加标准差
        axs[1].fill_between(range(len(means_2)), means_2['Mean']-std_devs_2['Std Dev'], means_2['Mean']+std_devs_2['Std Dev'], alpha=.1)

    # 添加标题和轴标签
    axs[0].set_title('Average accuracy every iteration methods comparison (1)')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Accuracy(%)')

    axs[1].set_title('Average accuracy every iteration methods comparison (2)')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Accuracy(%)')

    # 添加图例
    axs[0].legend()
    axs[1].legend()

    # 保存图形到Online_result_save_rootdir文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_comparison_cls2.png')
    plt.close()

def Online_simulation_synthesizing_results_comparison_polynomial(Online_result_save_rootdir, methods, random_acc=33.0):
    
    # 设置图形大小
    plt.figure(figsize=(16, 9))

    # 设置seaborn样式
    sns.set()

    # 遍历所有的方法
    for method in methods:
        # 读取means.csv和std_devs.csv文件
        means = pd.read_csv(f'{Online_result_save_rootdir}/{method}/means.csv', index_col=0)
        std_devs = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_devs.csv', index_col=0)

        # 对均值进行滑动窗口平均以进行平滑处理
        means_smooth = means.rolling(window=8, min_periods=1).mean()

        # 使用最小二乘法拟合3次多项式函数
        popt, pcov = curve_fit(polynomial_func, means_smooth.index, means_smooth['Mean'])

        # 绘制均值折线图
        sns.lineplot(data=means_smooth['Mean'], label=f'{method} Mean', linewidth=4)

        # 绘制拟合的3次多项式函数
        #plt.plot(means_smooth.index, polynomial_func(means_smooth.index, *popt), 'r-', label=f'fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, d={popt[3]:.3f}', linewidth=2)

        # 添加标准差
        plt.fill_between(range(len(means_smooth)), means_smooth['Mean']-std_devs['Std Dev'], means_smooth['Mean']+std_devs['Std Dev'], alpha=.1)

    # 添加y=random_acc的虚线
    plt.axhline(y=random_acc, color='g', linestyle='--', label=f'Random accuracy: {random_acc}')

    # 添加标题和轴标签
    plt.title('Average accuracy every iteration methods comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy(%)')

    # 添加图例
    plt.legend()

    # 保存图形到Online_result_save_rootdir文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_comparison_polynomial.png')
    plt.close()


def Online_simulation_synthesizing_results_comparison_polynomial_optimized(Online_result_save_rootdir, methods, std_weight=0.40, random_acc=33.0, lower=25.0, upper=60.0, retrain_method='Model retraining'):
    
    matplotlib.rcParams["font.family"] = "Times New Roman"
    # 设置图形大小
    plt.figure(figsize=(16, 9))

    # 设置seaborn样式
    sns.set_theme()

    # 遍历所有的方法
    for method in methods:
        # 读取means.csv和std_devs.csv文件
        means = pd.read_csv(f'{Online_result_save_rootdir}/{method}/means.csv', index_col=0)
        std_devs = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_devs.csv', index_col=0)

        # 对均值进行滑动窗口平均以进行平滑处理
        means_smooth = means.rolling(window=24, min_periods=1).mean()

        # 使用最小二乘法拟合3次多项式函数
        popt, pcov = curve_fit(polynomial_func, means_smooth.index, means_smooth['Mean'])
        
        method_name, method_color = name_change(method)
        # 绘制均值折线图
        if method_name == retrain_method:
            # sns.lineplot(data=means_smooth['Mean'], label=f'{method_name}', linewidth=2, linestyle='--', color=method_color)
            sns.lineplot(data=means_smooth['Mean'], label=f'{method_name}', linewidth=4, color=method_color)
        
        else:
            means_sm = means_smooth['Mean']
            sns.lineplot(data=means_smooth['Mean'], label=f'{method_name}', linewidth=4, color=method_color)
        
        # 添加标准差
        plt.fill_between(range(len(means_smooth)), means_smooth['Mean']-std_weight*std_devs['Std Dev'], means_smooth['Mean']+std_weight*std_devs['Std Dev'], alpha=.05, color=method_color)

    # 添加y=random_acc的虚线
    plt.axhline(y=random_acc, color='g', linestyle='--', label=f'Random accuracy: {random_acc}')
    # 创建一个字体对象，调整字体用
    font_prop = fm.FontProperties(family='Times New Roman', size=18)

    # 设置y轴的最小值为random_acc
    plt.ylim(bottom=lower, top=upper)
    
    # 设置x轴的范围为0到96
    plt.xlim(left=0, right=95)
    # 设置刻度的字体
    # 获取当前的坐标轴
    ax = plt.gca()
    # 设置x轴和y轴的刻度字体
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 添加标题和轴标签
    font_prop_label = fm.FontProperties(family='Times New Roman', size=20)
    #plt.title('Average accuracy every iteration')
    plt.xlabel('Trials', fontproperties=font_prop_label)
    plt.ylabel('Accuracy(%)', fontproperties=font_prop_label)
    #plt.xlabel('Trials', fontdict={'fontname':'Times New Roman', 'fontsize':18})
    #plt.ylabel('Accuracy(%)', fontdict={'fontname':'Times New Roman', 'fontsize':18})

    # 添加图例
    plt.legend(loc='upper left', prop=font_prop)

    # 减小白边
    plt.tight_layout()

    # 调整下右边距
    plt.subplots_adjust(right=0.95)

    # 保存图形到Online_result_save_rootdir文件夹
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_comparison_polynomial_optimized.png')
    plt.savefig(f'{Online_result_save_rootdir}/synthesizing_results_subjects_comparison_polynomial_optimized.pdf')
    plt.close()

def name_change(method, methods = ['baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_2', 'method5_encoder3_pretrainlight_baseline_1_9batchsize_Rest_2_mixed_3', 'method5_encoder3_pretrainlight_baseline_2_4_9batchsize_Rest_2_mixed_3', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_4', 'method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_8_mixed_4','method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1','method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_3']
        ):
    if method=='baseline1_encoder3_noupdate_noRest_val_6_9batchsize_Rest_mixed_2' or method=='baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2' or method=='baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new' or method=='baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new_150' or method=='baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new_1' or method=='baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new_2' or method=='baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new_3' or method=='baseline1_EEGNet_noupdate_noRest_val_6_9batchsize_Rest_mixed_2_new_4':
        return 'No updating','#386fb7'
    if method=='method5_encoder3_pretrainlight_baseline_1_9batchsize_Rest_2_mixed_3' or method=='method5_EEGNet_baseline_1_9batchsize_Rest_2_mixed_3_new' or method=='method5_EEGNet_baseline_1_1_9batchsize_Rest_2_mixed_3_new_2' or method=='method5_EEGNet_baseline_1_2_9batchsize_Rest_2_mixed_3_new_2' or method=='method5_EEGNet_baseline_1_2_9batchsize_Rest_2_mixed_3_new_1' or method=='method5_EEGNet_baseline_1_2_9batchsize_Rest_2_mixed_3_new_3' or method=='method5_EEGNet_baseline_1_3_9batchsize_Rest_2_mixed_3_new_1' or method=='method5_EEGNet_baseline_1_2_9batchsize_Rest_2_mixed_3_new_4':
        return "Lin's",'#E0802A'
    if method=='method5_encoder3_pretrainlight_baseline_2_4_9batchsize_Rest_2_mixed_3' or method=='method5_encoder3_pretrainlight_baseline_2_6_9batchsize_Rest_2_mixed_3' or method=='method5_encoder3_pretrainlight_baseline_2_7_9batchsize_Rest_2_mixed_3' or method=='method5_EEGNet_baseline_2_7_9batchsize_Rest_2_mixed_3_new' or method=='method5_EEGNet_baseline_2_8_9batchsize_Rest_2_mixed_3_new_2' or method=='method5_EEGNet_baseline_2_8_9batchsize_Rest_2_mixed_3_new_1':
        return "Wang's",'#3b9144'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_4':
        return  r'$L_{ce}+L_{kdr}+L_{focal}$', '#AF5A76'
        #return  'Lce+Lkdr+Lfocal', '#AF5A76'
    if method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_3_ablation3':
        return "Ablation 3",'#E0802A'
    if method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_3_ablation2':
        return "Ablation 2",'#3b9144'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_4' \
        or method=='method4_encoder3_pretrainlight_fixedepoch_Distillation_val_15_9batchsize_Rest_2_lessepoch_mixed' \
            or method=='method4_encoder3_pretrainlight_fixedepoch_Distillation_val_15_9batchsize_Rest_2_lessepoch_1_mixed' \
                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new' \
                    or method == 'method4_EEGNet_fixedepoch_FeatureDistillation_val_16_9batchsize_Rest_2_lessepoch_1_4_mixed_4_new' \
                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_16_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new' \
                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_5_mixed_4_new'\
                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_16_9batchsize_Rest_2_2_2_mixed_4_new'\
                                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_20_9batchsize_Rest_2_1_1_mixed_4_new'\
                                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_17_9batchsize_Rest_2_lessepoch_1_1_mixed_4_new'\
                                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_20_9batchsize_Rest_2_2_1_mixed_4_new'\
                                                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new_150'\
                                                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_1_2_mixed_4_new_150'\
                                                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new_1'\
                                                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_19_9batchsize_Rest_2_2_1_mixed_4_new_150'\
                                                                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new_2'\
                                                                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new_4'\
                                                                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new_5'\
                                                                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new'\
                                                                                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_1_1_mixed_4_new'\
                                                                                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_3_mixed_4_new'\
                                                                                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_6_mixed_4_new'\
                                                                                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_5_mixed_4_new'\
                                                                                                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_3_mixed_4_new'\
                                                                                                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_4_mixed_4_new'\
                                                                                                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_7_mixed_4_new'\
                                                                                                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_4_mixed_4_new_2'\
                                                                                                                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new'\
                                                                                                                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_1'\
                                                                                                                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_1'\
                                                                                                                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_2'\
                                                                                                                                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_3'\
                                                                                                                                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_21_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_3_1':
        return "Ours (all)",'#C23D3E'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_8_mixed_4' \
        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new'\
                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_19_9batchsize_Rest_2_2_1_mixed_4_new'\
                        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_13_9batchsize_Rest_2_lessepoch_1_2_mixed_4_new_distilmodified'\
                            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_22_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_2'\
                                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_22_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_2_1'\
                                    or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_23_9batchsize_Rest_2_lessepoch_1_8_mixed_7_new_3_2_1':
        return "Ours (incremental)",'#9273B2'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1' or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1_new':
        return "Model retraining",'#3852A4'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_3' or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_new':
        return "Abu-Rmileh's",'#838482'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_lessdata':
        return "Ours (less data)", '#AF5A76'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_3_scratch_focal' or method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_6_mixed_3_scratch':
        return "Ours (scratch)", '#AF5A76'
    if method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_1_mixed_4'\
        or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_18_9batchsize_Rest_2_1_1_mixed_4_new_1'\
            or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_19_9batchsize_Rest_2_1_1_mixed_4_new'\
                or method=='method4_EEGNet_fixedepoch_FeatureDistillation_val_17_9batchsize_Rest_2_1_1_mixed_4_new_1':
        return "Ours (EEGNet)", '#AF5A76'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_4_1':
        return r'$L_{ce}+L_{kdr}$', '#AF5A76'
        #return 'Lce+Lkdr', '#AF5A76'
    if method=='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_14_9batchsize_Rest_2_lessepoch_1_mixed_ablation_5_1':
        return r'$L_{ce}$', '#AF5A76'
        #return 'Lce', '#AF5A76'
    
    


def Online_simulation_synthesizing_results_comparison_polynomial_optimized_perclass(Online_result_save_rootdir, methods, std_weight=0.40, random_acc=33.0, lower=30.0, upper=60.0, \
                                                                                    retrain_method='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1', \
                                                                                        row=2, col=3, img_name='synthesizing_results_subjects_comparison_polynomial_optimized_perclass'):
    
    matplotlib.rcParams["font.family"] = "Times New Roman"
    # 设置图形大小
    fig, axs = plt.subplots(row, col, figsize=(16, 9))

    # 设置seaborn样式
    sns.set_theme()
    
    class_colors = ['#386fb7', '#3b9144', '#E0802A']
    # 遍历所有的方法
    for i, method in enumerate(methods):
        # 读取mean_perclass.csv和std_perclass.csv文件
        means = pd.read_csv(f'{Online_result_save_rootdir}/{method}/mean_perclass.csv', header=None)
        std_devs = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_perclass.csv', header=None)

        # 只保留第24个到第84个数据点，由于原数据是0-1区间的，所以乘上100转换成百分比
        means = means.values[1:,:] * 100
        std_devs = std_devs.values[1:,:] * 100

        # 绘制均值折线图
        method_name, method_color = name_change(method)
        for class_label, class_color in zip([0, 1, 2],class_colors):
            sns.lineplot(x=np.arange(means.shape[0]), y=means[:,int(class_label)], ax=axs[i//3, i%3], linewidth=4, color=class_color)
            #axs[i//3, i%3].plot(np.arange(means.shape[0]), means[:, int(class_label)], label=f'{method_name} class {int(class_label)}', linewidth=4)
            # 添加标准差
            axs[i//3, i%3].fill_between(np.arange(means.shape[0]), means[:,int(class_label)]-std_weight*std_devs[:, int(class_label)], means[:, int(class_label)]+std_weight*std_devs[:, int(class_label)], color=class_color, alpha=.075)

        # 添加y=random_acc的虚线
        axs[i//3, i%3].axhline(y=random_acc, color='g', linestyle='--')
        # 创建一个字体对象，调整字体用
        font_prop = fm.FontProperties(family='Times New Roman', size=16)

        # 设置y轴的最小值为random_acc
        if method_name=="Lin's" or method_name=="Wang's":    
            axs[i//3, i%3].set_ylim(bottom=lower, top=upper)
        else:
            axs[i//3, i%3].set_ylim(bottom=lower, top=upper)
    
        # 设置x轴的范围为24到85
        axs[i//3, i%3].set_xlim(left=24, right=85)
        
        """# 设置刻度的字体
        for label in axs[i//3, i%3].get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in axs[i//3, i%3].get_yticklabels():
            label.set_fontproperties(font_prop)
        """
        # 添加标题和轴标签
        font_prop_label = fm.FontProperties(family='Times New Roman', size=16)
        font_prop_title = fm.FontProperties(family='Times New Roman', size=18, weight='bold')
        axs[i//3, i%3].set_xlabel('Trials', fontproperties=font_prop_label)
        axs[i//3, i%3].set_ylabel('Accuracy(%)', fontproperties=font_prop_label)
        axs[i//3, i%3].set_title(method_name, fontproperties=font_prop_title)

        # 添加图例
        #axs[i//3, i%3].legend(loc='upper left', prop=font_prop)

    # 在绘制完所有子图之后，创建一个统一的图例
    # 你需要提供图例的句柄和标签
    handles, labels = [], []
    for class_label, class_color in zip([0, 1, 2], class_colors):
        handles.append(plt.Line2D([], [], color=class_color, linewidth=4))  # 假设颜色和线宽与实际绘制的线条相同
        labels.append(f'Class {int(class_label)}')
    handles.append(plt.Line2D([], [], color='g', linestyle='--'))
    labels.append(f'Random accuracy: {random_acc}')

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), prop=font_prop)
    # 减小白边
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)  # 可能需要根据实际情况调整这个值
    # 调整下右边距
    plt.subplots_adjust(right=0.95)

    # 保存图形到Online_result_save_rootdir文件夹
    plt.savefig(f'{Online_result_save_rootdir}/{img_name}.png')
    plt.savefig(f'{Online_result_save_rootdir}/{img_name}.pdf')
    plt.close()  

def Online_simulation_synthesizing_results_comparison_polynomial_optimized_perclass_ablation(Online_result_save_rootdir, methods, std_weight=0.40, random_acc=33.0, lower=30.0, upper=60.0, \
                                                                                    retrain_method='method4_encoder3_pretrainlight_fixedepoch_FeatureDistillation_val_9batchsize_Rest_2_mixed_retrain_1', \
                                                                                        row=1, col=3, img_name='synthesizing_results_subjects_comparison_polynomial_optimized_perclass'):
    
    matplotlib.rcParams["font.family"] = "Times New Roman"
    # 设置图形大小
    fig, axs = plt.subplots(row, col, figsize=(16, 5))

    # 设置seaborn样式
    sns.set_theme()
    
    class_colors = ['#386fb7', '#3b9144', '#E0802A']
    # 遍历所有的方法
    for i, method in enumerate(methods):
        # 读取mean_perclass.csv和std_perclass.csv文件
        means = pd.read_csv(f'{Online_result_save_rootdir}/{method}/mean_perclass.csv', header=None)
        std_devs = pd.read_csv(f'{Online_result_save_rootdir}/{method}/std_perclass.csv', header=None)

        # 只保留第24个到第84个数据点，由于原数据是0-1区间的，所以乘上100转换成百分比
        means = means.values * 100
        std_devs = std_devs.values * 100

        # 绘制均值折线图
        method_name, method_color = name_change(method)
        for class_label, class_color in zip([0, 1, 2],class_colors):
            sns.lineplot(x=np.arange(means.shape[0]), y=means[:,int(class_label)], ax=axs[i%3], linewidth=4, color=class_color)
            #axs[i//3, i%3].plot(np.arange(means.shape[0]), means[:, int(class_label)], label=f'{method_name} class {int(class_label)}', linewidth=4)
            # 添加标准差
            axs[i%3].fill_between(np.arange(means.shape[0]), means[:,int(class_label)]-std_weight*std_devs[:, int(class_label)], means[:, int(class_label)]+std_weight*std_devs[:, int(class_label)], color=class_color, alpha=.075)

        # 添加y=random_acc的虚线
        axs[i%3].axhline(y=random_acc, color='g', linestyle='--')
        # 创建一个字体对象，调整字体用
        font_prop = fm.FontProperties(family='Times New Roman', size=16)

        # 设置y轴的最小值为random_acc
        if method_name==r'$L_{ce}$' or method_name=="Wang's":    
            axs[i%3].set_ylim(bottom=10, top=45)
        else:
            axs[i%3].set_ylim(bottom=lower, top=upper)
    
        # 设置x轴的范围为24到85
        axs[i%3].set_xlim(left=24, right=85)
        
        """# 设置刻度的字体
        for label in axs[i//3, i%3].get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in axs[i//3, i%3].get_yticklabels():
            label.set_fontproperties(font_prop)
        """
        # 添加标题和轴标签
        font_prop_label = fm.FontProperties(family='Times New Roman', size=16)
        font_prop_title = fm.FontProperties(family='Times New Roman', size=18, weight='bold')
        axs[i%3].set_xlabel('Trials', fontproperties=font_prop_label)
        axs[i%3].set_ylabel('Accuracy(%)', fontproperties=font_prop_label)
        axs[i%3].set_title(method_name, fontproperties=font_prop_title)

        # 添加图例
        #axs[i//3, i%3].legend(loc='upper left', prop=font_prop)

    # 在绘制完所有子图之后，创建一个统一的图例
    # 你需要提供图例的句柄和标签
    handles, labels = [], []
    for class_label, class_color in zip([0, 1, 2], class_colors):
        handles.append(plt.Line2D([], [], color=class_color, linewidth=4))  # 假设颜色和线宽与实际绘制的线条相同
        labels.append(f'Class {int(class_label)}')
    handles.append(plt.Line2D([], [], color='g', linestyle='--'))
    labels.append(f'Random accuracy: {random_acc}')

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), prop=font_prop)
    # 减小白边
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.23)  # 可能需要根据实际情况调整这个值
    # 调整下右边距
    plt.subplots_adjust(right=0.98)
    
    # 保存图形到Online_result_save_rootdir文件夹
    plt.savefig(f'{Online_result_save_rootdir}/{img_name}.png')
    plt.savefig(f'{Online_result_save_rootdir}/{img_name}.pdf')
    plt.close()  