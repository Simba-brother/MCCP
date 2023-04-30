from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections import defaultdict
import os
import random

# 设置训练显卡
os.environ['CUDA_VISIBLE_DEVICES']='2'

def generate_generator_multiple(batches_A, batches_B):
    '''
    将连个模型的输入bath 同时返回
    '''
    # simulation

    # sports
    # genX1=generator_left.flow_from_dataframe(data_frame, x_col='filepaths', y_col='labels', target_size=(224,224), class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)   
    #                                                                                                             # weather:rgb  150, 150

    # genX2=generator_right.flow_from_dataframe(data_frame, x_col='filepaths', y_col='labels', target_size=(224,224), class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)  # weather:rgb 200, 400

    # weather
    # genX1=generator_left.flow_from_dataframe(data_frame, x_col='file_path', y_col='label', target_size=target_size, class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)   
                                                                                                                

    # genX2=generator_right.flow_from_dataframe(data_frame, x_col='file_path', y_col='label', target_size=target_size, class_mode='categorical',
    #                                 color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size)  
    while True:
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def getOutput():
    # 加载 合并模型
    model_path = "/data/mml/overlap_v2_datasets/weather/merged_model/combination_model_inheritWeights.h5"
    combination_model = load_model(model_path)
    hidden_model = keras.Model(inputs = combination_model.input, outputs = combination_model.get_layer(index = -2).output) 
    
    # 加载 合并的训练集csv
    merged_csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/train/merged_df.csv"
    merged_df = pd.read_csv(merged_csv_path)

    # 加载评估集
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv"
    df = pd.read_csv(csv_path) 

    test_gen_left = ImageDataGenerator()
    test_gen_right = ImageDataGenerator(rescale=1./255)

    batch_size = 32
    # 全局类别,字典序列
    classes = merged_df["label"].unique()
    classes = np.sort(classes).tolist()

    target_size_A = (100,100)
    target_size_B = (256,256)

    test_batches_A = test_gen_left.flow_from_dataframe(df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_A, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)
                                                                                                                # weather:rgb  150, 150

    test_batches_B = test_gen_right.flow_from_dataframe(df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_B, class_mode='categorical',
                                                color_mode='rgb', classes = classes, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)

    batches = generate_generator_multiple(test_batches_A, test_batches_B)
    last_hidden_array = hidden_model.predict_generator(batches, steps= test_batches_A.n / batch_size, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    print("getOutput() success")
    return last_hidden_array

def output_to_interval(one_neuron_output, interval):
    # 每个区间上样本数量
    num = []
    for i in range(interval.shape[0] - 1):
        # 遍历每个区间
        isIn_array = np.logical_and(one_neuron_output > interval[i], one_neuron_output < interval[i + 1])
        # 在该区间样本数
        count = np.sum(isIn_array)
        num.append(count)
    return np.array(num)

def get_neuron_interval_proba(output, divide = 5):
    '''
    得到每个神经元的区间和概率分布
    output: 所有样本在last_hidden_layer上的输出
    '''
    # 样本总数
    total_num = output.shape[0]
    # init dict and its input
    neuron_interval = defaultdict(np.array)
    neuron_proba = defaultdict(np.array)

    lower_bound = np.min(output, axis=0)
    upper_bound = np.max(output, axis=0)

    for neuron_index in range(output.shape[-1]):
        interval = np.linspace(lower_bound[neuron_index], upper_bound[neuron_index], divide)
        # 每个神经元的区间
        neuron_interval[neuron_index] = interval
        # 每个神经元的概率分布
        neuron_proba[neuron_index] = output_to_interval(
            output[:, neuron_index], interval) / total_num

    return neuron_interval, neuron_proba

def neuron_entropy(neuron_interval, neuron_proba, sample_index_array, output):
    '''
    function:
        计算该采样子集在所有神经元的熵
    args:
        neuron_interval: 所有神经元的字典
        neuron_proba: 所有神经元概率
        sample_index_array: 子集的逻辑index
        output: 所有样本在last_hidden_layer的输出
    '''
    # 已经采样的数量
    sampled_num = sample_index_array.shape[0]
    if(sampled_num == 0):
        return -1e3
    # 用于存储该采样子集在所有神经元上的熵
    neuron_entropy = []
    # 获得被采样样本的输出
    output = output[sample_index_array, :]
    for neuron_index in range(output.shape[-1]):
        # 该神经元的区间
        interval = neuron_interval[neuron_index]
        # 总池子样本在该神经元的概率分布
        bench_proba = neuron_proba[neuron_index]
        # 被采样样本们在该神经元概率分布
        test_proba = output_to_interval(output[:, neuron_index], interval) / sampled_num
        # 为概率分布设置上下界
        test_proba = np.clip(test_proba, 1e-10, 1 - 1e-10)
        # 概率分布取log
        log_proba = np.log(test_proba)
        # 获得总概率分布
        temp_proba = bench_proba.copy()
        # 将某些总概率分布设置为0
        temp_proba[temp_proba < (.5 / sampled_num)] = 0
        # 计算该神经元熵
        entropy = np.sum(log_proba * temp_proba)
        neuron_entropy.append(entropy)
    return np.array(neuron_entropy)

def average(entropy):
    return np.mean(entropy)

def sampling(batch_size, test_df, neuron_interval, neuron_proba, output, sample_size):
    '''
    开始采样
    args:
        batch_size: 每次选择一批样本次的size eg: 5
        test_df: 侯选池
        neuron_interval: 神经元的区间
        neuron_proba: 神经元区间的概率统计分布
        output: 所有样本在last_hidden_layer的输出
        sample_size: 采样数量 eg:30,60,90,120,150,180

    '''
    # 每一轮最终会确定某个批次进入 根据采样数和批次大小确定step
    step = int((sample_size - 30)/batch_size)
    # 采样池样本总数
    total_num = test_df.shape[0]

    # 随机从采样池子中采出30个,里面是element is merged_idx
    selected_index_array = np.random.choice(range(total_num),replace=False,size=30)
    for i in range(step):  # 每一步会在selected_index_list 添加 一个 batch
        print("第{}步".format(i))
        # 打乱样本池子
        arr = np.random.permutation(total_num)
        # 该步下通过30次迭代搜索，确定出一个batch
        max_iter = 30
        # selected_index_array 在每个神经元上的熵
        entropy_array = neuron_entropy(neuron_interval, neuron_proba, selected_index_array, output)
        # 算个平均，作为selected_index_array 在 该hidden_layer上的熵
        avg = average(entropy_array)
        # 设置一个最大值变量
        max_avg = avg

        candidates_avg_list = []
        condidates_index_list = []     
        # select
        for j in range(max_iter):
            print("第{}步，第{}次迭代".format(i,j))
            # 剩下的
            diff_array = np.setdiff1d(arr, selected_index_array)
            candidate = np.array(random.sample(list(diff_array), batch_size))
            # 随机拿出一批次候选者index，加到selected面一起计算 采样熵数组
            candidate_index_array = np.append(selected_index_array, candidate)
            cur_entropy_array = neuron_entropy(neuron_interval, neuron_proba, candidate_index_array, output)
            cur_avg = average(cur_entropy_array)
            # 将该批次数据index记录下来到list中
            condidates_index_list.append(candidate)
            # 将该批次数据的层均熵记录下来到list中
            candidates_avg_list.append(cur_avg)
        # 看看哪个批次的均熵最大，并取出
        max_avg = np.max(candidates_avg_list) # 最大层均熵
        local = np.argmax(candidates_avg_list) # 最大层均熵的批次位置
        max_index_array = condidates_index_list[local] # 最大层均熵的批次index
        if(max_avg <= avg):
            # 如果小的化 随机选择。等于该步骤是随机选择了一个batch
            max_index_array = np.array(random.sample(list(diff_array), batch_size))
            # max_index_array = np.random.choice(range(total_num),replace=False,size=batch_size)
        selected_index_array = np.append(selected_index_array, max_index_array)
    return selected_index_array

def start_sampling():
    test_df = pd.read_csv("/data/mml/overlap_v2_datasets/sport/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
    repeat_num = 10 
    output = np.load("exp_data/sport/sampling/num/ces/last_hidden_layer_mergedTest_output/output.npy")
    neuron_interval, neuron_proba = get_neuron_interval_proba(output)
    batch_size = 5
    sample_size_list = [30,60,90,120,150,180]
    save_dir = "exp_data/sport/sampling/num/ces/samples"
    for sample_size in sample_size_list:
        print("采样数量:{}".format(sample_size))
        cur_num_dir = os.path.join(save_dir, str(sample_size))
        if os.path.exists(cur_num_dir) == False:
            os.makedirs(cur_num_dir)
        for repeat in range(repeat_num):
            print("采样数量:{}, 重复第{}次".format(sample_size,repeat))
            selected_index_array = sampling(batch_size, test_df, neuron_interval, neuron_proba, output, sample_size)
            selected_index_list = selected_index_array.tolist()
            trueOrFalse_list = [ True if i in selected_index_list else False for i in test_df["merged_idx"] ]
            sampled_df = test_df[trueOrFalse_list]
            assert sampled_df.shape[0] == sample_size, "采样数量有误"
            sampled_df.to_csv(os.path.join(cur_num_dir,"sampled_"+str(repeat)+".csv"), index=False)


if __name__ == "__main__":
    '''
    得到隐藏层输出
    '''
    # last_hidden_array = getOutput()
    # save_dir = "exp_data/weather/sampling/num/ces/last_hidden_layer_mergedTest_output"
    # save_file_name = "output.npy"
    # file_path = os.path.join(save_dir, save_file_name)
    # np.save(file_path,last_hidden_array)
    '''
    开始采样
    '''
    start_sampling()
   
