import pandas as pd
import numpy as np
import sys
import heapq
import queue  # python queue模块
import multiprocessing
import os
import joblib

sys.path.append("/home/mml/workspace/model_reuse_v2/")
from utils import deleteIgnoreFile, saveData, getOverlapGlobalLabelIndex, str_probabilty_list_To_list

def get_max_secondMax(prob_list):
    max, secondMax = heapq.nlargest(2,prob_list)
    max_idex = prob_list.index(max)
    prob_list[max_idex] = 0
    second_idex = prob_list.index(secondMax)
    return max, secondMax, max_idex, second_idex

def getConfusionMatrix(df):
    # 获得类名
    classes = df["label"].unique()
    classes = np.sort(classes).tolist()
    # 声明一个二维矩阵
    matrix = np.zeros((len(classes), len(classes)),dtype=object)
    # 为每个槽位放一个优先级队列
    for i in range(len(classes)):
        for j in range(len(classes)):
            matrix[i][j] = queue.PriorityQueue()
    # 遍历采样池样本
    for row_index, row in df.iterrows():
        prob_list = str_probabilty_list_To_list(row["probability_vector_model_Combination"])
        merged_idx = row["merged_idx"]
        max, secondMax, max_idex, second_idex = get_max_secondMax(prob_list)
        priority = max / secondMax # 越小越头
        cur_queue = matrix[max_idex][second_idex]
        cur_queue.put([priority, merged_idx])
        matrix[max_idex][second_idex] = cur_queue
    return matrix

def verify(matrix):
    count = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count += matrix[i][j].qsize()
    print(count)

def sampling(matrix, sample_size):
    selected_idx_list = []
    used_size = 0
    while used_size < sample_size: # 数量还没达到来一轮
        # 用于存储该轮数据
        queue_tmp = queue.PriorityQueue()
        # 开始一轮扫描收集
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                cur_queue = matrix[i][j]
                if not cur_queue.empty():
                    queue_tmp.put(cur_queue.get())
        # 该轮数据收集好了，看看能不能全部放入
        if queue_tmp.qsize() <= sample_size - used_size:
            # 可以全部放入
            while not queue_tmp.empty():
                priority, merged_idx = queue_tmp.get()
                selected_idx_list.append(merged_idx)
                used_size += 1
        else: # 不可以全部放入
            while used_size < sample_size:
                if not queue_tmp.empty():
                     priority, merged_idx = queue_tmp.get()
                     selected_idx_list.append(merged_idx)
                     used_size += 1
    assert len(selected_idx_list) == sample_size, "采样数量有误"
    return selected_idx_list

def start_sampling():
    # 采样池子
    df = pd.read_csv("/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")   
    repeat_num = 1  # mcp 是没有随机性的
    sample_size_list = [30,60,90,120,150,180]
    save_dir = "exp_data/weather/sampling/num/mcp/samples"
    for sample_size in sample_size_list:
        print("采样数量:{}".format(sample_size))
        cur_num_dir = os.path.join(save_dir, str(sample_size))
        if os.path.exists(cur_num_dir) == False:
            os.makedirs(cur_num_dir)
        for repeat in range(repeat_num):
            print("采样数量:{}, 重复第{}次".format(sample_size,repeat))
            matrix = getConfusionMatrix(df)
            matrix = matrix.tolist() # 适配
            selected_index_list = sampling(matrix, sample_size)
            trueOrFalse_list = [ True if i in selected_index_list else False for i in df["merged_idx"] ]
            sampled_df = df[trueOrFalse_list]
            assert sampled_df.shape[0] == sample_size, "采样数量有误"
            sampled_df.to_csv(os.path.join(cur_num_dir,"sampled_"+str(repeat)+".csv"), index=False)

if __name__ == "__main__":
    '''
    直接采样
    '''
    start_sampling()
    '''
    得到预测分类混淆矩阵
    '''
    # matrix = getConfusionMatrix()
    # verify(matrix)
    # matrix = matrix.tolist()
    # save_dir = "exp_data/sport/sampling/num/mcp/matrix_data"
    # file_name = "matrix.data"
    # file_path = os.path.join(save_dir, file_name)
    # # np.save(file_path, matrix, False)
    # saveData(matrix, file_path)

