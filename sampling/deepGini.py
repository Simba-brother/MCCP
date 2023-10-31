import sys
import queue as Q
import pandas as pd
import os
import joblib
sys.path.append("./")
from utils import deleteIgnoreFile, saveData, getOverlapGlobalLabelIndex, str_probabilty_list_To_list

def getDeepGiniQueue(df):
    queue = Q.PriorityQueue()
    for row_index, row in df.iterrows():
        prob_list = str_probabilty_list_To_list(row["probability_vector_model_Combination"])
        merged_idx = row["merged_idx"]
        sum = 0
        for prob in prob_list:
            sum += prob*prob
        priority = -(1 - sum)
        queue.put([priority, merged_idx])
    assert queue.qsize() == df.shape[0], "数量不对"
    return queue

def sampling(df, queue, sample_size):
    used_size = 0
    selected_idex_list = []
    while used_size < sample_size:
        if not queue.empty():
            priority, merged_idx = queue.get()
            selected_idex_list.append(merged_idx)
            used_size += 1
    assert len(selected_idex_list) == sample_size, "采样数目不对"
    return selected_idex_list

def start_sampling():

    df = pd.read_csv("/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
    repeat_num = 1  # deepGini 是没有随机性的
    sample_size_list = [30,60,90,120,150,180]
    save_dir = "exp_data/weather/sampling/num/deepGini/samples"
    for sample_size in sample_size_list:
        print("采样数量:{}".format(sample_size))
        cur_num_dir = os.path.join(save_dir, str(sample_size))
        if os.path.exists(cur_num_dir) == False:
            os.makedirs(cur_num_dir)
        for repeat in range(repeat_num):
            print("采样数量:{}, 重复第{}次".format(sample_size,repeat))
            queue = joblib.load("exp_data/weather/sampling/num/deepGini/queue/queue.data")
            selected_index_list = sampling(df, queue, sample_size)
            trueOrFalse_list = [ True if i in selected_index_list else False for i in df["merged_idx"] ]
            sampled_df = df[trueOrFalse_list]
            assert sampled_df.shape[0] == sample_size, "采样数量有误"
            sampled_df.to_csv(os.path.join(cur_num_dir,"sampled_"+str(repeat)+".csv"), index=False)

if __name__ == "__main__":
    '''
    保存优先级队列
    '''
    # df = pd.read_csv("/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
    # queue = getDeepGiniQueue(df)
    # save_dir = "exp_data/weather/sampling/num/deepGini/queue"
    # file_name = "queue.data"
    # file_path = os.path.join(save_dir, file_name)
    # saveData(queue, file_path)
    '''
    开始采样
    '''
    start_sampling()
