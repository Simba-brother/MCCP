
from tensorflow.keras.models import Model, Sequential, load_model
import sys
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import queue  # python queue模块
import joblib
import random
sys.path.append("./")
from utils import deleteIgnoreFile, saveData, getOverlapGlobalLabelIndex, str_probabilty_list_To_list

# 设置训练显卡
os.environ['CUDA_VISIBLE_DEVICES']='1'

def start_sampling():
    sample_num_list = [30,60,90,120,150,180]
    repeat_num = 10 
    # 采样池子csv_path
    csv_path = "/data/mml/overlap_v2_datasets/animal/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv"
    # 打好伪标记的csv
    # pseudo_csv_path_score_4 = "exp_data/Fruit/sampling/num/our/samples/pseudo/score_e_4/pseudo_df.csv"
    # pseudo_csv_path_score_3 = "exp_data/Fruit/sampling/num/our/samples/pseudo/score_e_3/pseudo_df.csv"
    
    # 保存的队列的目录path
    # rest_queue_dirPath = "exp_data/sport/sampling/num/our/pseudo/score_e_4/queueData"
    # 全池子queue
    total_queue_dirPath = "exp_data/animal/sampling/queueData"
    # score_queue_dir
    # score_queue_dirPath = "exp_data/sport/sampling/queueData/pseudo"
    # 采样后保存的目录
    save_dir = "exp_data/animal/sampling/num/our/pseudo/score_e_4_3/method_11/cutOff_10"
    curOff_ratio = 0.1
    for sample_num in sample_num_list:
        # 创建出采样数目的文件夹
        cur_num_dir = os.path.join(save_dir, str(sample_num))
        if os.path.exists(cur_num_dir) == False:
            os.makedirs(cur_num_dir)
        for repeat in range(repeat_num):
            sampled_df = sampling_method_11(sample_num, csv_path, total_queue_dirPath, curOff_ratio)
            sampled_df.to_csv(os.path.join(cur_num_dir,"sampled_"+str(repeat)+".csv"), index=False)
    print("start_sampling() success")

def sampling_method_1(sample_num, csv_path, queue_dirPath):
    '''
        - hardOrder
        - 你出一个我出一个,(同权抽样)
        - 无随机性

    '''
    df = pd.read_csv(csv_path)    
    # 加载队列
    queue_agree = joblib.load(os.path.join(queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(queue_dirPath, "queue_exception.data"))
    # 开始出队列
    sampled_list = []
    while len(sampled_list) < sample_num:
        if not queue_agree.empty():
            cur_dif, merged_idx = queue_agree.get()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
        if not queue_adversary.empty():
            cur_dif, merged_idx = queue_adversary.get()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
        if not queue_adv_A_ex_B.empty():
            cur_dif, merged_idx = queue_adv_A_ex_B.get()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
        if not queue_adv_B_ex_A.empty():
            cur_dif, merged_idx = queue_adv_B_ex_A.get()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
        if not queue_exception.empty():
            cur_dif, merged_idx = queue_exception.get()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"
    return sampled_df

def sampling_method_2(sample_num, csv_path, queue_dirPath):
    '''
        - 每个队列都是随机采样(不从queue_agree采样)
        - 你出一个我出一个,(同权抽样)
        - 有随机性
    '''
    df = pd.read_csv(csv_path)    
    # 加载队列
    # queue_agree = joblib.load(os.path.join(queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(queue_dirPath, "queue_exception.data"))
    # 队列转list方便好操作
    list_adversary = queueToList(queue_adversary)
    list_adv_A_ex_B = queueToList(queue_adv_A_ex_B)
    list_adv_B_ex_A = queueToList(queue_adv_B_ex_A)
    list_exception = queueToList(queue_exception)
    # 打乱 list
    random.shuffle(list_adversary)
    random.shuffle(list_adv_A_ex_B)
    random.shuffle(list_adv_B_ex_A)
    random.shuffle(list_exception)

    # 开始采样
    sampled_list = []
    while len(sampled_list) < sample_num:
        if len(list_adversary) != 0:
            merged_idx = list_adversary.pop()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
        if len(list_adv_A_ex_B) != 0:
            merged_idx = list_adv_A_ex_B.pop()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
        if len(list_adv_B_ex_A) != 0:
            merged_idx = list_adv_B_ex_A.pop()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
        if len(list_exception) != 0:
            merged_idx = list_exception.pop()
            sampled_list.append(merged_idx)
            if len(sampled_list) == sample_num:
                break
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"
    return sampled_df

def sampling_method_3(sample_num, csv_path, queue_dirPath):
    '''
        - 每个队列都是随机采样(不从queue_agree采样)
        - 按照队列占比采样(带权采样)
        - 有随机性
    '''
    df = pd.read_csv(csv_path)    
    # 加载队列
    # queue_agree = joblib.load(os.path.join(queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(queue_dirPath, "queue_exception.data"))
    # 队列转list方便好操作
    list_adversary = queueToList(queue_adversary)
    list_adv_A_ex_B = queueToList(queue_adv_A_ex_B)
    list_adv_B_ex_A = queueToList(queue_adv_B_ex_A)
    list_exception = queueToList(queue_exception)
    # 打乱 list
    random.shuffle(list_adversary)
    random.shuffle(list_adv_A_ex_B)
    random.shuffle(list_adv_B_ex_A)
    random.shuffle(list_exception)

    # 按队列占比 分配 采样数目
    total = len(list_adversary)+len(list_adv_A_ex_B)+len(list_adv_B_ex_A)+len(list_exception)
    target_adv =  round(len(list_adversary)/total * sample_num)  # 四舍五入取整数
    target_adv_A_ex_B = round(len(list_adv_A_ex_B)/total * sample_num)
    target_adv_B_ex_A = round(len(list_adv_B_ex_A)/total * sample_num)
    target_exception = sample_num - (target_adv+target_adv_A_ex_B+target_adv_B_ex_A)
    # 开始采样
    sampled_list = []
    sampled_list.extend(random.sample(list_adversary, target_adv)) 
    sampled_list.extend(random.sample(list_adv_A_ex_B, target_adv_A_ex_B))
    sampled_list.extend(random.sample(list_adv_B_ex_A, target_adv_B_ex_A))
    sampled_list.extend(random.sample(list_exception, target_exception))
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"
    return sampled_df

def sampling_method_4(sample_num, csv_path, pseudo_csv_path, queue_dirPath):
    '''
        - 每个队列都是硬序列(不从queue_agree采样)
        - 按照队列占比采样(带权采样)
        - 无随机性
    '''
    df = pd.read_csv(csv_path)    
    # 加载队列
    # queue_agree = joblib.load(os.path.join(queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(queue_dirPath, "queue_exception.data"))

    # 按队列占比 分配 采样数目
    total = queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()
    target_adv =  min(round(queue_adversary.qsize()/total * sample_num),  queue_adversary.qsize()) # 四舍五入取整数
    target_adv_A_ex_B = min(round(queue_adv_A_ex_B.qsize()/total * sample_num), queue_adv_A_ex_B.qsize())
    target_adv_B_ex_A = min(round(queue_adv_B_ex_A.qsize()/total * sample_num), queue_adv_B_ex_A.qsize())
    target_exception = min(sample_num-(target_adv+target_adv_A_ex_B+target_adv_B_ex_A), queue_exception.qsize())
    # target_agree = min(sample_num-(target_adv+target_adv_A_ex_B+target_adv_B_ex_A+target_exception), queue_agree.qsize())
    # 开始采样
    sampled_list = []
    for i in range(target_adv):
        cur_dif, merged_idx = queue_adversary.get()
        sampled_list.append(merged_idx)
    for i in range(target_adv_A_ex_B):
        cur_dif, merged_idx = queue_adv_A_ex_B.get()
        sampled_list.append(merged_idx)
    for i in range(target_adv_B_ex_A):
        cur_dif, merged_idx = queue_adv_B_ex_A.get()
        sampled_list.append(merged_idx)
    for i in range(target_exception):
        cur_dif, merged_idx = queue_exception.get()
        sampled_list.append(merged_idx)
    # for i in range(target_agree):
    #     cur_dif, merged_idx = queue_agree.get()
    #     sampled_list.append(merged_idx)
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"
    # predic_IsOverlap_model_Combination_series=pd.Series(predic_IsOverlap_model_Combination_list, name="predic_IsOverlap_model_Combination")
    sampled_df["label_P"] = sampled_df["label"]
    pseudo_df = pd.read_csv(pseudo_csv_path)
    pseudo_df["label_P"] = pseudo_df["pseudo_label"]
    # 将sampled_df 和 pseudo_df 合并
    new_df = pd.concat([sampled_df, pseudo_df], ignore_index=True)
    return new_df

def sampling_method_5(sample_num, csv_path, pseudo_csv_path, rest_queue_dirPath, total_queue_dirPath):

    '''
    将4分的伪标注, 按照agree队列所占比例。去做一下额外采样。保证采样队列分布不被打偏
    csv_path: 全池子
    pseudo_csv_path: 伪标注csv
    total_queue_dirPath: 全池子下的队列数据
    rest_queue_dirPath: 对应伪标注剩余的数据被分的队列
    '''
    # 读取候选池
    df = pd.read_csv(csv_path)    
    # 伪标注池子
    pseudo_df = pd.read_csv(pseudo_csv_path)

    # 加载total队列
    queue_agree = joblib.load(os.path.join(total_queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(total_queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(total_queue_dirPath, "queue_exception.data"))
    total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()

    rest_queue_agree = joblib.load(os.path.join(rest_queue_dirPath, "queue_agree.data")) # mis
    rest_queue_adversary = joblib.load(os.path.join(rest_queue_dirPath, "queue_adversary.data"))
    rest_queue_adv_A_ex_B = joblib.load(os.path.join(rest_queue_dirPath, "queue_adv_A_ex_B.data"))
    rest_queue_adv_B_ex_A = joblib.load(os.path.join(rest_queue_dirPath, "queue_adv_B_ex_A.data"))
    rest_queue_exception = joblib.load(os.path.join(rest_queue_dirPath, "queue_exception.data"))


    target_agree = min( round( queue_agree.qsize()/total * sample_num ) , pseudo_df.shape[0])
    target_adv= min( round( queue_adversary.qsize()/total * sample_num) , rest_queue_adversary.qsize())
    target_AB = min( round( queue_adv_A_ex_B.qsize()/total * sample_num) , rest_queue_adv_A_ex_B.qsize())
    target_BA = min( round( queue_adv_B_ex_A.qsize()/total * sample_num) , rest_queue_adv_B_ex_A.qsize())
    target_exception = min( sample_num-(target_adv+target_AB+target_BA), rest_queue_exception.qsize())

    sampled_list = []

    list_adversary = queueToList(rest_queue_adversary)
    list_adv_A_ex_B = queueToList(rest_queue_adv_A_ex_B)
    list_adv_B_ex_A = queueToList(rest_queue_adv_B_ex_A)
    list_exception = queueToList(rest_queue_exception)
    sampled_list.extend(random.sample(list_adversary, target_adv)) 
    sampled_list.extend(random.sample(list_adv_A_ex_B, target_AB))
    sampled_list.extend(random.sample(list_adv_B_ex_A, target_BA))
    sampled_list.extend(random.sample(list_exception, target_exception))
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    sampled_df["label_P"] = sampled_df["label"]

    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"

    # 从伪标注中挑选
    pseudo_sampled_df = pseudo_df.sample(n=target_agree, replace=False) # 不允许重复采样 
    pseudo_sampled_df["label_P"] = pseudo_sampled_df["pseudo_label"]

    # 将sampled_df 和 pseudo_df 合并
    new_df = pd.concat([sampled_df, pseudo_sampled_df], ignore_index=True)

    return new_df

def sampling_method_6(sample_num, csv_path, pseudo_csv_path_score_4, pseudo_csv_path_score_3, total_queue_dirPath):
    '''
    伪标注3分和4分都选
    score_3_num:score_4_num = queue_agree : other
    order: random
    有随机性
    '''
    # 所有4 分 伪标记df
    pseudo_df_score_4 = pd.read_csv(pseudo_csv_path_score_4)
    # 所有3 分 伪标记df
    pseudo_df_score_3 = pd.read_csv(pseudo_csv_path_score_3)
    # 所有df
    df = pd.read_csv(csv_path)

    # 加载total队列
    queue_agree = joblib.load(os.path.join(total_queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(total_queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(total_queue_dirPath, "queue_exception.data"))
    total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()

    # 每个队列应该采样的数量
    target_agree = round( queue_agree.qsize()/total * sample_num )
    target_adv = round( queue_adversary.qsize()/total * sample_num )
    target_AB = round( queue_adv_A_ex_B.qsize()/total * sample_num )
    target_BA = round( queue_adv_B_ex_A.qsize()/total * sample_num )
    target_exception = sample_num - (target_agree+target_adv+target_AB+target_BA)

    # score 4 伪标注 采样
    pseudo_sampled_score_4_df = pseudo_df_score_4.sample(n=target_agree, replace=False) # 不允许重复采样 
    pseudo_sampled_score_4_df["label_P"] = pseudo_sampled_score_4_df["pseudo_label"]
    # score 3 伪标注采样，只采样 不是overlap 的
    pseudo_df_score_3 = pseudo_df_score_3[pseudo_df_score_3["predic_IsOverlap_model_Combination"] == 0]
    pseudo_sampled_score_3_df = pseudo_df_score_3.sample(n=sample_num-target_agree, replace=False)
    pseudo_sampled_score_3_df["label_P"] = pseudo_sampled_score_3_df["pseudo_label"]
    # 记录已经用于伪标注的样本merged_idx_list
    pseudo_merged_idx_list = list(pseudo_sampled_score_4_df["merged_idx"])+list(pseudo_sampled_score_3_df["merged_idx"])
    # 队列tolist
    list_agree = queueToList(queue_agree)
    list_adversary = queueToList(queue_adversary)
    list_adv_A_ex_B = queueToList(queue_adv_A_ex_B)
    list_adv_B_ex_A = queueToList(queue_adv_B_ex_A)
    list_exception = queueToList(queue_exception)
    # 每个list 剔除 已经用于伪标记的
    list_agree_rest = [i for i in list_agree if i not in pseudo_merged_idx_list] 
    list_adversary_rest = [i for i in list_adversary if i not in pseudo_merged_idx_list] 
    list_adv_A_ex_B_rest = [i for i in list_adv_A_ex_B if i not in pseudo_merged_idx_list] 
    list_adv_B_ex_A_rest = [i for i in list_adv_B_ex_A if i not in pseudo_merged_idx_list] 
    list_exception_rest = [i for i in list_exception if i not in pseudo_merged_idx_list] 

    # 确定一下每个list 采样数量
    target_agree = min(target_agree, len(list_agree_rest))
    target_adv = min(target_adv, len(list_adversary_rest))
    target_AB = min(target_AB, len(list_adv_A_ex_B_rest))
    target_BA = min(target_BA, len(list_adv_B_ex_A_rest))
    target_exception = min(target_exception, len(list_exception_rest)) 
    # 采样
    sampled_list = []
    sampled_list.extend(random.sample(list_agree_rest, target_agree)) 
    sampled_list.extend(random.sample(list_adversary_rest, target_adv)) 
    sampled_list.extend(random.sample(list_adv_A_ex_B_rest, target_AB)) 
    sampled_list.extend(random.sample(list_adv_B_ex_A_rest, target_BA)) 
    sampled_list.extend(random.sample(list_exception_rest, target_exception)) 
    
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"
    sampled_df["label_P"] = sampled_df["label"]

    # 将sampled_df 和 pseudo_df 合并
    new_df = pd.concat([sampled_df, pseudo_sampled_score_4_df, pseudo_sampled_score_3_df], ignore_index=True)

    return new_df

def sampling_method_7(sample_num, csv_path, total_queue_dirPath, score_queue_dirPath):
    '''
    score 4 : score_3 = queue_agree:other。 
    score_3数量在精度还可以的情况下尽可能多的采样。以便增加伪标注数量
    '''
    # 采样池
    df = pd.read_csv(csv_path)
    # 加载total队列
    queue_agree = joblib.load(os.path.join(total_queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(total_queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(total_queue_dirPath, "queue_exception.data"))
    total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()

    # 每个队列应该采样的数量
    target_agree = round( queue_agree.qsize()/total * sample_num )
    target_adv = round( queue_adversary.qsize()/total * sample_num )
    target_AB = round( queue_adv_A_ex_B.qsize()/total * sample_num )
    target_BA = round( queue_adv_B_ex_A.qsize()/total * sample_num )
    target_exception = sample_num - (target_agree+target_adv+target_AB+target_BA)

    # 加载伪标记分数队列
    queue_score_4 = joblib.load(os.path.join(score_queue_dirPath, "queue_score_4.data"))
    queue_score_3 = joblib.load(os.path.join(score_queue_dirPath, "queue_score_3.data"))
    # target_score_4 = target_agree*2
    # target_score_3 = sample_num*2-target_score_4
    r3_r4 = round((total-queue_agree.qsize()) / queue_agree.qsize(), 4)
    target_score_3 = 800 # 尽可能选大一点，同时保证精度要高
    target_score_4 = round(round(1 / r3_r4, 4) * target_score_3)

    sampled_pseudo_list = []

    list_score_4 = queueToList(queue_score_4)
    sampled_pseudo_list.extend(random.sample(list_score_4, target_score_4))

    for _ in range(target_score_3):
        if not queue_score_3.empty():
            priority,merged_idx = queue_score_3.get()
            sampled_pseudo_list.append(merged_idx)
    
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_pseudo_list else False for i in df["merged_idx"] ]
    sampled_pseudo_df = df[trueOrFalse_list]
    sampled_pseudo_df["label_P"] = sampled_pseudo_df["pseudo_label"]

    # 队列tolist
    list_agree = queueToList(queue_agree)
    list_adversary = queueToList(queue_adversary)
    list_adv_A_ex_B = queueToList(queue_adv_A_ex_B)
    list_adv_B_ex_A = queueToList(queue_adv_B_ex_A)
    list_exception = queueToList(queue_exception)
    # 每个list 剔除 已经用于伪标记的
    list_agree_rest = [i for i in list_agree if i not in sampled_pseudo_list] 
    list_adversary_rest = [i for i in list_adversary if i not in sampled_pseudo_list] 
    list_adv_A_ex_B_rest = [i for i in list_adv_A_ex_B if i not in sampled_pseudo_list] 
    list_adv_B_ex_A_rest = [i for i in list_adv_B_ex_A if i not in sampled_pseudo_list] 
    list_exception_rest = [i for i in list_exception if i not in sampled_pseudo_list] 
    # 确定一下每个list 采样数量
    target_agree = min(target_agree, len(list_agree_rest))
    target_adv = min(target_adv, len(list_adversary_rest))
    target_AB = min(target_AB, len(list_adv_A_ex_B_rest))
    target_BA = min(target_BA, len(list_adv_B_ex_A_rest))
    target_exception = min(target_exception, len(list_exception_rest)) 
    # 采样
    sampled_list = []
    sampled_list.extend(random.sample(list_agree_rest, target_agree)) 
    sampled_list.extend(random.sample(list_adversary_rest, target_adv)) 
    sampled_list.extend(random.sample(list_adv_A_ex_B_rest, target_AB)) 
    sampled_list.extend(random.sample(list_adv_B_ex_A_rest, target_BA)) 
    sampled_list.extend(random.sample(list_exception_rest, target_exception)) 

    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"
    sampled_df["label_P"] = sampled_df["label"]

    # 将sampled_df 和 pseudo_df 合并
    new_df = pd.concat([sampled_df, sampled_pseudo_df], ignore_index=True)

    return new_df
    
def sampling_method_8(sample_num, csv_path, total_queue_dirPath):
    '''
    验证性采样
    大量数据采样 分布满足 5队列分布。
    '''
    df = pd.read_csv(csv_path)
    # 加载total队列
    queue_agree = joblib.load(os.path.join(total_queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(total_queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(total_queue_dirPath, "queue_exception.data"))
    total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()

    # 每个队列应该采样的数量
    target_agree = round( queue_agree.qsize()/total * sample_num )
    target_adv = round( queue_adversary.qsize()/total * sample_num )
    target_AB = round( queue_adv_A_ex_B.qsize()/total * sample_num )
    target_BA = round( queue_adv_B_ex_A.qsize()/total * sample_num )
    target_exception = sample_num - (target_agree+target_adv+target_AB+target_BA)

    # queueTolist
    list_agree = queueToList(queue_agree)
    list_adversary = queueToList(queue_adversary)
    list_AB = queueToList(queue_adv_A_ex_B)
    list_BA = queueToList(queue_adv_B_ex_A)
    list_exp = queueToList(queue_exception)

    # 随机
    sampled_list = []
    sampled_list.extend(random.sample(list_agree, target_agree)) 
    sampled_list.extend(random.sample(list_adversary, target_adv)) 
    sampled_list.extend(random.sample(list_AB, target_AB)) 
    sampled_list.extend(random.sample(list_BA, target_BA)) 
    sampled_list.extend(random.sample(list_exp, target_exception)) 

    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == sample_num, "否则采样数量出错"

    return sampled_df

def sampling_method_9(sample_num, csv_path, total_queue_dirPath, acc_hold = 0.95):
    '''
    既要保证伪标注3分的精度阈值 超过acc_hold
    又要保证3分伪标注满足队列分布
    '''
    # 加载 采样池
    df = pd.read_csv(csv_path)
    # 加载total队列
    queue_agree = joblib.load(os.path.join(total_queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(total_queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(total_queue_dirPath, "queue_exception.data"))
    total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()

    # queueTolist
    list_agree = queueToList(queue_agree)
    list_adversary = queueToList(queue_adversary)
    list_AB = queueToList(queue_adv_A_ex_B)
    list_BA = queueToList(queue_adv_B_ex_A)
    list_exp = queueToList(queue_exception)

    total = len(list_agree) + len(list_adversary) + len(list_AB) + len(list_BA) + len(list_exp)
    # 只保留每个队列中的3分样本
    
    list_adversary_score3 = []
    for merged_idx in list_adversary:
        row = df[df["merged_idx"] == merged_idx].iloc[0]
        if row["pseudo_label_score"] == 3:
            list_adversary_score3.append(merged_idx)

    list_AB_score3 = []
    for merged_idx in list_AB:
        row = df[df["merged_idx"] == merged_idx].iloc[0]
        if row["pseudo_label_score"] == 3:
            list_AB_score3.append(merged_idx)
    
    list_BA_score3 = []
    for merged_idx in list_BA:
        row = df[df["merged_idx"] == merged_idx].iloc[0]
        if row["pseudo_label_score"] == 3:
            list_BA_score3.append(merged_idx)

    list_exp_score3 = []
    for merged_idx in list_exp:
        row = df[df["merged_idx"] == merged_idx].iloc[0]
        if row["pseudo_label_score"] == 3:
            list_exp_score3.append(merged_idx)

    # 打乱 队列
    random.shuffle(list_agree)
    random.shuffle(list_adversary_score3)
    random.shuffle(list_AB_score3)
    random.shuffle(list_BA_score3)
    random.shuffle(list_exp_score3)

    # 准备挨个遍历各个组
    group_list = [list_adversary_score3, list_AB_score3, list_BA_score3, list_exp_score3]

    # 最短的队列长度为多少，设置为单位1
    min_len = float("inf")
    # 记录最短非空队列 的 index
    min_i = 0
    # 记录 剩余三分的数量
    total_score_3 = 0

    # 遍历 group_list
    for i, cur_group in enumerate(group_list):
        total_score_3 += len(cur_group) # 累加数量
        if len(cur_group) < min_len and len(cur_group) != 0:
            min_i = i # 记录 最小队列索引位置

    # 将最小队列 放到 第1组
    temp = group_list[0]
    group_list[0] = group_list[min_i]
    group_list[min_i] = temp
    
    # 记录每一组当前已经采样的数量
    sampled_num_list = [0 for i in range(len(group_list))]

    # 先存一下各组的数量
    groupOfNum_list = []
    for cur_group in group_list:
        groupOfNum_list.append(len(cur_group))

    # 记录样本的 merged_idex
    sampled_idex_list = []
    # 记录当前样本为标志匹配计数情况
    match_list = []
    # 人力标注资源消耗
    cost = sample_num
    # 记录人力资源标注样本的merged_idx
    cost_merged_idx_list = []
    # 消耗还没用完，而且 队列中还有值得时候 进行一轮 比例采样
    while cost > 0 and len(sampled_idex_list) < total_score_3:
        # 开始一轮
        for i, cur_group in enumerate(group_list):
            if cost == 0:
                break
            # 如果是单位 1 队列 当前就采样1个
            if i == 0:
                if len(cur_group) > 0:
                    # 从第一个list 也就是 数量最少并且非零的 list 中 先随机挑一个 
                    index = random.randint(0,len(cur_group)-1) 
                    merged_index = cur_group.pop(index)
                    sampled_idex_list.append(merged_index)
                    sampled_num_list[0] += 1
                    row = df[df["merged_idx"] == merged_index].iloc[0]
                    if row["label"] == row["pseudo_label"]:
                        match_list.append(1)
                    else:
                        match_list.append(0)
                    cur_acc = round(sum(match_list) / len(match_list),2)
                    if cur_acc < acc_hold:
                        match_list[-1] = 1 # 给最后采样的这个样本 人工 掰回来 即 消耗一个人力
                        cost -= 1
                        cost_merged_idx_list.append(merged_index)
                        if cost == 0:
                            break
                    
            else:
                # 根据当前单位1 采样数量 确定一下 该组需要采样的一个 数量
                cur_sample_num =  round(groupOfNum_list[i] / groupOfNum_list[0] * sampled_num_list[0])
                # 减去已经从当前队列 采样的 数量，就是 本轮还需要采样的数量
                need_sample_num = cur_sample_num - sampled_num_list[i]
                for _ in range(need_sample_num):
                    if len(cur_group) > 0:
                        # 从第一个list 也就是 数量最少并且非零的 list 中 先随机挑一个 
                        index = random.randint(0,len(cur_group)-1) 
                        merged_index = cur_group.pop(index)
                        sampled_idex_list.append(merged_index)
                        sampled_num_list[i] += 1
                        row = df[df["merged_idx"] == merged_index].iloc[0]
                        if row["label"] == row["pseudo_label"]:
                            match_list.append(1)
                        else:
                            match_list.append(0)
                        cur_acc = round(sum(match_list) / len(match_list),2)
                        if cur_acc < acc_hold:
                            match_list[-1] = 1 # 给最后采样的这个样本 人工 掰回来 即 消耗一个人力
                            cost -= 1
                            cost_merged_idx_list.append(merged_index)
                            if cost == 0:
                                break
                        
    assert round(sum(match_list) / len(match_list),2) >= acc_hold, "采样策略有误"
    # 根据队列分布确定agree队列数量
    target_agree_temp = len(list_agree) / (total-len(list_agree))  * len(sampled_idex_list)
    target_agree = min(round(target_agree_temp), len(list_agree))
    sampled_idex_list.extend(random.sample(list_agree, target_agree))
    if cost == 0:
        print("人力指标:{}, 消耗完了".format(sample_num))
    else:
        print("人力指标:{}, 未消耗完".format(sample_num))
    
    # 开始采样
    # 从池子df中挑选
    trueOrFalse_list = [ True if i in sampled_idex_list else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    assert sampled_df.shape[0] == (sum(sampled_num_list) + target_agree), "采样数量有误"

    print("总共的采样数量: {}".format(sampled_df.shape[0]))
    # 人工打标
    label_P_list = []
    for row_idx, row in sampled_df.iterrows():
        if row["merged_idx"] in cost_merged_idx_list:
            label_P_list.append(row["label"])
        else:
            label_P_list.append(row["pseudo_label"])
    label_P_series=pd.Series(label_P_list, name="label_P")
    sampled_df.reset_index(inplace=True,drop=True)
    sampled_df = pd.concat([sampled_df, label_P_series], axis=1)
    return sampled_df

def sampling_method_10(sample_num, csv_path, total_queue_dirPath):
    # 加载 采样池
    df = pd.read_csv(csv_path)
    # 加载total队列
    queue_agree = joblib.load(os.path.join(total_queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(total_queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(total_queue_dirPath, "queue_exception.data"))
    total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()
    # 队列排序
    sort_flag = "1"
    sorted_queue_adversary = sortQueue(queue_adversary, sort_flag, df)
    sorted_queue_adv_A_ex_B= sortQueue(queue_adv_A_ex_B, sort_flag, df)
    sorted_queue_adv_B_ex_A = sortQueue(queue_adv_B_ex_A, sort_flag, df)
    sorted_queue_exception = sortQueue(queue_exception, sort_flag, df)
    # 队列截取
    curOff_ratio = 0.25
    agree_list = cutQueueToList(queue_agree, curOff_ratio)
    sorted_adversary_list = cutQueueToList(sorted_queue_adversary, curOff_ratio)
    sorted_adv_A_ex_B_list = cutQueueToList(sorted_queue_adv_A_ex_B,curOff_ratio)
    sorted_adv_B_ex_A_list = cutQueueToList(sorted_queue_adv_B_ex_A,curOff_ratio)
    sorted_exception_list = cutQueueToList(sorted_queue_exception,curOff_ratio)
    total_disagree = len(sorted_adversary_list) + len(sorted_adv_A_ex_B_list) + len(sorted_adv_B_ex_A_list) + len(sorted_exception_list)
    # 队列采样
    disagree_selected = []
    adversary_targetSize = round(len(sorted_adversary_list) / total_disagree * sample_num)
    AB_targetSize = round(len(sorted_adv_A_ex_B_list) / total_disagree * sample_num)
    BA_targetSize = round(len(sorted_adv_B_ex_A_list) / total_disagree * sample_num)
    exp_targetSize = sample_num - (adversary_targetSize+AB_targetSize+BA_targetSize)
    adv_seletced = random.sample(sorted_adversary_list, adversary_targetSize)
    AB_selected = random.sample(sorted_adv_A_ex_B_list, AB_targetSize)
    BA_selected = random.sample(sorted_adv_B_ex_A_list, BA_targetSize)
    exp_selected = random.sample(sorted_exception_list, exp_targetSize)
    # disAgree
    disagree_selected = adv_seletced+AB_selected+BA_selected+exp_selected
    assert len(disagree_selected) == sample_num, "采样数目不对"
    # agree
    agree_targetSize = round(len(disagree_selected) * (len(agree_list) / total_disagree))
    agree_selected = random.sample(agree_list, agree_targetSize)

    selected = disagree_selected+agree_selected
    trueOrFalse_list = [ True if i in selected else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    # 人工打标
    label_P_list = []
    for row_idx, row in sampled_df.iterrows():
        if row["merged_idx"] in disagree_selected: # 消耗在diAgree 知识分布中
            label_P_list.append(row["label"])
        else:
            label_P_list.append(row["pseudo_label"])
    label_P_series=pd.Series(label_P_list, name="label_P")
    sampled_df.reset_index(inplace=True,drop=True)
    sampled_df = pd.concat([sampled_df, label_P_series], axis=1)
    return sampled_df

def sampling_method_11(sample_num, csv_path, total_queue_dirPath, curOff_ratio):
    '''
    disagree 队列 用 deepGini 顺序。前25%(curOff_ratio=0.25) 使用伪标记, rest random sample_num 用于 真标记 cost
    args:
        sample_num:cost
        csv_path:总采样池
        total_queue_dirPath: 总知识分布
        cut_off: disagree 切
    '''
    # 加载 采样池
    df = pd.read_csv(csv_path)
    # 加载total队列
    queue_agree = joblib.load(os.path.join(total_queue_dirPath, "queue_agree.data"))
    queue_adversary = joblib.load(os.path.join(total_queue_dirPath, "queue_adversary.data"))
    queue_adv_A_ex_B = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_A_ex_B.data"))
    queue_adv_B_ex_A = joblib.load(os.path.join(total_queue_dirPath, "queue_adv_B_ex_A.data"))
    queue_exception = joblib.load(os.path.join(total_queue_dirPath, "queue_exception.data"))
    # 计算 池子中样本总数量，不同意数量， 同意数量
    total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()
    total_disagree = queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()
    total_agree = queue_agree.qsize()
    # adv, AB, BA, exp 的 cost
    cost_adv = round(queue_adversary.qsize() / total_disagree * sample_num)
    cost_AB = round(queue_adv_A_ex_B.qsize() / total_disagree * sample_num)
    cost_BA = round(queue_adv_B_ex_A.qsize() / total_disagree * sample_num)
    cost_exp = sample_num - (cost_adv+cost_AB+cost_BA)

    # 队列排序
    sort_flag = "2" # deepGini顺序
    sorted_queue_adversary = sortQueue(queue_adversary, sort_flag, df)
    sorted_queue_adv_A_ex_B= sortQueue(queue_adv_A_ex_B, sort_flag, df)
    sorted_queue_adv_B_ex_A = sortQueue(queue_adv_B_ex_A, sort_flag, df)
    sorted_queue_exception = sortQueue(queue_exception, sort_flag, df)

    # 队列截取
    sampled_idex_list_disagree_p = []  # disagree 队列 前 百分比 伪精度较高
    adv_p_targetNum = int(sorted_queue_adversary.qsize()*curOff_ratio)
    for _ in range(adv_p_targetNum):
        priority, merged_idx = sorted_queue_adversary.get()
        sampled_idex_list_disagree_p.append(merged_idx)

    AB_p_targetNum = int(sorted_queue_adv_A_ex_B.qsize()*curOff_ratio)
    for _ in range(AB_p_targetNum):
        priority, merged_idx = sorted_queue_adv_A_ex_B.get()
        sampled_idex_list_disagree_p.append(merged_idx)

    BA_p_targetNum = int(sorted_queue_adv_B_ex_A.qsize()*curOff_ratio)
    for _ in range(BA_p_targetNum):
        priority, merged_idx = sorted_queue_adv_B_ex_A.get()
        sampled_idex_list_disagree_p.append(merged_idx)

    exception_targetNum = int(sorted_queue_exception.qsize()*curOff_ratio)
    for _ in range(exception_targetNum):
        priority, merged_idx = sorted_queue_exception.get()
        sampled_idex_list_disagree_p.append(merged_idx)

    # 百分比后面的部分，也就是cost消耗的部分
    rest_adv_list = queueToList(sorted_queue_adversary) 
    rest_AB_list = queueToList(sorted_queue_adv_A_ex_B)
    rest_BA_list = queueToList(sorted_queue_adv_B_ex_A)
    rest_exp_list = queueToList(sorted_queue_exception)

    sampled_idex_list_disagree_cost = [] # groundTruth
    sampled_idex_list_disagree_cost.extend(random.sample(rest_adv_list, cost_adv))
    sampled_idex_list_disagree_cost.extend(random.sample(rest_AB_list, cost_AB))
    sampled_idex_list_disagree_cost.extend(random.sample(rest_BA_list, cost_BA))
    sampled_idex_list_disagree_cost.extend(random.sample(rest_exp_list, cost_exp))
    assert len(sampled_idex_list_disagree_cost) == sample_num, "采用数量有误"
    # 计算agree_p的采样数量
    selected_disagree_num =  len(sampled_idex_list_disagree_p) + len(sampled_idex_list_disagree_cost)
    target_agreeNum = round((total_agree / total_disagree) * selected_disagree_num)
    agree_list = queueToList(queue_agree) 
    sampled_idex_list_agree_p = []
    sampled_idex_list_agree_p.extend(random.sample(agree_list, target_agreeNum))

    selected = sampled_idex_list_disagree_p+sampled_idex_list_disagree_cost+sampled_idex_list_agree_p
    trueOrFalse_list = [ True if i in selected else False for i in df["merged_idx"] ]
    sampled_df = df[trueOrFalse_list]
    # 人工打标
    label_P_list = []
    for row_idx, row in sampled_df.iterrows():
        if row["merged_idx"] in sampled_idex_list_disagree_cost: # 消耗在diAgree 知识分布中
            label_P_list.append(row["label"])  # label_P 使用 真标签
        else:
            label_P_list.append(row["pseudo_label"]) # label_P 使用伪标签
    label_P_series=pd.Series(label_P_list, name="label_P")
    sampled_df.reset_index(inplace=True,drop=True)  # importent 因为是从df中抽取产生的sampled_df 故先将 row_idx drop
    sampled_df = pd.concat([sampled_df, label_P_series], axis=1)
    return sampled_df

    

def cutQueueToList(queue, ratio):
    ans_list = []
    top_size = int(queue.qsize()*ratio)
    for _ in range(top_size):
        priorty, merged_idx = queue.get()
        ans_list.append(merged_idx)
    return ans_list

def sortQueue(queue_data, sort_flag, df):
    new_queue = queue.PriorityQueue()
    while not queue_data.empty():
        if sort_flag == "1": # max(Mc,Mb)-Ma 越小优先级越高
            old_priority, merged_idx = queue_data.get()
            row = df[df["merged_idx"] == merged_idx].iloc[0]
            confidence_A = row["confidence_model_A"]
            confidence_B = row["confidence_model_B"]
            confidence_Combination = row["confidence_model_Combination"]
            pseudo_label_score = row["pseudo_label_score"]
            if pseudo_label_score == 3:
                if row["predic_label_model_A"] == row["predic_label_model_Combination"]:
                    # 越小优先级越高
                    new_priority = round(max(confidence_A, confidence_Combination) - confidence_B,4)
                    new_queue.put([new_priority, merged_idx])
                    continue
                elif row["predic_label_model_B"] == row["predic_label_model_Combination"]:
                    # 越小优先级越高
                    new_priority = round(max(confidence_B, confidence_Combination) - confidence_A,4)
                    new_queue.put([new_priority, merged_idx])
                    continue
            elif pseudo_label_score == 0:
                new_priority = round(confidence_Combination - max(confidence_B, confidence_A), 4)
                # 越小优先级越高
                new_queue.put([new_priority, merged_idx])
                continue
            elif pseudo_label_score == 2:
                new_priority = round(max(confidence_B, confidence_A) - confidence_Combination, 4)
                # 越小优先级越高
                new_queue.put([new_priority, merged_idx])
        elif sort_flag == "2": # deepGini 反着来
            old_priority, merged_idx = queue_data.get()
            row = df[df["merged_idx"] == merged_idx].iloc[0]
            probability_vector_model_Combination = row["probability_vector_model_Combination"]
            prob_list = str_probabilty_list_To_list(probability_vector_model_Combination)
            sum = 0
            for prob in prob_list:
                sum += prob*prob
            new_priority = (1 - sum)
            new_queue.put([new_priority, merged_idx])
        else:
            raise Exception("请指定排序方式")
    return new_queue

def getPsedo_df_and_rest_df():
    '''
    从数据集中,确定出哪些是可以打出伪标签的,采样从rest_df中来
    '''
    csv_path = "/data/mml/overlap_v2_datasets/sport/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv"
    df = pd.read_csv(csv_path)
    trueOrFalse_list = []
    for row_index, row in df.iterrows():
        if row["pseudo_label_score"] == 3:
            trueOrFalse_list.append(True)
        else:
            trueOrFalse_list.append(False)
    pseudo_df = df[trueOrFalse_list]
    
    rest_trueOrFalse_list = []
    for flag in trueOrFalse_list:
        if flag:
            rest_trueOrFalse_list.append(False)
        else:
            rest_trueOrFalse_list.append(True)
    rest_df = df[rest_trueOrFalse_list]

    # 保存数据
    save_dir = "exp_data/sport/sampling/num/our/pseudo/score_e_3"
    pseudo_df.to_csv(os.path.join(save_dir, "pseudo_df.csv"), index=False)
    rest_df.to_csv(os.path.join(save_dir, "rest_df.csv"), index=False)
    print("getPsedo_df_and_rest_df() success")
    return pseudo_df, rest_df

def generateQueue():
    # 声明出队列
    queue_agree  = queue.PriorityQueue()
    queue_adversary = queue.PriorityQueue()
    queue_adv_A_ex_B = queue.PriorityQueue()
    queue_adv_B_ex_A = queue.PriorityQueue()
    queue_exception = queue.PriorityQueue()
    # 加载侯选池df
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv"
    df = pd.read_csv(csv_path)
    # 遍历每个样本的blackInfo, 根据他们被model_A,model_B 预测出的label 分类到相应的队列
    for row_index, row in df.iterrows():
        merged_idx = row["merged_idx"]

        predic_label_model_A = row["predic_label_model_A"]
        confidence_model_A = row["confidence_model_A"]
        predic_IsOverlap_model_A = row["predic_IsOverlap_model_A"]

        predic_label_model_B = row["predic_label_model_B"]
        confidence_model_B = row["confidence_model_B"]
        predic_IsOverlap_model_B = row["predic_IsOverlap_model_B"]

        cur_dif = abs(confidence_model_A-confidence_model_B) # dif越小优先级越高，双方对这个样本的同意度越近
        if predic_label_model_A == predic_label_model_B:
            queue_agree.put([cur_dif, merged_idx])
        elif predic_IsOverlap_model_A == 1 and predic_IsOverlap_model_B == 1:
            queue_adversary.put([cur_dif, merged_idx]) # dif越小优先级越高，双方对这个样本都自信自己是对的，dif较小
        elif predic_IsOverlap_model_A == 0 and  predic_IsOverlap_model_B == 1:
            queue_adv_A_ex_B.put([cur_dif, merged_idx])
        elif predic_IsOverlap_model_A == 1 and  predic_IsOverlap_model_B == 0:
            queue_adv_B_ex_A.put([cur_dif, merged_idx])
        elif predic_IsOverlap_model_A == 0 and  predic_IsOverlap_model_B == 0:
            queue_exception.put([cur_dif, merged_idx])
        else:
            raise Exception("预测是否是overlap统计数据出错")
    print("generateQueue() success")
    return queue_agree, queue_adversary, queue_adv_A_ex_B, queue_adv_B_ex_A, queue_exception

def generatePseudoQueue():
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv"
    df = pd.read_csv(csv_path)
    queue_score_4  = queue.PriorityQueue()
    queue_score_3 = queue.PriorityQueue()
    queue_score_2 = queue.PriorityQueue()
    queue_score_0 = queue.PriorityQueue()
    priority = 0
    for row_idx, row in df.iterrows():
        confidence_A = row["confidence_model_A"]
        confidence_B = row["confidence_model_B"]
        confidence_Combination = row["confidence_model_Combination"]
        merged_idx = row["merged_idx"]
        if row["pseudo_label_score"] == 3:
            if row["predic_label_model_A"] == row["predic_label_model_Combination"]:
                priority = round(max(confidence_A, confidence_Combination) - confidence_B,4)
            elif row["predic_label_model_B"] == row["predic_label_model_Combination"]:
                priority = round(max(confidence_B, confidence_Combination) - confidence_A,4)
            else:
                raise Exception("3分伪标记有误")
            priority = -priority # 越大优先级越高
            queue_score_3.put([priority, merged_idx])
        elif row["pseudo_label_score"] == 4:
            priority = confidence_Combination
            priority = -priority # 越大优先级越高
            queue_score_4.put([priority, merged_idx])
        elif row["pseudo_label_score"] == 2:
            priority = round(max(confidence_A, confidence_B) - confidence_Combination,4)
            priority = -priority # 越大优先级越高
            queue_score_2.put([priority, merged_idx])
        elif row["pseudo_label_score"] == 0:
            priority = confidence_Combination
            priority = -priority # 越大优先级越高
            queue_score_0.put([priority, merged_idx])
        else:
            raise Exception("伪标记有误")
    print("generatePseudoQueue() success")
    return queue_score_4, queue_score_3, queue_score_2, queue_score_0

def queueToList(cur_queue):
    ans = []
    while not cur_queue.empty():
        cur_dif, merged_idx = cur_queue.get()
        ans.append(merged_idx)
    return ans

def getBlackInfo():
    # 加载模型(各方的extended model)
    model = load_model("/data/mml/overlap_v2_datasets/weather/merged_model/model_B_extended.h5")
    # 加载侯选池df merged_withPredic_withPredicOverlap_df.csv
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_df.csv"
    df = pd.read_csv(csv_path)    
    # 生成器
    generator = ImageDataGenerator(rescale=1./255) # 具体要去model_prepare中去看 model评估时是如何设置的
    # 生成batches
    target_size=(256,256)
    batch_size=32
    # 全局类别,字典序列
    classes = df["label"].unique()
    classes = np.sort(classes).tolist()
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            x_col='file_path', y_col='label', 
                                            target_size=target_size, class_mode='categorical',
                                            color_mode='rgb', classes = classes, 
                                            shuffle=False,  # importent
                                            batch_size=batch_size,
                                            validate_filenames=False
                                            )
    # 每个样本的CNN黑盒值
    predict_label_list = []
    confidence_list = []
    probability_vector_list = []
    # 模型看样本
    predict_array = model.predict_generator(batches, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    for item in predict_array:
        predict_label_list.append(np.argmax(item))
        confidence_list.append(np.max(item))
        probability_vector_list.append(item)
    # 将黑盒信息拓展到csv中
    predict_label_series=pd.Series(predict_label_list, name="predic_label_model_B")
    confidence_series=pd.Series(confidence_list, name="confidence_model_B")
    probability_vector_series=pd.Series(probability_vector_list, name="probability_vector_model_B")
    df = pd.concat([df, predict_label_series, confidence_series,  probability_vector_series], axis=1)
    save_dir = "/data/mml/overlap_v2_datasets/weather/merged_data/test"
    save_path = os.path.join(save_dir, "merged_withPredic_df.csv")
    df.to_csv(save_path, index=False)
    print("getBlackInfo() success")

def getCombinationModelBlackInfo():
    # 加载合并模型
    model_path = "/data/mml/overlap_v2_datasets/weather/merged_model/combination_model_inheritWeights.h5"
    model = load_model(model_path)
    # 加载侯选池df
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_df.csv"
    df = pd.read_csv(csv_path)
    # 训练集的合并df,为了得到class_name_list
    merged_csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/train/merged_df.csv"
    # A,B双方的数据生成器-测试集
    # flower
    # generator_left_test = ImageDataGenerator(rescale = 1./255)  # 归一化
    # generator_right_test = ImageDataGenerator(rescale = 1./255)  # 归一化

    # food
    # generator_left_test =ImageDataGenerator() 
    # generator_right_test = ImageDataGenerator(rescale = 1./255)

    # Fruit
    # generator_left_test = ImageDataGenerator(rescale = 1./255)
    # generator_right_test = ImageDataGenerator(rescale = 1./255)

    # sport
    # generator_left_test = ImageDataGenerator()
    # generator_right_test = ImageDataGenerator(rescale = 1./255)

    # car_body_style
    # generator_left_test =ImageDataGenerator(rescale=1./255) 
    # generator_right_test =ImageDataGenerator(rescale=1./255) 

    # animal
    # generator_left_test =ImageDataGenerator(rescale=1./255) 
    # generator_right_test =ImageDataGenerator(rescale=1./255) 

    #weather
    generator_left_test =ImageDataGenerator() 
    generator_right_test =ImageDataGenerator(rescale=1./255) 


    # 全局类别,使用字典序
    merged_df = pd.read_csv(merged_csv_path)
    classes = merged_df["label"].unique()
    classes = np.sort(classes).tolist()
    batch_size = 32
    target_size_A = (100,100) # flower (224,224), food:(256,256), Fruit:(224,224), sport:(224,224), car_body_style:(256,256), animal:(224,224), weather:(100,100)
    target_size_B = (256,256) # flower (150,150), food:(224,224), Fruit:(224,224), sport:(224,224), car_body_style:(224,224), animal:(150,150), weather:(256,256)
    
    # A,B双方的batches 测试集
    batches_A_test = generator_left_test.flow_from_dataframe(df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col='label', 
                                                    target_size=target_size_A, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size,
                                                    validate_filenames=False)
    batches_B_test = generator_right_test.flow_from_dataframe(df, 
                                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                    # subset="training",
                                                    seed=42,
                                                    x_col='file_path', y_col='label', 
                                                    target_size=target_size_B, class_mode='categorical',
                                                    color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size,
                                                    validate_filenames=False)

    batches = generate_generator_multiple(batches_A_test, batches_B_test)

        # 每个样本的CNN黑盒值
    predict_label_list = []
    confidence_list = []
    probability_vector_list = []
    # 模型看样本
    predict_array = model.predict_generator(batches, steps= batches_A_test.n / batch_size, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    for item in predict_array:
        predict_label_list.append(np.argmax(item))
        confidence_list.append(np.max(item))
        probability_vector_list.append(item)
    # 将黑盒信息拓展到csv中
    predict_label_series=pd.Series(predict_label_list, name="predic_label_model_Combination")
    confidence_series=pd.Series(confidence_list, name="confidence_model_Combination")
    probability_vector_series=pd.Series(probability_vector_list, name="probability_vector_model_Combination")
    df = pd.concat([df, predict_label_series, confidence_series,  probability_vector_series], axis=1)
    save_dir = "/data/mml/overlap_v2_datasets/weather/merged_data/test"
    save_path = os.path.join(save_dir, "merged_withPredic_df.csv")
    df.to_csv(save_path, index=False)
    print("getCombinationModelBlackInfo() success")

def generate_generator_multiple(batches_A, batches_B):
    '''
    将连个模型的输入bath 同时返回
    '''
    while True:
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def isPredicOverlap():
    # 加载侯选池df
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_df.csv"
    df = pd.read_csv(csv_path)
    # 双方的local to global
    local_to_global_party_A = joblib.load("exp_data/weather/LocalToGlobal/local_to_gobal_party_A.data")
    local_to_global_party_B = joblib.load("exp_data/weather/LocalToGlobal/local_to_gobal_party_B.data")
    # 得到global overlap label index
    overlapGlobalLabelIndex_list = getOverlapGlobalLabelIndex(local_to_global_party_A, local_to_global_party_B)
    
    predic_IsOverlap_model_A_list = []
    predic_IsOverlap_model_B_list = []
    predic_IsOverlap_model_Combination_list = []
    # 遍历每个样本的blackInfo
    for row_index, row in df.iterrows():
        predic_label_model_A = row["predic_label_model_A"]
        predic_label_model_B = row["predic_label_model_B"]
        predic_label_model_Combination = row["predic_label_model_Combination"]
        if predic_label_model_A in overlapGlobalLabelIndex_list:
            predic_IsOverlap_model_A_list.append(1)
        else:
            predic_IsOverlap_model_A_list.append(0)

        if predic_label_model_B in overlapGlobalLabelIndex_list:
            predic_IsOverlap_model_B_list.append(1)
        else:
            predic_IsOverlap_model_B_list.append(0)

        if predic_label_model_Combination in overlapGlobalLabelIndex_list:
            predic_IsOverlap_model_Combination_list.append(1)
        else:
            predic_IsOverlap_model_Combination_list.append(0)
    predic_IsOverlap_model_A_series=pd.Series(predic_IsOverlap_model_A_list, name="predic_IsOverlap_model_A")
    predic_IsOverlap_model_B_series=pd.Series(predic_IsOverlap_model_B_list, name="predic_IsOverlap_model_B")
    predic_IsOverlap_model_Combination_series=pd.Series(predic_IsOverlap_model_Combination_list, name="predic_IsOverlap_model_Combination")
    df = pd.concat([df, predic_IsOverlap_model_A_series, predic_IsOverlap_model_B_series, predic_IsOverlap_model_Combination_series], axis=1)
    save_dir = "/data/mml/overlap_v2_datasets/weather/merged_data/test"
    save_path = os.path.join(save_dir, "merged_withPredic_df.csv")
    df.to_csv(save_path, index=False)
    print("isPredicOverlap() success")

def generatorPseudoInfo():
    '''
    model_A, model_B, combination_model 打伪标签
    '''
    # 加载csv
    csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_df.csv"
    df = pd.read_csv(csv_path)
    merged_csv_path = "/data/mml/overlap_v2_datasets/weather/merged_data/train/merged_df.csv"
    merged_df = pd.read_csv(merged_csv_path)
    classes = merged_df["label"].unique()
    classes_list = np.sort(classes).tolist() # importent global字典序
    pseudo_label_list = []
    pseudo_GlobalLabelIndex_list = []
    pseudo_label_score_list = []
    for row_index, row in df.iterrows():
        # 获得三个模型的 pseudo_globalLabel_index
        predic_label_model_A = row["predic_label_model_A"]
        predic_label_model_B = row["predic_label_model_B"]
        predic_label_model_Combination = row["predic_label_model_Combination"]
        # 根据一致情况打分, combination_model 有2分
        if predic_label_model_A == predic_label_model_B and predic_label_model_A == predic_label_model_Combination:
            pseudo_label_score = 4
            pseudo_globalLabelIndex = predic_label_model_Combination
            pseudo_label = classes_list[pseudo_globalLabelIndex]
        elif (predic_label_model_A == predic_label_model_Combination) or (predic_label_model_B == predic_label_model_Combination):
            pseudo_label_score = 3
            pseudo_globalLabelIndex = predic_label_model_Combination
            pseudo_label = classes_list[pseudo_globalLabelIndex]
        elif predic_label_model_A == predic_label_model_B:
            pseudo_label_score = 2
            pseudo_globalLabelIndex = predic_label_model_A
            pseudo_label = classes_list[pseudo_globalLabelIndex]
        else:
            pseudo_label_score = 0
            pseudo_globalLabelIndex = predic_label_model_Combination
            pseudo_label = classes_list[pseudo_globalLabelIndex]

        pseudo_label_list.append(pseudo_label)
        pseudo_GlobalLabelIndex_list.append(pseudo_globalLabelIndex)
        pseudo_label_score_list.append(pseudo_label_score)

    pseudo_label_series=pd.Series(pseudo_label_list, name="pseudo_label")
    pseudo_GlobalLabelIndex_series=pd.Series(pseudo_GlobalLabelIndex_list, name="pseudo_GlobalLabelIndex")
    pseudo_label_score_series=pd.Series(pseudo_label_score_list, name="pseudo_label_score")
    df = pd.concat([df, pseudo_label_series, pseudo_GlobalLabelIndex_series, pseudo_label_score_series], axis=1)
    save_dir = "/data/mml/overlap_v2_datasets/weather/merged_data/test"
    save_path = os.path.join(save_dir, "merged_withPredic_withPredicOverlap_Pseudo_df.csv")
    df.to_csv(save_path, index=False)
    print("generatorPseudoInfo() success")

if __name__ == "__main__":

    '''
    第一步,得到样本的黑盒信息
    '''
    getBlackInfo()
    # getCombinationModelBlackInfo()

    '''
    第二步,得到各个model预测是否是Overlap区域
    '''
    # isPredicOverlap()

    '''
    第三步, 生成伪标注信息
    '''
    # generatorPseudoInfo()
    # getPsedo_df_and_rest_df()

    '''
    第三步,得到队列数据
    '''
    # queue_agree, queue_adversary, queue_adv_A_ex_B, queue_adv_B_ex_A, queue_exception = generateQueue()
    # save_dir = "exp_data/weather/sampling/queueData"
    # saveData(queue_agree, os.path.join(save_dir, "queue_agree.data"))
    # saveData(queue_adversary, os.path.join(save_dir, "queue_adversary.data"))
    # saveData(queue_adv_A_ex_B, os.path.join(save_dir, "queue_adv_A_ex_B.data"))
    # saveData(queue_adv_B_ex_A, os.path.join(save_dir, "queue_adv_B_ex_A.data"))
    # saveData(queue_exception, os.path.join(save_dir, "queue_exception.data"))

    '''
    第三步的结果验证
    '''
    # save_dir = "exp_data/weather/sampling/queueData"
    # queue_agree = joblib.load(os.path.join(save_dir, "queue_agree.data"))
    # queue_adversary = joblib.load(os.path.join(save_dir, "queue_adversary.data"))
    # queue_adv_A_ex_B = joblib.load(os.path.join(save_dir, "queue_adv_A_ex_B.data"))
    # queue_adv_B_ex_A = joblib.load(os.path.join(save_dir, "queue_adv_B_ex_A.data"))
    # queue_exception = joblib.load(os.path.join(save_dir, "queue_exception.data"))
    # total = queue_agree.qsize()+queue_adversary.qsize()+queue_adv_A_ex_B.qsize()+queue_adv_B_ex_A.qsize()+queue_exception.qsize()
    # df = pd.read_csv("/data/mml/overlap_v2_datasets/weather/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
    # print("queue total size: {}, 候选集size: {}".format(total, df.shape[0]))
    
    '''
    形成伪标记分数队列
    '''
    # queue_score_4, queue_score_3, queue_score_2, queue_score_0 = generatePseudoQueue()
    # save_dir = "exp_data/weather/sampling/pseudo"
    # saveData(queue_score_4, os.path.join(save_dir, "queue_score_4.data"))
    # saveData(queue_score_3, os.path.join(save_dir, "queue_score_3.data"))
    # saveData(queue_score_2, os.path.join(save_dir, "queue_score_2.data"))
    # saveData(queue_score_0, os.path.join(save_dir, "queue_score_0.data"))
    '''
    开始采样
    '''
    start_sampling()


    pass    

