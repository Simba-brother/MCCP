from fileinput import filename
import pandas as pd
import os
import sys
sys.path.append("/home/mml/workspace/model_reuse_v2/")
from utils import deleteIgnoreFile, saveData

def random_samping(csv_path, num):
    '''
    从dataFrame中随机采样 
    '''
    merged_df = pd.read_csv(csv_path)
    # 随机采样（作为重训练集）
    sampled_df = merged_df.sample(n=num, axis=0) # random_state=123
    rest_df = None
    return sampled_df, rest_df

def random_sampling_misAgree(csv_path, num):
    '''
    从dataFrame中随机采样(剔除掉agree) 
    '''
    merged_df = pd.read_csv(csv_path)
    # 随机采样（(剔除掉agree) 作为重训练集）
    trueOrFalse_list = []
    for row_index, row in merged_df.iterrows():
        if row["predic_label_model_A"] != row["predic_label_model_B"]:
            trueOrFalse_list.append(True)
        else:
            trueOrFalse_list.append(False)

    merged_df_disAgree = merged_df[trueOrFalse_list]
    sampled_df = merged_df_disAgree.sample(n=num, axis=0) # random_state=123
    rest_df = None
    return sampled_df, rest_df

def random_sampling_unique(csv_path, num):
    '''
    从dataFrame中随机采样(剔除掉Overlap) 
    '''
    merged_df = pd.read_csv(csv_path)
    trueOrFalse_list = []
    for row_index, row in merged_df.iterrows():
        if row["is_overlap"] == 1: # 是 overlap的剔除
            trueOrFalse_list.append(False)
        else:
            trueOrFalse_list.append(True)

    merged_df_unique = merged_df[trueOrFalse_list]
    sampled_df = merged_df_unique.sample(n=num, axis=0) # random_state=123
    rest_df = None
    return sampled_df, rest_df

def updateCSVFromNewCSV(csv_path_small, csv_path_big):
    '''
    根据 csv_path_small 中的 merged_idx 检索出来 csv_path_big 那行 
    '''
    df_small = pd.read_csv(csv_path_small)
    df_big = pd.read_csv(csv_path_big)
    # 待检索的merged_idx
    merged_idx_list = list(df_small["merged_idx"])
    # 开始检索
    trueOrFalse_list = [ True if i in merged_idx_list else False for i in df_big["merged_idx"] ]
    # 得出检索df
    new_df_small = df_big[trueOrFalse_list]
    return new_df_small

def delAgree(csv_path):
    df = pd.read_csv(csv_path)
    # 选择flag_list
    trueOrFalse_list = []
    for row_index, row in df.iterrows():
        if row["predic_label_model_A"] != row["predic_label_model_B"]:
            trueOrFalse_list.append(True)
        else:
            trueOrFalse_list.append(False)
    df_disAgress = df[trueOrFalse_list]
    return df_disAgress

def getSampling_num(csv_path, percent):
    df = pd.read_csv(csv_path)
    return int(df.shape[0]*percent)



if __name__ == '__main__':
    # 混合集csv_path
    csv_path = "/data/mml/overlap_v2_datasets/animal_3/merged_data/test/merged_df.csv"
    # 重复实验次数
    repeat_num = 10
    # 采样数量
    # sampling_num_list = [30,60,90,120,150,180]
    sampling_percent_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0]
    # 保存路径
    base_dir = "exp_data/animal_3/sampling/percent/random"
    for repeat in range(repeat_num):
        for sampling_percent in sampling_percent_list:
        # for sampling_num in sampling_num_list:
            sampling_num = getSampling_num(csv_path, sampling_percent)
            sampled_df, rest_df = random_samping(csv_path, sampling_num)
            # 保存路径
            save_dir = os.path.join(base_dir, str(int(sampling_percent*100))) # str(int(sampling_percent*100)), str(sampling_num)
            # 文件名
            file_name = "sampled_"+str(repeat)+".csv"
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            sampled_df.to_csv(os.path.join(save_dir, file_name), index=False)
    print("random sampling success")



    # ----------------华丽分割线----------------------------
    '''
    根据原来采样df的merged_idx, 重新采更信息的样
    '''
    # common_dir = "exp_data/Fruit/sampling/random"
    # csv_path_big = "/data/mml/overlap_v2_datasets/Fruit/merged_data/test/merged_withPredic_withPredicOverlap_df.csv"
    # save_dir_base = "exp_data/Fruit/sampling/random_temp"
    # num_dir_list = os.listdir(common_dir)
    # num_dir_list = deleteIgnoreFile(num_dir_list)
    # for num_dir in num_dir_list:
    #     cur_dir = os.path.join(common_dir, num_dir)
    #     file_name_list = os.listdir(cur_dir)
    #     file_name_list = deleteIgnoreFile(file_name_list)
    #     for file_name in file_name_list:
    #         csv_path_small = os.path.join(cur_dir, file_name)
    #         df = updateCSVFromNewCSV(csv_path_small, csv_path_big)
    #         save_dir_cur = os.path.join(save_dir_base, num_dir)
    #         if os.path.exists(save_dir_cur) == False:
    #             os.makedirs(save_dir_cur)
    #         df.to_csv(os.path.join(save_dir_cur, file_name), index=False)
    '''
    将采样中的agree样本去掉
    '''
    # common_dir = "exp_data/flower/sampling/random"
    # save_dir_base = "exp_data/flower/sampling/random_delAgree"
    # num_dir_list = os.listdir(common_dir)
    # num_dir_list = deleteIgnoreFile(num_dir_list)
    # for num_dir in num_dir_list:
    #     cur_dir = os.path.join(common_dir, num_dir)
    #     file_name_list = os.listdir(cur_dir)
    #     file_name_list = deleteIgnoreFile(file_name_list)
    #     for file_name in file_name_list:
    #         csv_path = os.path.join(cur_dir, file_name)
    #         df = delAgree(csv_path)
    #         save_dir_cur = os.path.join(save_dir_base, num_dir)
    #         if os.path.exists(save_dir_cur) == False:
    #             os.makedirs(save_dir_cur)
    #         df.to_csv(os.path.join(save_dir_cur, file_name), index=False)