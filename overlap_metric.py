'''
量化overlap程度的指标,conda环境是MCCP环境
'''
import time
import os
from collections import defaultdict
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from numpy import linalg as LA
from DataSetConfig import exp_dir, car_body_style_config,flower_2_config,food_config,fruit_config, sport_config, weather_config, animal_config, animal_2_config, animal_3_config
import setproctitle



def analysis_similiary_sore(similarity_matrix):
    A_num = similarity_matrix.shape[0]
    B_num = similarity_matrix.shape[1]
    # 以A数据做为库，遍历每个B数据从库中找最大相似度
    A_cover_B = 0
    for B_i in range(B_num):
       A_cover_B += max(similarity_matrix[:,B_i])
    A_cover_B = round(A_cover_B / B_num,4)
    # 以B数据做为库，遍历每个A数据从库中找最大相似度
    B_cover_A = 0
    for A_i in range(A_num):
       B_cover_A += max(similarity_matrix[A_i,:])
    B_cover_A = round(B_cover_A / A_num,4)

    res = {}
    res["A_cover_B"] = A_cover_B
    res["B_cover_A"] = B_cover_A
    return res


def cal_similiary():
    '''
    根据模型输出的相似性
    '''
    data_path = os.path.join(exp_dir, config["dataset_name"], "features.data")
    features = joblib.load(data_path)
    AA_features = features["AA_features"]
    AB_features = features["AB_features"]
    BA_features = features["BA_features"]
    BB_features = features["BB_features"]
    A_num,_ = AA_features.shape
    B_num,_ = AB_features.shape
    similiary_array_A = np.zeros((A_num,B_num))
    for i in range(A_num):
        for j in range(B_num):
            score = calcu_cosine_similarity(AA_features[i], AB_features[j])
            similiary_array_A[i][j] = score

    similiary_array_B = np.zeros((A_num,B_num))
    for i in range(A_num):
        for j in range(B_num):
            score = calcu_cosine_similarity(BA_features[i], BB_features[j])
            similiary_array_B[i][j] = score

    A_cover_B = 0
    for B_i in range(B_num):
        A_cover_B += max(similiary_array_A[:,B_i])
    A_cover_B  = round(A_cover_B/B_num,4)

    B_cover_A = 0
    for A_i in range(A_num):
        B_cover_A += max(similiary_array_B[A_i,:])
    B_cover_A  = round(B_cover_A/A_num,4)

    print("A_cover_B:",A_cover_B)
    print("B_cover_A:",B_cover_A)


def cal_similarity_by_group():
    '''
    以每个overlap label为一组，分组计算cosine similarity
    '''

    proc_title = f"{config['dataset_name']}|similarity_group"
    print(proc_title)
    setproctitle.setproctitle(proc_title)
    start_time = time.perf_counter()
    # 加载数据集
    testset_merged_overlap_df = pd.read_csv(config["merged_overlap_df"])
    df_A_val_overlap = testset_merged_overlap_df[testset_merged_overlap_df["source"]==1]
    df_B_val_overlap = testset_merged_overlap_df[testset_merged_overlap_df["source"]==2]

    A_label_globalIndex_list = df_A_val_overlap["label_globalIndex"].to_list()
    B_label_globalIndex_list = df_B_val_overlap["label_globalIndex"].to_list()
    global_labelIndex_set = set(A_label_globalIndex_list)

    group_dict = {}
    for group_label in global_labelIndex_set:
        group_dict[group_label] = defaultdict(list)

        A_indices_of_value = [index for index, element in enumerate(A_label_globalIndex_list) if element == group_label]
        for i in A_indices_of_value:
            group_dict[group_label]["A"].append(os.path.join(exp_dir,df_A_val_overlap.iloc[i]["file_path"]))

        B_indices_of_value = [index for index, element in enumerate(B_label_globalIndex_list) if element == group_label]
        for i in B_indices_of_value:
            group_dict[group_label]["B"].append(os.path.join(exp_dir,df_B_val_overlap.iloc[i]["file_path"]))

    model = SentenceTransformer('clip-ViT-B-32')
    group_ans = {}
    for group_label in group_dict:
        A_img_file_path_list = group_dict[group_label]["A"]
        B_img_file_path_list = group_dict[group_label]["B"]
        matrix = np.zeros((len(A_img_file_path_list),len(B_img_file_path_list)))
        for i, A_img_file_path in  enumerate(A_img_file_path_list):
            for j, B_img_file_path in  enumerate(B_img_file_path_list):
                img_A_emb = model.encode(Image.open(A_img_file_path))
                img_B_emb = model.encode(Image.open(B_img_file_path))
                similarity_score = model.similarity(img_A_emb, img_B_emb)
                matrix[i][j] = similarity_score[0][0].item()
        
        score_dic = analysis_similiary_sore(matrix)
        A_cover_B = score_dic["A_cover_B"]
        B_cover_A = score_dic["B_cover_A"]
        avg_all =  matrix.mean()
        avg = round((A_cover_B+B_cover_A)/2,4)

        group_ans[group_label] = {
            "A_cover_B":A_cover_B,
            "B_cover_A":B_cover_A,
            "avg_all":avg_all,
            "avg":avg
        }

    sum_A_cover_B = 0
    sum_B_cover_A = 0
    sum_avg_all = 0
    sum_avg = 0
    count = 0
    for group_label in group_ans.keys():
        sum_A_cover_B += group_ans[group_label]["A_cover_B"]
        sum_B_cover_A += group_ans[group_label]["B_cover_A"]
        sum_avg_all += group_ans[group_label]["avg_all"]
        sum_avg += group_ans[group_label]["avg"]
        count += 1
    avg_A_cover_B  = round(sum_A_cover_B/count,4)
    avg_B_cover_A  = round(sum_B_cover_A/count,4)
    avg_avg_all = round(sum_avg_all/count, 4)
    avg_avg = round(sum_avg/count, 4)
    
    ans = {}
    ans["avg_A_cover_B"] = avg_A_cover_B
    ans["avg_B_cover_A"] = avg_B_cover_A
    ans["avg_avg_all"] = avg_avg_all
    ans["avg_avg"] = avg_avg
    print(ans)
    end_time = time.perf_counter()
    cost_time = end_time - start_time
    print("cost_time",cost_time)
    
def calcu_cosine_similarity(feature_1, feature_2):
    similarity = cosine_similarity([feature_1], [feature_2])
    return similarity[0][0]

def calcu_euclidean(feature_1, feature_2):
    distances = euclidean_distances([feature_1], [feature_2])
    return distances[0][0]



if __name__ == "__main__":
    config = animal_3_config
    cal_similarity_by_group()
    
    # save_dir = os.path.join("exp_data", config["dataset_name"])
    # save_file_name = "overlap_similarity_score.data"
    # save_file_path = os.path.join(save_dir, save_file_name)
    # joblib.dump(similiary_matrix, save_file_path)
    # print(f"data is saved in:{save_file_path}")

    # similiary_matrix = joblib.load(f"exp_data/{config['dataset_name']}/overlap_similarity_score.data")
    # covered_measure = analysis_similiary_sore(similiary_matrix)
    #print(covered_measure)