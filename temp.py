from collections import defaultdict
import numpy as np
import random
import heapq
import multiprocessing
import pandas as pd
import os
import math
import joblib
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model, Sequential, load_model
from matplotlib import pyplot as plt
# 查询当前系统所有字体
from matplotlib.font_manager import FontManager
import subprocess
import mplcyberpunk

def dawline():
    def std(x):
        y = x*x
        return y
    def real(x):
        e = random.uniform(-50, +50)
        y = x*x + e
        return y
    
    x_list = np.arange(0, 20, 0.2)
    std_list = []
    real_list = []
    x_tick = []
    for x in x_list:
        std_list.append(std(x))
        real_list.append(real(x))
        x_tick.append("")
    plt.style.use("cyberpunk")
    plt.plot(x_list, std_list, label = "Perfect Love")
    plt.plot(x_list, real_list, label = "Our Love")
    # mplcyberpunk.make_lines_glow()
    # 图例
    plt.legend()
    # 坐标轴说明
    plt.xlabel("Time")
    plt.ylabel("Love")
    
    plt.axes().set_xticklabels(x_tick)
    plt.axes().set_yticklabels(x_tick)
    plt.show()

def test():
    df = pd.read_csv("/data/mml/overlap_v2_datasets/animal_2/merged_data/train/merged_df.csv")
    print()
    # unique_df = df[df["is_overlap"] == 0]
if __name__ == "__main__":
    test()
    # dawline()
# a = pd.read_csv("exp_data/all/spearman_corr.csv")
# a.to_excel("exp_data/all/a.xlsx")

# car_A = pd.read_csv("/data/mml/overlap_v2_datasets/car_body_style/party_A/dataset_split/train.csv")
# classes_A  = car_A["label"].unique()
# classes_A = np.sort(classes_A).tolist()
# car_B = pd.read_csv("/data/mml/overlap_v2_datasets/car_body_style/party_B/dataset_split/train.csv")
# classes_B  = car_B["label"].unique()
# classes_B = np.sort(classes_B).tolist()
# overlap = pd.read_csv("/data/mml/overlap_v2_datasets/car_body_style/merged_data/train/merged_df_overlap.csv")
# classes_o  = overlap["label"].unique()
# classes_o = np.sort(classes_o).tolist()

# print("")
# pre_trained_model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# a = np.array(
#     [
#         [1,1,1],
#         [3,3,3],
#         [4,5,6]
#     ]
# )

# b = np.array(
#     [
#         [2,2,2],
#         [0,0,0],
#         [3,6,9]
#     ]
# )

# c = np.max([a,b],axis=0)

# C = np.array([[0,0,0]])
# A = np.arange(6).reshape(-1,3)

# d = np.concatenate((C,A),axis=0)
# print("")
# data = [
#     {"id":0, "label":"dog", "is_overlap":1, "source":1},
#     {"id":1, "label":"cat", "is_overlap":0, "source":1},
#     {"id":2, "label":"pig", "is_overlap":0, "source":1},
#     {"id":3, "label":"dog", "is_overlap":1, "source":2},
#     {"id":4, "label":"tigger", "is_overlap":0, "source":2},
#     {"id":5, "label":"elephant", "is_overlap":0, "source":2},
# ]
# df = pd.DataFrame(data)
# df["is_overlap"]==1
# print()



# data = defaultdict(np.array)
# data[("mml", 1)] = np.array([1,2,3])


# p = np.random.permutation(5)
# print(p)

# dataset =  np.array([[1,1,1], [2,2,2], [3,3,3]])
# subset = dataset[np.array([0,2]),:]
# print(subset)

# temp_proba = np.array([0.2,0.3,0.3,0.2,0.03])
# sampled_num = 10
# temp_proba[temp_proba < (.5 / sampled_num)] = 0


# a = [3,4,2,9,0]
# idex = np.argmax(a)
# print("faf")

# a = np.array([5,7,9])
# a_list = a.tolist()
# print("faf")

# b = np.array([5,7,9,8,5,2,1,0,8,7,3])
# data = random.sample(list(b), 4)
# print("jfla")

# b = np.array([5,7,9,8,5,2,1,0,8,7,3])
# c = np.array([5,7,9])
# diff_array = np.setdiff1d(b, c)
# print("fjlakjd")

# arr = np.random.permutation(5)
# print("fjla")

# prob_list = [1,2,3,4,5,6]
# data = heapq.nlargest(2,prob_list)
# print("jflkadj")

# a = -(3-5)
# print("fjal")

# a = 3.7353403
# b = round(a)
# print("ljfal;s")

# c = 8 * (5/3)
# print(c)

# a = [9,9,9]
# b = np.array(a)-3
# print("fjal")

# for _ in range(10):
#     print("1")
# print("jfla")

# d = np.sort(["bbc", "abc"])
# print("fasf")

# df = pd.read_csv("/data/mml/overlap_v2_datasets/food/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
# data = np.sort(df["label_globalIndex"].unique()).tolist()
# print("fajladf")

# a = 1
# b  = 2
# print(a != b)
# print("fa")

# df = pd.read_csv("/data/mml/overlap_v2_datasets/food/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
# new_df = df[(df["source"] == 1) & (df["is_overlap"] == 0)]
# print("fjal")

# df = pd.read_csv("/data/mml/overlap_v2_datasets/food/merged_data/test/merged_df.csv")
# row_0 = df.iloc[0]
# row_1 = df.iloc[1]
# row_2 = df.iloc[2]
# rows = [row_0,row_1,row_2]
# df_new = pd.DataFrame(rows) # columns = ["merged_idx","logic_id"]
# print("faf")

# a = set([0,1,2])
# b = set([0,1,2,3])
# c = a-b
# print(c)

# a = np.zeros((5,4,3))
# count = 1
# for i in range(5):
#     for j in range(4):
#         for k in range(3):
#             a[i][j][k] = count
#             count += 1
# print(a)
# b = np.max(a,axis = 0)


# c = [[1,2,3],[8,4,5],[6,7,9]]
# d = np.max(c,axis = 0)
# print("jfla")

# q = np.array([1,2,3,4,5])
# x = np.array([1,3,4,4,5])
# d = np.sum(q == x)
# print("fjal")

# a = [[1,2,3],[4,5,6]]
# b = a[1]
# b.append(0)
# print(a)

# d = math.ceil(3.25)
# print(d)

# data = [[0,4,7], [1,3,6]]
# np.max(data,axis=0)


