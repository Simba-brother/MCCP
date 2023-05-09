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


pre_trained_model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
a = np.array(
    [
        [1,1,1],
        [3,3,3],
        [4,5,6]
    ]
)

b = np.array(
    [
        [2,2,2],
        [0,0,0],
        [3,6,9]
    ]
)

c = np.max([a,b],axis=0)

C = np.array([[0,0,0]])
A = np.arange(6).reshape(-1,3)

d = np.concatenate((C,A),axis=0)
print("")
data = [
    {"id":0, "label":"dog", "is_overlap":1, "source":1},
    {"id":1, "label":"cat", "is_overlap":0, "source":1},
    {"id":2, "label":"pig", "is_overlap":0, "source":1},
    {"id":3, "label":"dog", "is_overlap":1, "source":2},
    {"id":4, "label":"tigger", "is_overlap":0, "source":2},
    {"id":5, "label":"elephant", "is_overlap":0, "source":2},
]
df = pd.DataFrame(data)
df["is_overlap"]==1
print()



data = defaultdict(np.array)
data[("mml", 1)] = np.array([1,2,3])


p = np.random.permutation(5)
print(p)

dataset =  np.array([[1,1,1], [2,2,2], [3,3,3]])
subset = dataset[np.array([0,2]),:]
print(subset)

temp_proba = np.array([0.2,0.3,0.3,0.2,0.03])
sampled_num = 10
temp_proba[temp_proba < (.5 / sampled_num)] = 0


a = [3,4,2,9,0]
idex = np.argmax(a)
print("faf")

a = np.array([5,7,9])
a_list = a.tolist()
print("faf")

b = np.array([5,7,9,8,5,2,1,0,8,7,3])
data = random.sample(list(b), 4)
print("jfla")

b = np.array([5,7,9,8,5,2,1,0,8,7,3])
c = np.array([5,7,9])
diff_array = np.setdiff1d(b, c)
print("fjlakjd")

arr = np.random.permutation(5)
print("fjla")

prob_list = [1,2,3,4,5,6]
data = heapq.nlargest(2,prob_list)
print("jflkadj")

a = -(3-5)
print("fjal")

a = 3.7353403
b = round(a)
print("ljfal;s")

c = 8 * (5/3)
print(c)

a = [9,9,9]
b = np.array(a)-3
print("fjal")

for _ in range(10):
    print("1")
print("jfla")

d = np.sort(["bbc", "abc"])
print("fasf")

df = pd.read_csv("/data/mml/overlap_v2_datasets/food/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
data = np.sort(df["label_globalIndex"].unique()).tolist()
print("fajladf")

a = 1
b  = 2
print(a != b)
print("fa")

df = pd.read_csv("/data/mml/overlap_v2_datasets/food/merged_data/test/merged_withPredic_withPredicOverlap_Pseudo_df.csv")
new_df = df[(df["source"] == 1) & (df["is_overlap"] == 0)]
print("fjal")

df = pd.read_csv("/data/mml/overlap_v2_datasets/food/merged_data/test/merged_df.csv")
row_0 = df.iloc[0]
row_1 = df.iloc[1]
row_2 = df.iloc[2]
rows = [row_0,row_1,row_2]
df_new = pd.DataFrame(rows) # columns = ["merged_idx","logic_id"]
print("faf")

a = set([0,1,2])
b = set([0,1,2,3])
c = a-b
print(c)

a = np.zeros((5,4,3))
count = 1
for i in range(5):
    for j in range(4):
        for k in range(3):
            a[i][j][k] = count
            count += 1
print(a)
b = np.max(a,axis = 0)


c = [[1,2,3],[8,4,5],[6,7,9]]
d = np.max(c,axis = 0)
print("jfla")

q = np.array([1,2,3,4,5])
x = np.array([1,3,4,4,5])
d = np.sum(q == x)
print("fjal")

a = [[1,2,3],[4,5,6]]
b = a[1]
b.append(0)
print(a)

d = math.ceil(3.25)
print(d)

data = [[0,4,7], [1,3,6]]
np.max(data,axis=0)


