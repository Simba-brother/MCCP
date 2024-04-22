import sys
sys.path.append("./")

from collections import defaultdict

from tensorflow.keras.models import load_model
import pandas as pd
import os
import sys
import numpy as np
import joblib

from DataSetConfig import car_body_style_config,flower_2_config, food_config, fruit_config,sport_config,weather_config, animal_config, animal_2_config, animal_3_config
from Jaccard_exp.code.utils import getClasses



def my_perdict(config,domain):
    '''
    对数据进行预测，并保存结果
    '''
    # 加载model结构
    model = load_model(config["model_struct_path"])
    if not config["model_weight_path"] is None:
        # 加载model权重
        model.load_weights(config["model_weight_path"])
    # 加载待预测数据集
    df = pd.read_csv(config["csv_path"])
    batch_size = 16
    gen = config["generator_test"]
    classes= getClasses(config["dataset_train_path"]) # sorted
    target_size = config["target_size"]
    # 样本batch
    batches = gen.flow_from_dataframe(
                                    df, 
                                    directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                    x_col='file_path', y_col='label', 
                                    target_size=target_size, class_mode='categorical',
                                    color_mode='rgb', classes = classes, 
                                    shuffle=False, batch_size=batch_size,
                                    validate_filenames=False
                                    )
    print(batches.class_indices)
    print("评估集样本数: {}".format(df.shape[0]))
    # 开始评估
    predicts =  model.predict(x=batches,steps=batches.n / batch_size, verbose = 1)
    local_labels = np.argmax(predicts, axis=1)
    local_to_global = joblib.load(config["local_to_global_path"])
    global_label_list = []
    for local_label in local_labels:
        global_label_list.append(local_to_global[local_label])
    df[config["col_name"]] = global_label_list
    save_dir = "complement_exp/result/"
    file_name = f"{domain}_predic_overlap.csv"
    file_path = os.path.join(save_dir, file_name)
    df.to_csv(file_path, index=False)
    print(f"result is saved in {file_path}")

def caculate_Jaccard(domain:str):
    df = pd.read_csv(f"complement_exp/result/{domain}_predic_overlap.csv")
    predict_global_label_A_list = list(df["predict_global_label_A"])
    predict_global_label_B_list = list(df["predict_global_label_B"])

    count_fenzi_dic = defaultdict(int)
    count_fenmu_dic = defaultdict(int)
    for l_A, l_B in zip(predict_global_label_A_list, predict_global_label_B_list):
        if l_A == l_B:
            count_fenzi_dic[l_A] += 1
            count_fenmu_dic[l_A] += 1
        else:
            count_fenmu_dic[l_A] += 1
            count_fenmu_dic[l_B] += 1
    ans = 0        
    cur_sum = 0
    n0 = 0
    for label,fenzi in count_fenzi_dic.items():
        n0 += 1
        fenmu = count_fenmu_dic[label]
        cur_sum += fenzi/fenmu
    ans = round(cur_sum / n0,2)
    print(ans)

if __name__ == "__main__":
    domain = "animal_3"
    config = animal_3_config
    config_A = {
        "model_struct_path":config["model_A_struct_path"],
        "model_weight_path":config["model_A_weight_path"],
        "merged_overlap_df":config["merged_overlap_df"],
        "generator_test":config["generator_A_test"],
        "dataset_train_path":config["dataset_A_train_path"],
        "target_size":config["target_size_A"],
        "local_to_global_path":config["local_to_global_party_A_path"],
        # specific
        "col_name":"predict_global_label_A",
        "csv_path":config["merged_overlap_df"]
    }
    config_B = {
        "model_struct_path":config["model_B_struct_path"],
        "model_weight_path":config["model_B_weight_path"],
        "merged_overlap_df":config["merged_overlap_df"],
        "generator_test":config["generator_B_test"],
        "dataset_train_path":config["dataset_B_train_path"],
        "target_size":config["target_size_B"],
        "local_to_global_path":config["local_to_global_party_B_path"],
        # specific
        "col_name":"predict_global_label_B",
        "csv_path":f"complement_exp/result/{domain}_predic_overlap.csv"
    }
    # my_perdict(config_B,domain=domain)
    
    domain_list = ["car", "flower", "food", "fruit", "sport", "weather", "animal_1", "animal_2","animal_3"]
    for d in domain_list:
        caculate_Jaccard(d)
