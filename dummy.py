from tensorflow.keras.models import Model, Sequential, load_model
import pandas as pd
import numpy as np
import joblib
from utils import deleteIgnoreFile, makedir_help
import os
from DataSetConfig import food_config, fruit_config, sport_config, weather_config, flower_2_config, car_body_style_config, animal_config, animal_2_config, animal_3_config
import Base_acc


def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

def get_confidences(model, df, generator, target_size):
   batch_size = 5
   batches = generator.flow_from_dataframe(df, 
                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                            x_col='file_path', y_col="label", 
                            target_size=target_size, class_mode='categorical', # one-hot
                            color_mode='rgb', classes=None,
                            shuffle=False, batch_size=batch_size,
                            validate_filenames=False)
   probs = model.predict_generator(generator = batches, steps=batches.n/batch_size)
   confidences = np.max(probs, axis = 1)
   pseudo_label_indexes = np.argmax(probs, axis=1)
   return confidences, pseudo_label_indexes

def eval(model, df, generator, target_size, classes):
    batch_size = 5
    batches = generator.flow_from_dataframe(df, 
                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                x_col='file_path', y_col="label", 
                                target_size=target_size, class_mode='categorical', # one-hot
                                color_mode='rgb', classes=classes,
                                shuffle=False, batch_size=batch_size,
                                validate_filenames=False)
    print(batches.class_indices)
    loss, acc = model.evaluate_generator(generator=batches,steps=batches.n/batch_size)
    return acc


def dummy():
    ans = {}
    ans["base_A_acc"] = Base_acc_config["A_acc"]
    ans["base_B_acc"] = Base_acc_config["B_acc"]
    ans["combin_acc"] = None
    pseudo_labels = []
    confidences_A, pseudo_labels_A = get_confidences(model_A, merged_test_df, generator_A_test, target_size_A)
    confidences_B, pseudo_labels_B = get_confidences(model_B, merged_test_df, generator_B_test, target_size_B)
    for i in range(confidences_A.shape[0]):
        if confidences_A[i] > confidences_B[i]:
            pseudo_global_label = local_to_global_party_A[pseudo_labels_A[i]]
            pseudo_labels.append(pseudo_global_label)
            
        else:
            pseudo_global_label = local_to_global_party_B[pseudo_labels_B[i]]
            pseudo_labels.append(pseudo_global_label)
    
    ground_truths = merged_test_df["label_globalIndex"].to_numpy(dtype="int")
    pseudo_labels = np.array(pseudo_labels)
    acc = np.sum(pseudo_labels == ground_truths) / merged_test_df.shape[0]
    acc = round(acc,4)
    ans["combin_acc"] = acc
    return ans

def get_col_value(model_A, model_B, df, data_source, flag_2):
    num = 0
    percent = 0
    acc_model_A = None
    acc_model_B = None
    confidences_model_A, _ = get_confidences(model_A, df, generator_A_test, target_size_A)
    confidences_model_B, _ = get_confidences(model_B, df, generator_B_test, target_size_B)
    for i in range(confidences_model_A.shape[0]):
        confidence_model_A =  confidences_model_A[i]
        confidence_model_B =  confidences_model_B[i]
        if data_source == "A":
            if confidence_model_A > confidence_model_B:
                num += 1
        elif data_source == "B":
            if confidence_model_B > confidence_model_A:
                num += 1
        else:
            raise Exception("数据来源flag error")
    percent = round(num / df.shape[0],4)
    percent_str = "{:.2f}%".format(percent*100)
    if flag_2 == "A" or flag_2 == "overlap":
        acc_model_A = eval(model_A, df, generator_A_test, target_size_A, classes_A)
        acc_model_A = round(acc_model_A,4)
    if flag_2 == "B" or flag_2 == "overlap":
        acc_model_B = eval(model_B, df, generator_B_test, target_size_B, classes_B)
        acc_model_B = round(acc_model_B,4)
    return [num, percent_str, acc_model_A, acc_model_B]

def analyse():

    test_A_csv_path = f"/data/mml/overlap_v2_datasets/{dataset_name}/party_A/dataset_split/val.csv"
    test_A_unique_csv_path = f"/data/mml/overlap_v2_datasets/{dataset_name}/party_A/dataset_split/val_unique.csv"
    test_A_overlap_csv_path = f"/data/mml/overlap_v2_datasets/{dataset_name}/party_A/dataset_split/val_overlap.csv"
    test_B_csv_path = f"/data/mml/overlap_v2_datasets/{dataset_name}/party_B/dataset_split/val.csv"
    test_B_unique_csv_path = f"/data/mml/overlap_v2_datasets/{dataset_name}/party_B/dataset_split/val_unique.csv"
    test_B_overlap_csv_path = f"/data/mml/overlap_v2_datasets/{dataset_name}/party_B/dataset_split/val_overlap.csv"

    test_A_df = pd.read_csv(test_A_csv_path)
    test_A_unique_df = pd.read_csv(test_A_unique_csv_path)
    test_A_overlap_df = pd.read_csv(test_A_overlap_csv_path)
    test_B_df = pd.read_csv(test_B_csv_path)
    test_B_unique_df = pd.read_csv(test_B_unique_csv_path)
    test_B_overlap_df = pd.read_csv(test_B_overlap_csv_path)

    col_A = get_col_value(model_A, model_B, test_A_df, data_source = "A", flag_2 = "A")
    col_A_unique = get_col_value(model_A, model_B, test_A_unique_df, data_source = "A", flag_2 = "A")
    col_A_overlap = get_col_value(model_A, model_B, test_A_overlap_df, data_source = "A", flag_2 = "overlap")
    col_B = get_col_value(model_A, model_B, test_B_df, data_source = "B", flag_2 = "B")
    col_B_unique = get_col_value(model_A, model_B, test_B_unique_df, data_source = "B", flag_2 = "B")
    col_B_overlap = get_col_value(model_A, model_B, test_B_overlap_df, data_source = "B", flag_2 = "overlap")

    df = pd.DataFrame(np.random.randint(3, 9, size=(4, 6)), 
    index= ["num", "percent", "model_A_acc", "model_B_acc"],
    columns=pd.MultiIndex.from_arrays([["A","A_unique","A_overlap", "B", "B_unique", "B_overlap"],
                                    ["A>B", "A>B","A>B", "B>A", "B>A", "B>A"]]))
    df.loc[:, ("A","A>B")] = col_A
    df.loc[:, ("A_unique","A>B")] = col_A_unique
    df.loc[:, ("A_overlap","A>B")] = col_A_overlap
    df.loc[:, ("B","B>A")] = col_B
    df.loc[:, ("B_unique","B>A")] = col_B_unique
    df.loc[:, ("B_overlap","B>A")] = col_B_overlap
    print("analyse final")
    return df

os.environ['CUDA_VISIBLE_DEVICES']='6'
config = animal_3_config
Base_acc_config = Base_acc.animal_3
dataset_name = config["dataset_name"]
model_A = load_model(config["model_A_struct_path"])
model_A.load_weights(config["model_A_weight_path"])
model_B = load_model(config["model_B_struct_path"])
if not config["model_B_weight_path"] is None:
    model_B.load_weights(config["model_B_weight_path"])
merged_test_df = pd.read_csv(config["merged_df_path"])

generator_A_test = config["generator_A_test"]
generator_B_test = config["generator_B_test"]
target_size_A = config["target_size_A"]
target_size_B = config["target_size_B"]
local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
# 双方的class_name_list
classes_A = getClasses(config["dataset_A_train_path"]) # sorted
classes_B = getClasses(config["dataset_B_train_path"]) # sorted

if __name__ == "__main__":
    df = analyse()
    save_dir = f"exp_table/{dataset_name}"
    makedir_help(save_dir)
    file_name = "RQ_1.xlsx"
    file_path = os.path.join(save_dir, file_name)
    print(df)
    df.to_excel(file_path)
    print(f"savepath:{file_path}")

    # ans = dummy()
    # print(ans)
    # save_dir = f"exp_data/{dataset_name}/retrainResult/dummy"
    # file_name = "dummy.data"
    # file_path = os.path.join(save_dir, file_name)
    # joblib.dump(ans, file_path)
    # print('success')
    pass