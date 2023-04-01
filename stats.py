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

def get_pLabel_model(model, df, generator, target_size, local_to_global):
    global_pLabels= []
    batch_size = 5
    batches = generator.flow_from_dataframe(df, 
                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                            x_col='file_path', y_col="label", 
                            target_size=target_size, class_mode='categorical', # one-hot
                            color_mode='rgb', classes=None,
                            shuffle=False, batch_size=batch_size,
                            validate_filenames=False)
    probs = model.predict_generator(batches, steps=batches.n/batch_size)
    pseudo_label_indexes = np.argmax(probs, axis=1)
    for i in range(pseudo_label_indexes.shape[0]):
        local_pLabel = pseudo_label_indexes[i]
        global_pLabel = local_to_global[local_pLabel]
        global_pLabels.append(global_pLabel)
    global_pLabels = np.array(global_pLabels)
    return global_pLabels

def get_ourCombin_model_pLabel(config):
    config["combination_model_path"]


def get_dummy_row():

    pass
def get_HMR_row():
    pass
def get_CFL_row():
    pass
def get_OurCombin_row():
    pass


def get_table():
    df = pd.DataFrame([[None, None, None, None, None]], columns=["ABC", "CA", "CB", "AB", "unique"], index=["dummy", "HMR", "CFL", "OurCombin"])

    pass





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
    pass