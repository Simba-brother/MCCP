import os
import setproctitle
import numpy as np
import joblib
from DataSetConfig import car_body_style_config,flower_2_config,food_config,fruit_config,sport_config,weather_config,animal_config,animal_2_config,animal_3_config
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adamax
from utils import makedir_help


def load_models(config, lr=1e-3):
    # 加载模型
    model_A = load_model(config["model_A_struct_path"])
    if not config["model_A_weight_path"] is None:
        model_A.load_weights(config["model_A_weight_path"])

    model_B = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_B.load_weights(config["model_B_weight_path"])

    model_A.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    model_B.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    return model_A,model_B

def get_confidences(model, df, generator, target_size):
   
   batch_size = 5
   root_dir = "/data2/mml/overlap_v2_datasets/"
   batches = generator.flow_from_dataframe(df, 
                            directory = root_dir, # 添加绝对路径前缀
                            x_col='file_path', y_col="label", 
                            target_size=target_size, class_mode='categorical', # one-hot
                            color_mode='rgb', classes=None,
                            shuffle=False, batch_size=batch_size,
                            validate_filenames=False)
   probs = model.predict_generator(generator = batches, steps=batches.n/batch_size)
   confidences = np.max(probs, axis = 1)
   pseudo_label_indexes = np.argmax(probs, axis=1)
   return confidences, pseudo_label_indexes

class Dummy(object):
    def __init__(self, model_A, model_B, df_test):
        self.model_A = model_A
        self.model_B = model_B
        self.df_test = df_test
    def integrate(self, generator_A_test,  generator_B_test, target_size_A, target_size_B, local_to_global_A, local_to_global_B):
        pseudo_labels = []
        confidences_A, pseudo_labels_A = get_confidences(self.model_A, self.df_test, generator_A_test, target_size_A)
        confidences_B, pseudo_labels_B = get_confidences(self.model_B, self.df_test, generator_B_test, target_size_B)
        for i in range(confidences_A.shape[0]):
            if confidences_A[i] > confidences_B[i]:
                pseudo_global_label = local_to_global_A[pseudo_labels_A[i]]
                pseudo_labels.append(pseudo_global_label)
                
            else:
                pseudo_global_label = local_to_global_B[pseudo_labels_B[i]]
                pseudo_labels.append(pseudo_global_label)
        ground_truths = self.df_test["label_globalIndex"].to_numpy(dtype="int")
        pseudo_labels = np.array(pseudo_labels)
        acc = np.sum(pseudo_labels == ground_truths) / self.df_test.shape[0]
        acc = round(acc,4)
        return acc
    def predict_labels(self, generator_A_test,  generator_B_test, target_size_A, target_size_B, local_to_global_A, local_to_global_B):
        pseudo_labels = []
        confidences_A, pseudo_labels_A = get_confidences(self.model_A, self.df_test, generator_A_test, target_size_A)
        confidences_B, pseudo_labels_B = get_confidences(self.model_B, self.df_test, generator_B_test, target_size_B)
        for i in range(confidences_A.shape[0]):
            if confidences_A[i] > confidences_B[i]:
                pseudo_global_label = local_to_global_A[pseudo_labels_A[i]]
                pseudo_labels.append(pseudo_global_label)
                
            else:
                pseudo_global_label = local_to_global_B[pseudo_labels_B[i]]
                pseudo_labels.append(pseudo_global_label)
        pseudo_labels = np.array(pseudo_labels)
        return pseudo_labels
    
def app_Dummy_NoFangHui(config):
    dataset_name = config["dataset_name"]
    root_dir = "/data2/mml/overlap_v2_datasets/"
    setproctitle.setproctitle(f"{dataset_name}|Dummy|eval|noFangHui")
    test_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/test"
    df_test = pd.read_csv(os.path.join(test_dir, "test.csv"))
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    model_A, model_B = load_models(config=config)
    dummy = Dummy(model_A, model_B, df_test)
    acc = dummy.integrate(generator_A_test,  generator_B_test, target_size_A, target_size_B, local_to_global_A, local_to_global_B)
    save_dir = os.path.join(root_dir, dataset_name, "Dummy")
    makedir_help(save_dir)
    save_file_name = f"eval_ans_NoFangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(acc, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("app_Dummy_NoFangHui end")
    return acc

def app_Dummy_FangHui():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = weather_config
    dataset_name = config["dataset_name"]
    root_dir = "/data2/mml/overlap_v2_datasets/"
    setproctitle.setproctitle(f"{dataset_name}|Dummy|eval|FangHui")
    df_test = pd.read_csv(config["merged_df_path"])
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    model_A, model_B = load_models(config=config)
    dummy = Dummy(model_A, model_B, df_test)
    acc = dummy.integrate(generator_A_test,  generator_B_test, target_size_A, target_size_B, local_to_global_A, local_to_global_B)
    save_dir = os.path.join(root_dir, dataset_name, "Dummy")
    makedir_help(save_dir)
    save_file_name = f"eval_ans_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(acc, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("Dummy evaluation FangHui end")
    return acc

def app_Dummy_classes_FangHui():
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = animal_3_config
    dataset_name = config["dataset_name"]
    root_dir = "/data2/mml/overlap_v2_datasets/"
    setproctitle.setproctitle(f"{dataset_name}|Dummy|eval|FangHui")
    df_test = pd.read_csv(config["merged_df_path"])
    true_labels = df_test["label_globalIndex"].values
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    model_A, model_B = load_models(config=config)
    dummy = Dummy(model_A, model_B, df_test)
    predict_labels = dummy.predict_labels(generator_A_test,  generator_B_test, target_size_A, target_size_B, local_to_global_A, local_to_global_B)
    report = classification_report(true_labels, predict_labels, output_dict=True)
    save_dir = os.path.join(root_dir, dataset_name, "Dummy")
    makedir_help(save_dir)
    save_file_name = f"eval_classes_FangHui.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(report, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("Dummy evaluation FangHui end")
    return report

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='6'
    # tf设置GPU内存分配
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    config = flower_2_config
    app_Dummy_NoFangHui(config)
    # app_Dummy_FangHui()
    # app_Dummy_classes_FangHui()
    pass
