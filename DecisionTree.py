
import os
import joblib

import setproctitle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adamax
from tensorflow.python.keras.backend import set_session

from DataSetConfig import car_body_style_config,flower_2_config,food_config,fruit_config,sport_config,weather_config,animal_config,animal_2_config,animal_3_config
from eval_origin_model import EvalOriginModel
from utils import deleteIgnoreFile, makedir_help


def load_models_pool(config, lr=1e-5):
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

def load_cutted_models(config, lr=1e-5):
    # 加载模型
    model_A_cutted = load_model(config["model_A_cutted_path"])

    model_B_cutted = load_model(config["model_B_cutted_path"])

    model_A_cutted.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    model_B_cutted.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    return model_A_cutted, model_B_cutted

def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

def combin_predict_prob_AB(
        root_dir,
        model_A, 
        model_B,
        generator_A_test,
        generator_B_test,
        target_size_A,target_size_B,
        batch_size, 
        all_class_name_list, 
        local_to_global_A,
        local_to_global_B,
        df):
    
    total_num = df.shape[0]
    evalOriginModel_A = EvalOriginModel(model_A, df)
    predicts_A = evalOriginModel_A.predict_prob(root_dir, batch_size, generator_A_test, target_size_A)
    evalOriginModel_B = EvalOriginModel(model_B, df)
    predicts_B = evalOriginModel_B.predict_prob(root_dir, batch_size, generator_B_test, target_size_B)
    predicts_C = np.zeros((total_num,len(all_class_name_list)))
    for i in range(df.shape[0]):
        prob_A = predicts_A[i]
        prob_B = predicts_B[i]
        prob_c = np.zeros(len(all_class_name_list))
        for k,v in local_to_global_A.items():
            prob_c[v] = prob_A[k]
        for k,v in local_to_global_B.items():
            prob_c[v] = prob_B[k]
        predicts_C[i] = prob_c
    return predicts_C

def combin_features_AB(
        root_dir,
        model_A_cutted, 
        model_B_cutted,
        generator_A_test,
        generator_B_test,
        target_size_A,
        target_size_B,
        batch_size, 
        df):
    
    evalOriginModel = EvalOriginModel(model_A_cutted, df)
    features_A = evalOriginModel.get_features(root_dir, batch_size, generator_A_test, target_size_A)
    evalOriginModel = EvalOriginModel(model_B_cutted, df)
    features_B = evalOriginModel.get_features(root_dir, batch_size, generator_B_test, target_size_B)
    total_num = df.shape[0]
    features_num = features_A.shape[1] + features_B.shape[1]
    features_C = np.zeros((total_num,features_num))
    for i in range(df.shape[0]):
        feature_A = features_A[i]
        feature_B = features_B[i]
        feature_C = np.append(feature_A,feature_B)
        features_C[i] = feature_C
    return features_C


def myGenerator(batch_size, X, Y):
    total_size=X.shape[0]
    while True:
        for i in range(total_size//batch_size):
            yield X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]
    return myGenerator


def app_DT_train():
    config = animal_3_config
    dataset_name = config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|DecisionTree|train")
    root_dir = "/data2/mml/overlap_v2_datasets/"
    classes_A = getClasses(config["dataset_A_train_path"])
    classes_B = getClasses(config["dataset_B_train_path"])
    all_class_name_list = list(set(classes_A+classes_B))
    all_class_name_list.sort()
    batch_size = 5
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    model_A_cutted, model_B_cutted = load_cutted_models(config)
    sampled_dir = f"exp_data/{dataset_name}/sampling/percent/random/"
    sample_rate_list = [0.01,0.03,0.05,0.1,0.15,0.2]
    repeat_num = 10
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        rate_dir = os.path.join(sampled_dir, str(int(sample_rate*100)))
        for repeat_i in range(repeat_num):
            print(f"repeat_i:{repeat_i}")
            sampled_df = pd.read_csv(os.path.join(rate_dir, f"sampled_{repeat_i}.csv"))
            combined_features = combin_features_AB(
                root_dir,
                model_A_cutted, 
                model_B_cutted,
                generator_A_test,
                generator_B_test,
                target_size_A,
                target_size_B,
                batch_size, 
                sampled_df)
            Y = sampled_df["label_globalIndex"].to_numpy()
            model = DecisionTreeClassifier(random_state=666)
            model.fit(combined_features, Y)
            save_file_name = f"model_{repeat_i}.joblib"
            save_dir = os.path.join(root_dir,dataset_name,"DecisionTree","trained_models",str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_path = os.path.join(save_dir, save_file_name)
            joblib.dump(model, save_file_path)
            print(f"save_file_path:{save_file_path}")

def app_DT_eval():
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    # 定义出存储结果的数据结构
    '''
    ans = {
        0.01:[],
        0.03:[]
    }
    '''
    ans = {}
    for sample_rate in sample_rate_list:
        ans[sample_rate] = []

    config = animal_3_config
    dataset_name = config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|DecisionTree|eval")
    root_dir = "/data2/mml/overlap_v2_datasets/"
    df_merged = pd.read_csv(config["merged_df_path"])
    classes_A = getClasses(config["dataset_A_train_path"])
    classes_B = getClasses(config["dataset_B_train_path"])
    all_class_name_list = list(set(classes_A+classes_B))
    all_class_name_list.sort()
    batch_size = 5
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    model_A_cutted, model_B_cutted = load_cutted_models(config)
    combined_features = combin_features_AB(
                root_dir,
                model_A_cutted, 
                model_B_cutted,
                generator_A_test,
                generator_B_test,
                target_size_A,
                target_size_B,
                batch_size, 
                df_merged)
    Y = df_merged["label_globalIndex"].to_numpy()
    
    repeat_num = 10
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_i in range(repeat_num):
            print(f"repeat_i:{repeat_i}")   
            save_file_path = os.path.join(root_dir,dataset_name,"DecisionTree","trained_models",str(int(sample_rate*100)), f"model_{repeat_i}.joblib")
            model = joblib.load(save_file_path)
            predictions = model.predict(combined_features)
            acc = accuracy_score(Y, predictions)
            ans[sample_rate].append(acc)
    
    save_dir = os.path.join(root_dir, dataset_name, "DecisionTree")
    save_file_name = f"eval_ans.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("DecisionTree evaluation end")
    return ans

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)    
    # app_DT_train()
    app_DT_eval()




