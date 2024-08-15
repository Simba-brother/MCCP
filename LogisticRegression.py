import os
import joblib

import setproctitle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adamax
from tensorflow.python.keras.backend import set_session

from DataSetConfig import exp_dir,car_body_style_config,flower_2_config,food_config,fruit_config,sport_config,weather_config,animal_config,animal_2_config,animal_3_config
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

def get_LogisticRegression_model(classes_num, n_features):
    # 初始化模型
    model = Sequential()
    model.add(Dense(classes_num, input_dim=n_features, activation='sigmoid'))
    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def myGenerator(batch_size, X, Y):
    total_size=X.shape[0]
    while True:
        for i in range(total_size//batch_size):
            yield X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]
    return myGenerator


def app_LR_train():
    config = car_body_style_config
    root_dir = "/data2/mml/overlap_v2_datasets/"
    dataset_name = config["dataset_name"]
    print(f"dataset_name:{dataset_name}")
    model_A, model_B = load_models_pool(config, lr=1e-5)
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    classes_A = getClasses(config["dataset_A_train_path"])
    classes_B = getClasses(config["dataset_B_train_path"])
    all_class_name_list = list(set(classes_A+classes_B))
    all_class_name_list.sort()
    local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    batch_size = 5
    sampled_dir = f"exp_data/{dataset_name}/sampling/percent/random/"
    sample_rate_list = [0.01,0.03,0.05,0.1,0.15,0.2]
    repeat_num = 10
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        rate_dir = os.path.join(sampled_dir, str(int(sample_rate*100)))
        for repeat_i in range(repeat_num):
            print(f"repeat_i:{repeat_i}")
            cur_df = pd.read_csv(os.path.join(rate_dir, f"sampled_{repeat_i}.csv"))
            X = combin_predict_prob_AB(
                root_dir,
                model_A, 
                model_B,
                generator_A_test,
                generator_B_test,
                target_size_A,
                target_size_B,
                batch_size, 
                all_class_name_list,
                local_to_global_A,
                local_to_global_B,
                cur_df)
            Y = np.array(cur_df["label_globalIndex"])
            classes_num = len(all_class_name_list)
            n_features = len(all_class_name_list)
            model = get_LogisticRegression_model(classes_num, n_features)
            model.fit(
                x = X,
                y = Y,
                batch_size = 5,
                epochs = 5,
                verbose = 1,
                shuffle = True)
            save_file_name = f"weight_{repeat_i}.h5"
            save_dir = os.path.join(root_dir,dataset_name,"LogisticRegression","trained_weights",str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_path = os.path.join(save_dir, save_file_name)
            model.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")

def app_LR_eval():
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

    config = car_body_style_config
    
    root_dir = "/data2/mml/overlap_v2_datasets/"
    dataset_name = config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|LR|eval")
    df_merged = pd.read_csv(config["merged_df_path"])
    print(f"dataset_name:{dataset_name}")
    model_A, model_B = load_models_pool(config, lr=1e-5)
    generator_A_test = config["generator_A_test"]
    generator_B_test = config["generator_B_test"]
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    classes_A = getClasses(config["dataset_A_train_path"])
    classes_B = getClasses(config["dataset_B_train_path"])
    all_class_name_list = list(set(classes_A+classes_B))
    all_class_name_list.sort()
    local_to_global_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_B = joblib.load(config["local_to_global_party_B_path"])
    batch_size = 5
    X = combin_predict_prob_AB(
                root_dir,
                model_A, 
                model_B,
                generator_A_test,
                generator_B_test,
                target_size_A,
                target_size_B,
                batch_size, 
                all_class_name_list,
                local_to_global_A,
                local_to_global_B,
                df_merged)
    Y = np.array(df_merged["label_globalIndex"])
    classes_num = len(all_class_name_list)
    n_features = len(all_class_name_list)
    model = get_LogisticRegression_model(classes_num, n_features)
    repeat_num = 10
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        for repeat_i in range(repeat_num):
            print(f"repeat_i:{repeat_i}")
            weight_path = os.path.join(root_dir, dataset_name, "LogisticRegression", "trained_weights", str(int(sample_rate*100)), f"weight_{repeat_i}.h5")
            model.load_weights(weight_path)
            eval_res = model.evaluate(
                x = X,
                y = Y,
                batch_size = 32,
                verbose = 1,
                return_dict=True
                )
            acc = eval_res['accuracy']
            ans[sample_rate].append(acc)
    save_dir = os.path.join(root_dir, dataset_name, "LogisticRegression")
    save_file_name = f"eval_ans.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("LogisticRegression evaluation end")
    return ans

def app_LR2_train(config, isFangHuiFlag):
    if isFangHuiFlag is True:
        suffix = "FangHui"
    else:
        suffix = "NoFangHui"
    dataset_name = config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|LogisticRegression|train")
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
    if isFangHuiFlag is True:
        sampled_dir = f"exp_data/{dataset_name}/sampling/percent/random/"
    else:
        sampled_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/train/"
    sample_rate_list = [0.01,0.03,0.05,0.1,0.15,0.2]
    repeat_num = 10
    for sample_rate in sample_rate_list:
        print(f"sample_rate:{sample_rate}")
        rate_dir = os.path.join(sampled_dir, str(int(sample_rate*100)))
        for repeat_i in range(repeat_num):
            print(f"repeat_i:{repeat_i}")
            if isFangHuiFlag is True:
                sampled_df = pd.read_csv(os.path.join(rate_dir, f"sampled_{repeat_i}.csv"))
            else:
                sampled_df = pd.read_csv(os.path.join(rate_dir, f"sample_{repeat_i}.csv"))
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
            classes_num = len(all_class_name_list)
            n_features = combined_features.shape[1]
            model = get_LogisticRegression_model(classes_num, n_features)
            model.fit(
                x = combined_features,
                y = Y,
                batch_size = 5,
                epochs = 5,
                verbose = 1,
                shuffle = True)
            save_file_name = f"weight_{repeat_i}.h5"
            save_dir = os.path.join(root_dir,dataset_name,"LogisticRegression",f"trained_weights_{suffix}",str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_path = os.path.join(save_dir, save_file_name)
            model.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
            
    

def app_LR2_eval(config, isFangHuiFlag):
    if isFangHuiFlag is True:
        suffix = "FangHui"
    else:
        suffix = "NoFangHui"
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

    dataset_name = config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|LogisticRegression|eval")
    root_dir = exp_dir
    if isFangHuiFlag is True:
        df_merged = pd.read_csv(config["merged_df_path"])
    else:
        df_merged = pd.read_csv(os.path.join(f"exp_data/{dataset_name}/sampling/percent/random_split/test/test.csv"))
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
            classes_num = len(all_class_name_list)
            n_features = combined_features.shape[1]
            model = get_LogisticRegression_model(classes_num, n_features)
            weight_path = os.path.join(root_dir,dataset_name,"LogisticRegression",f"trained_weights_{suffix}",str(int(sample_rate*100)),  f"weight_{repeat_i}.h5")
            model.load_weights(weight_path)
            eval_res = model.evaluate(
                x = combined_features,
                y = Y,
                batch_size = 32,
                verbose = 1,
                return_dict=True
                )
            acc = eval_res['accuracy']
            ans[sample_rate].append(acc)
    
    save_dir = os.path.join(root_dir, dataset_name, "LogisticRegression")
    save_file_name = f"eval_ans_{suffix}.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("LogisticRegression evaluation end")
    return ans

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='7'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    # combin probability output
    # app_LR_train()
    # app_LR_eval()
    # combin the last hidden layer output features
    config = animal_3_config
    # app_LR2_train(config, isFangHuiFlag=False)
    app_LR2_eval(config, isFangHuiFlag=False)




