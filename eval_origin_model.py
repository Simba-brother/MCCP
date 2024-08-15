import os
import joblib
import numpy as np
import setproctitle
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adamax
from DataSetConfig import exp_dir,car_body_style_config,flower_2_config,food_config,fruit_config,sport_config,weather_config,animal_config,animal_2_config,animal_3_config
from utils import deleteIgnoreFile, makedir_help
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

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

class EvalOriginModel(object):
    def __init__(self, model, df):
        self.model = model
        self.df = df
    def eval(self, root_dir,batch_size,  generator_test, target_size, classes):
        batches = generator_test.flow_from_dataframe(
            self.df, 
            directory = root_dir, # 添加绝对路径前缀
            # subset="training",
            seed=666,
            x_col='file_path', y_col="label", 
            target_size=target_size, class_mode='categorical', # one-hot
            color_mode='rgb', classes=classes, 
            shuffle=False, batch_size=batch_size,
            validate_filenames=False)
        eval_res = self.model.evaluate(
                    batches, 
                    batch_size = batch_size, 
                    verbose=1,
                    steps = batches.n/batch_size, 
                    return_dict=True)
        return eval_res
    
    def predict_prob(self, root_dir, batch_size, generator, target_size):
        batches = generator.flow_from_dataframe(
            self.df, 
            directory = root_dir, # 添加绝对路径前缀
            x_col='file_path', y_col="label", 
            target_size=target_size, 
            class_mode='categorical', # one-hot
            color_mode='rgb', 
            classes=None,
            shuffle=False, 
            batch_size=batch_size,
            validate_filenames=False)
        probs = self.model.predict_generator(generator = batches, steps=batches.n/batch_size)
        return probs
    
    def get_feature(self, root_dir, batch_size, generator, target_size):
        batches = generator.flow_from_dataframe(
            self.df, 
            directory = root_dir, # 添加绝对路径前缀
            x_col='file_path', y_col="label", 
            target_size=target_size, 
            class_mode='categorical', # one-hot
            color_mode='rgb', 
            classes=None,
            shuffle=False, 
            batch_size=batch_size,
            validate_filenames=False)
        features = self.model.predict(batches)
        return features
    

    
def app_eval_origin_model_with_overlap_merged(config,isFangHuiFlag):
    if isFangHuiFlag is True:
        suffix = "FangHui"
    else:
        suffix = "NoFangHui"
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = exp_dir
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|OriginModel|eval_Overlap_mergertd_test")
    if isFangHuiFlag is True:
        merged_overlap_df = pd.read_csv(config["merged_overlap_df"]) 
    else:
        df =  pd.read_csv(f"exp_data/{dataset_name}/sampling/percent/random_split/test/test.csv")
        overlap_df = df[df["is_overlap"] == 1]
        merged_overlap_df = overlap_df
    model_A, model_B = load_models_pool(config) # compiled
    # 评估model_A
    evalOriginModel = EvalOriginModel(model_A, merged_overlap_df)
    eval_ans_A = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_A_test"],
        target_size = config["target_size_A"], 
        classes = getClasses(config["dataset_A_train_path"]) # sorted
        )
    # 评估model_B
    evalOriginModel = EvalOriginModel(model_B, merged_overlap_df)
    eval_ans_B = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_B_test"],
        target_size = config["target_size_B"], 
        classes = getClasses(config["dataset_B_train_path"]) # sorted
        )
    acc_A = eval_ans_A["accuracy"]
    acc_B = eval_ans_B["accuracy"]
    ans = {"acc_A":acc_A, "acc_B":acc_B}
    save_dir = os.path.join(root_dir, dataset_name, "OriginModel")
    makedir_help(save_dir)
    save_file_name = f"eval_overlap_merged_test_{suffix}.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("origin model eval overlap merged test end")
    return ans

def app_eval_origin_model_with_unique_merged(config,isFangHuiFlag):
    if isFangHuiFlag is True:
        suffix = "FangHui"
    else:
        suffix = "NoFangHui"
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = exp_dir
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|OriginModel|unique_merged")
    if isFangHuiFlag is True:
        merged_unique_df = pd.read_csv(config["merged_unique_df"]) 
    else:
        df =  pd.read_csv(f"exp_data/{dataset_name}/sampling/percent/random_split/test/test.csv")
        unique_df = df[df["is_overlap"] == 0]
        merged_unique_df = unique_df
    unique_A_df = merged_unique_df[merged_unique_df["source"]==1]
    unique_B_df = merged_unique_df[merged_unique_df["source"]==2]
    model_A, model_B = load_models_pool(config) # compiled
    A_classes = getClasses(config["dataset_A_train_path"])
    B_classes = getClasses(config["dataset_B_train_path"])
    # 评估model_A
    evalOriginModel = EvalOriginModel(model_A, unique_A_df)
    eval_ans_A = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_A_test"],
        target_size = config["target_size_A"], 
        classes = A_classes # sorted
        )
    acc_A = eval_ans_A["accuracy"] * unique_A_df.shape[0] / merged_unique_df.shape[0]
    # 评估model_B
    evalOriginModel = EvalOriginModel(model_B, unique_B_df)
    eval_ans_B = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_B_test"], 
        target_size = config["target_size_B"], 
        classes = B_classes
        )
    acc_B = eval_ans_B["accuracy"] * unique_B_df.shape[0] / merged_unique_df.shape[0]
    ans = {"acc_A":acc_A, "acc_B":acc_B}
    save_dir = os.path.join(root_dir, dataset_name, "OriginModel")
    makedir_help(save_dir)
    save_file_name = f"eval_unique_merged_test_{suffix}.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("origin model eval unique merged test end")
    return ans

def app_eval_origin_model_with_merged(config,isFangHuiFlag):
    if isFangHuiFlag is True:
        suffix = "FangHui"
    else:
        suffix = "NoFangHui"
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = exp_dir
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|OriginModel|unique_merged")
    if isFangHuiFlag is True:
        merged_df = pd.read_csv(config["merged_df_path"]) 
    else:
        merged_df =  pd.read_csv(f"exp_data/{dataset_name}/sampling/percent/random_split/test/test.csv")
    merged_df_A = merged_df[merged_df["source"] == 1]
    merged_df_B = merged_df[merged_df["source"] == 2]
    model_A, model_B = load_models_pool(config) # compiled
    A_classes = getClasses(config["dataset_A_train_path"])
    B_classes = getClasses(config["dataset_B_train_path"])
    # 评估model_A
    evalOriginModel = EvalOriginModel(model_A, merged_df_A)
    eval_ans_A = evalOriginModel.eval(
        root_dir, 
        batch_size=32,  
        generator_test = config["generator_A_test"],
        target_size = config["target_size_A"], 
        classes = A_classes # sorted
        )
    acc_A = eval_ans_A["accuracy"] * merged_df_A.shape[0] / merged_df.shape[0]
    # 评估model_B
    evalOriginModel = EvalOriginModel(model_B, merged_df_B)
    eval_ans_B = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_B_test"], 
        target_size = config["target_size_B"], 
        classes = B_classes
        )
    acc_B = eval_ans_B["accuracy"] * merged_df_B.shape[0] / merged_df.shape[0]
    ans = {"acc_A":acc_A, "acc_B":acc_B}
    save_dir = os.path.join(root_dir, dataset_name, "OriginModel")
    makedir_help(save_dir)
    save_file_name = f"eval_merged_test_{suffix}.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("origin model eval unique merged test end")
    return ans


def app_get_features():
    model_A, model_B = load_models_pool(config)
    model_A_layerIndex = config["model_A_cut"]
    model_B_layerIndex = config["model_B_cut"]
    model_A_cutted = keras.Model(inputs = model_A.input, outputs = model_A.get_layer(index = model_A_layerIndex).output)
    model_B_cutted = keras.Model(inputs = model_B.input, outputs = model_B.get_layer(index = model_B_layerIndex).output)
    df_A_val_overlap = pd.read_csv(config["df_A_val_overlap"])
    df_B_val_overlap = pd.read_csv(config["df_B_val_overlap"])

    A_label_globalIndex_list = df_A_val_overlap["label_globalIndex"]
    B_label_globalIndex_list = df_B_val_overlap["label_globalIndex"]
    

    evalOriginModel = EvalOriginModel(model_A_cutted, df_A_val_overlap)
    AA_features = evalOriginModel.get_feature(        
        root_dir= exp_dir,
        batch_size=32,  
        generator = config["generator_A_test"],
        target_size = config["target_size_A"],
        )

    evalOriginModel = EvalOriginModel(model_A_cutted, df_B_val_overlap)
    AB_features = evalOriginModel.get_feature(        
        root_dir= exp_dir,
        batch_size=32,  
        generator = config["generator_A_test"],
        target_size = config["target_size_A"],
        )
    
    evalOriginModel = EvalOriginModel(model_B_cutted, df_A_val_overlap)
    BA_features = evalOriginModel.get_feature(        
        root_dir= exp_dir,
        batch_size=32,  
        generator = config["generator_B_test"],
        target_size = config["target_size_B"],
        )
    
    evalOriginModel = EvalOriginModel(model_B_cutted, df_B_val_overlap)
    BB_features = evalOriginModel.get_feature(        
        root_dir= exp_dir,
        batch_size=32,  
        generator = config["generator_B_test"],
        target_size = config["target_size_B"],
        )
    save_dir = os.path.join(exp_dir, config["dataset_name"])
    save_filename = "features.data"
    features = {
        "AA_features":AA_features,
        "AB_features":AB_features,
        "BA_features":BA_features,
        "BB_features":BB_features,
        "A_label_globalIndex_list":A_label_globalIndex_list,
        "B_label_globalIndex_list":B_label_globalIndex_list
    }
    save_path = os.path.join(save_dir,save_filename)
    joblib.dump(features, save_path)
    print(f"save file in {save_path}")



def app_eval_origin_model(config):
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = exp_dir
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|OriginModel|eval")

    A_overlap_df = pd.read_csv(config["df_A_val_overlap"]) 
    B_overlap_df = pd.read_csv(config["df_B_val_overlap"]) 

    model_A, model_B = load_models_pool(config) # compiled

    # 评估model_A,在overlap A
    evalOriginModel = EvalOriginModel(model_A, A_overlap_df)
    eval_ans_AA = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_A_test"],
        target_size = config["target_size_A"], 
        classes = getClasses(config["dataset_A_train_path"]) # sorted
        )
    
    # 评估model_A,在overlap B
    evalOriginModel = EvalOriginModel(model_A, B_overlap_df)
    eval_ans_AB = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_A_test"],
        target_size = config["target_size_A"], 
        classes = getClasses(config["dataset_A_train_path"]) # sorted
        )
    
    # 评估model_B 在overlap A
    evalOriginModel = EvalOriginModel(model_B, A_overlap_df)
    eval_ans_BA = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_B_test"],
        target_size = config["target_size_B"], 
        classes = getClasses(config["dataset_B_train_path"]) # sorted
        )
    
    # 评估model_B 在overlap A
    evalOriginModel = EvalOriginModel(model_B, B_overlap_df)
    eval_ans_BB = evalOriginModel.eval(
        root_dir,
        batch_size=32,  
        generator_test = config["generator_B_test"],
        target_size = config["target_size_B"], 
        classes = getClasses(config["dataset_B_train_path"]) # sorted
        )
    acc_AA = eval_ans_AA["accuracy"]
    acc_AB = eval_ans_AB["accuracy"]
    acc_BA = eval_ans_BA["accuracy"]
    acc_BB = eval_ans_BB["accuracy"]
    ans = {"acc_AA":acc_AA, "acc_AB":acc_AB, "acc_BA":acc_BA, "acc_BB":acc_BB}
    print(ans)
    print("origin model eval end")
    return ans
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    config = animal_3_config

    # app_eval_origin_model_with_overlap_merged(config,isFangHuiFlag=False)
    # app_eval_origin_model_with_unique_merged(config,isFangHuiFlag=False)
    # app_eval_origin_model_with_merged(config,isFangHuiFlag=False)

    # app_eval_origin_model(config)

    app_get_features()
    

