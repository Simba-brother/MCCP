import os
import joblib
import numpy as np
import setproctitle
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adamax
from DataSetConfig import car_body_style_config,flower_2_config,food_config,fruit_config,sport_config,weather_config,animal_config,animal_2_config,animal_3_config
from utils import deleteIgnoreFile, makedir_help
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
    
def app_eval_origin_model_with_overlap_merged():
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth=True 
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.compat.v1.Session(config=config_tf)
    set_session(session)
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = animal_3_config
    dataset_name =  config["dataset_name"]
    setproctitle.setproctitle(f"{dataset_name}|OriginModel|eval_Overlap_mergertd_test")
    merged_overlap_df = pd.read_csv(config["merged_overlap_df"]) 
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
    save_file_name = f"eval_overlap_merged_test_.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(ans, save_file_path)
    print(f"save_file_path:{save_file_path}")
    print("origin model eval overlap merged test end")
    return ans


if __name__ == "__main__":
    app_eval_origin_model_with_overlap_merged()