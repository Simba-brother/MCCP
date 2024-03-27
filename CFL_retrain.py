import os
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
from DataSetConfig import car_body_style_config
from utils import deleteIgnoreFile, makedir_help


def local_to_global(localToGlobal_dic, proba_local, all_class_nums):
    '''
    i方的proba => globa proba
    '''
    predict_value = np.zeros((proba_local.shape[0], all_class_nums))
    mapping = []
    for key, value in localToGlobal_dic.items():
        mapping.append(value)
    predict_value[:, mapping] = proba_local
    return predict_value

def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

class CFL_Retrain(object):
    def __init__(self, t1, t2, stu, df_retrain, df_test):    
        self.t1 = t1
        self.t2 = t2
        self.stu = stu
        self.df_retrain = df_retrain
        self.df_test = df_test

        
    def train(self, 
              epoches, 
              batch_size, 
              lr, 
              generator_A, 
              target_size_A, 
              generator_B, 
              target_size_B, 
              generator_stu_train, 
              target_size_stu,
              local_to_global_party_A,
              local_to_global_party_B,
              all_class_nums):
        root_dir = "/data2/mml/overlap_v2_datasets/"
        df_retrain = self.df_retrain
        optimizer=Adamax(learning_rate=lr)
        t1 = self.t1
        t2 = self.t2
        stu = self.stu
        
        for epoch in range(epoches):
            print("epoch:{epoch}")
            epoch_loss = 0
            batches_A = generator_A.flow_from_dataframe(df_retrain, 
                                        directory = root_dir, # 添加绝对路径前缀
                                        # subset="training",
                                        seed=666,
                                        x_col='file_path', y_col="label", 
                                        target_size=target_size_A, class_mode='categorical', # one-hot
                                        color_mode='rgb', classes=None,
                                        shuffle=False, batch_size=batch_size,
                                        validate_filenames=False)
            batches_B = generator_B.flow_from_dataframe(df_retrain, 
                                        directory = root_dir, # 添加绝对路径前缀
                                        # subset="training",
                                        seed=666,
                                        x_col='file_path', y_col="label", 
                                        target_size=target_size_B, class_mode='categorical', # one-hot
                                        color_mode='rgb', classes=None,
                                        shuffle=False, batch_size=batch_size,
                                        validate_filenames=False)
            batches_stu = generator_stu_train.flow_from_dataframe(df_retrain, 
                                    directory = root_dir, # 添加绝对路径前缀
                                    # subset="training",
                                    seed=666,
                                    x_col='file_path', y_col="label", 
                                    target_size=target_size_stu, class_mode='categorical', # one-hot
                                    color_mode='rgb', classes=None,
                                    shuffle=False, batch_size=batch_size,
                                    validate_filenames=False)
            for i in range(len(batches_A)):
                print(f"训练批次:{i}")
                batch_A = next(batches_A)
                batch_B = next(batches_B)
                batch_stu = next(batches_stu)
                X_a = batch_A[0]
                X_b = batch_B[0]
                X_stu = batch_stu[0]
                out_a = t1(X_a, training=False)
                out_b = t2(X_b, training=False)
                out_a_global = local_to_global(local_to_global_party_A,out_a,all_class_nums)
                out_b_global = local_to_global(local_to_global_party_B,out_b,all_class_nums)
                out_ab = np.max(np.array([out_a_global,out_b_global]), axis=0)
               
                with tf.GradientTape() as tape:
                    out_stu = stu(X_stu, training=True)
                    batch_loss = categorical_crossentropy(out_ab, out_stu)
                    epoch_loss += tf.reduce_sum(batch_loss)
                # with tf.GradientTape() as tape:
                #     tape.watch(stu_model.trainable_variables)
                #     stu_out = get_stu_out(stu_model, retrain_df)
                #     loss = soft_cross_entropy(stu_out,integ_out)
                gradients = tape.gradient(batch_loss, stu.trainable_weights)
                optimizer.apply_gradients(zip(gradients, stu.trainable_weights))
        # 返回训练好的stu
        return stu



def app_CFL_retrain():
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    root_dir = "/data2/mml/overlap_v2_datasets/"
    config = car_body_style_config
    dataset_name = config["dataset_name"]
    train_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/train"
    test_dir = f"exp_data/{dataset_name}/sampling/percent/random_split/test"
    df_test = pd.read_csv(os.path.join(test_dir, "test.csv"))
    generator_A = config["generator_A"]
    generator_B = config["generator_B"]
    generator_stu_train = ImageDataGenerator(rescale=1/255.)
    # generator_A_test = config["generator_A_test"]
    # generator_B_test = config["generator_B_test"]
    # generator_stu_test = ImageDataGenerator(rescale=1/255.)
    local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
    local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
    target_size_A = config["target_size_A"]
    target_size_B = config["target_size_B"]
    target_size_stu = (224,224)
        # 双方的class_name_list
    class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
    class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
    # all_class_name_list
    all_class_name_list = list(set(class_name_list_A+class_name_list_B))
    all_class_name_list.sort()
    # 总分类数
    all_class_nums = len(all_class_name_list)
    sample_rate_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    for sample_rate in sample_rate_list:
        sample_rate_dir = os.path.join(train_dir, str(int(sample_rate*100)))
        for repeat_num in range(5):
            df_retrain = pd.read_csv(os.path.join(sample_rate_dir, f"sample_{repeat_num}.csv"))
             # 加载模型
            model_A = load_model(config["model_A_struct_path"])
            if not config["model_A_weight_path"] is None:
                model_A.load_weights(config["model_A_weight_path"])
                model_B = load_model(config["model_B_struct_path"])
            if not config["model_B_weight_path"] is None:
                model_B.load_weights(config["model_B_weight_path"])
            # 编译model
            model_A.compile(loss=categorical_crossentropy,
                        optimizer=Adamax(learning_rate=1e-3),
                        metrics=['accuracy'])
            model_B.compile(loss=categorical_crossentropy,
                        optimizer=Adamax(learning_rate=1e-3),
                        metrics=['accuracy'])
            stu_model = load_model(config["stu_model_path"])
            for layer in stu_model.layers[:-1]:
                layer.trainable = False
            cfl_retrain = CFL_Retrain(model_A, model_B, stu_model, df_retrain, df_test)
            model = cfl_retrain.train(
                epoches = 5,
                batch_size = 5,
                lr = 1e-3,
                generator_A = generator_A,
                target_size_A = target_size_A,
                generator_B = generator_B,
                target_size_B = target_size_B,
                generator_stu_train = generator_stu_train,
                target_size_stu = target_size_stu,
                local_to_global_party_A = local_to_global_party_A,
                local_to_global_party_B = local_to_global_party_B,
                all_class_nums = all_class_nums
                )
            
            save_dir = os.path.join(root_dir, dataset_name, "CFL", "trained_weights", str(int(sample_rate*100)))
            makedir_help(save_dir)
            save_file_name = f"weight_{repeat_num}.h5"
            save_file_path = os.path.join(save_dir, save_file_name)
            model.save_weights(save_file_path)
            print(f"save_file_path:{save_file_path}")
    print("==========CFL retraining ends==========")

if __name__ == "__main__":
    app_CFL_retrain()
    


