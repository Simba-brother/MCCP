'''
简化版本的CFL
code: https://github.com/zju-vipa/CommonFeatureLearning
'''
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam,Adamax
import tensorflow.keras as keras
import torch.nn.functional as F

import torch
from sklearn.utils import shuffle
import math
import os
import random
import joblib
from utils import deleteIgnoreFile,saveData
from tensorflow.keras.activations import softmax
# 加载数据集 config
from DataSetConfig import food_config, fruit_config, sport_config, weather_config, flower_2_config, car_body_style_config, animal_config, animal_2_config, animal_3_config
import Base_acc


# 方法区

def froze_model(model):
    for layer in model.layers[:-1]:
        layer.trainable = False
    return model

def load_models_pool(config, lr=1e-3):
    # 加载模型
    model_A = load_model(config["model_A_struct_path"])
    if not config["model_A_weight_path"] is None:
        model_A.load_weights(config["model_A_weight_path"])

    model_B = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_B.load_weights(config["model_B_weight_path"])
    # 编译model
    model_A.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    model_B.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    # 各方模型池
    model_pool = [model_A, model_B]
    return model_pool

def evaluate_integ(model_pool,df):
    '''
    集成评估
    args:
        model_pool:模型池
        df:评估数据集
    return:
        accuracy
    '''
    n = df.shape[0]
    predict_value = np.zeros((len(model_pool), n, all_class_nums))
    for i,model in enumerate(model_pool):
        generator = generator_test_list[i]
        target_size = target_size_list[i]
        proba_local = predict_proba(model, df, generator, target_size, batch_size=8)
        localToGlobal_dic = localToGlobal_mapping[i]
        proba_global = local_to_global(localToGlobal_dic,proba_local)
        predict_value[i, :, :] = proba_global
    proba_vector = np.max(predict_value, axis=0)
    predict_global_idx_array = np.argmax(proba_vector,axis=1)
    true_global_idx_array = np.array(df["label_globalIndex"])
    accuracy = np.sum(predict_global_idx_array == true_global_idx_array) / df.shape[0]
    return accuracy


def local_to_global(localToGlobal_dic,proba_local):
    '''
    i方的proba => globa proba
    '''
    predict_value = np.zeros((proba_local.shape[0], all_class_nums))
    mapping = []
    for key, value in localToGlobal_dic.items():
        mapping.append(value)
    predict_value[:, mapping] = proba_local
    return predict_value

def predict_proba(model, df, generator, target_size,batch_size):
    '''
    i方的对df的预测概率
    '''
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            # subset="training",
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=None,
                                            shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)

    proba = model.predict(batches,    
                    batch_size=None,
                    verbose="auto",
                    steps=None,
                    callbacks=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False)
    return proba

def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list



def get_out(model, df, generator, target_size, batch_size):
    '''
    i方的对df的out(no softmax)
    '''
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            # subset="training",
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=None,
                                            shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)
    # 禁用softmax
    model.layers[-1].activation = None 
    out = model.predict(batches, batch_size=batch_size, steps=len(batches), verbose = 1)
    return out


def get_integ_out(model_pool,df):
    '''
    功能: 得到集成的out(no soft_max)
    args:
        model_pool:模型池
        df:retrain_df
    return:
        out
    '''
    n = df.shape[0]
    outs = np.zeros((len(model_pool), n, all_class_nums))
    for i,model in enumerate(model_pool):
        generator = generator_test_list[i]
        target_size = target_size_list[i]
        out = get_out(model, df, generator, target_size, batch_size=5)
        localToGlobal_dic = localToGlobal_mapping[i]
        out_global = local_to_global(localToGlobal_dic,out)
        outs[i, :, :] = out_global
    out_integ = np.max(outs, axis=0)
    return out_integ

def get_batches(batches_A,batches_B):
    '''
    将连个模型的输入bath 同时返回
    '''
    while True:
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def get_stu_out(model, df):
    batch_size = 5
    classes = all_class_name_list
    outs = np.zeros((1,len(classes)))
    batches_A = generator_A.flow_from_dataframe(df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                seed=42,
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_A, class_mode='categorical',
                                                color_mode='rgb', classes=classes, 
                                                shuffle=False, batch_size=batch_size)

    batches_B =  generator_B.flow_from_dataframe(df, 
                                                directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                                seed=42,
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_B, class_mode='categorical',
                                                color_mode='rgb', classes=classes, 
                                                shuffle=False, batch_size=batch_size)
    batches = get_batches(batches_A, batches_B)
    # 禁用softmax
    model.layers[-1].activation = None 
    for i in range(len(batches_A)):
        batch = next(batches)
        X = batch[0]
        Y = batch[1]
        out = model(X) 
        outs = np.concatenate((outs, out), axis=0)
    return outs[1:,:]


def soft_cross_entropy(logics,targets):
    p_targets = tf.nn.softmax(targets, axis=1)
    logics = tf.nn.log_softmax(logics, axis=1)
    soft_ce = tf.reduce_sum(-p_targets * logics, axis=1)
    soft_ce = sum(soft_ce)
    soft_ce = tf.Variable(soft_ce)

    return soft_ce

def combin_batches(batches_A, batches_B):
    while True:
        X1i = batches_A.next()
        X2i = batches_B.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def eval_combin_model(model, df):
    total = df.shape[0]
    correct_num = 0
    batch_size = 5
    prefix_path = "/data/mml/overlap_v2_datasets/"

    batches_A = generator_A_test.flow_from_dataframe(df, 
                                                directory = prefix_path , # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_A, class_mode='categorical',
                                                color_mode='rgb', classes = all_class_name_list, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)
                                                                                                                # weather:rgb  150, 150

    batches_B = generator_B_test.flow_from_dataframe(df, 
                                                directory = prefix_path, # 添加绝对路径前缀
                                                x_col='file_path', y_col='label', 
                                                target_size=target_size_B, class_mode='categorical',
                                                color_mode='rgb', classes = all_class_name_list, shuffle=False, batch_size=batch_size,
                                                validate_filenames=False)

    test_batches = combin_batches(batches_A, batches_B)

    for i in range(len(batches_A)):
        batch = next(test_batches)
        X = batch[0]
        Y = batch[1]
        # model.layers[-1].activation = None 
        out = model(X, training=False)
        p_label = np.argmax(out, axis = 1)
        ground_label = np.argmax(Y, axis = 1)
        correct_num += np.sum(p_label==ground_label)
    acc = round(correct_num / total, 4)
    # model.compile(loss=categorical_crossentropy,
    #             optimizer=Adamax(learning_rate=lr),
    #             metrics=['accuracy'])
    # acc = model.evaluate(test_batches, batch_size = batch_size, verbose=1,steps = batches_A.n/batch_size, return_dict=True)
    return acc

def eval_stu_model(model, df):
    total = df.shape[0]
    correct_num = 0
    batch_size = 5
    prefix_path = "/data/mml/overlap_v2_datasets/"

    batches = generator_stu_test.flow_from_dataframe(df, 
                                            directory = prefix_path , # 添加绝对路径前缀
                                            x_col='file_path', y_col='label', 
                                            target_size=target_size_stu, class_mode='categorical',
                                            color_mode='rgb', classes = all_class_name_list, 
                                            shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)
    
    model.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    ans_dic = model.evaluate(batches, batch_size = batch_size, verbose=1,steps = batches.n/batch_size, return_dict=True)
    loss = ans_dic["loss"]
    accuracy = ans_dic["accuracy"]
    return accuracy

def kd(model_pool, stu_model, retrain_df, test_df, base_acc, epoches):
    '''
    功能:使用知识蒸馏方法,利用model_pool对stu_model进行训练。是CFL的baseLine(作为我们的CFL_simple)
    参数：
        retrain_df:用于stu_model训练的数据集,即随机采样集
        test_df: 用于评估stu_model的数据集,即merged_test_df
    '''
    model_A = model_pool[0]
    model_B = model_pool[1]
    batch_size = 5
    optimizer=Adamax(learning_rate=lr)
    for epoch in range(epoches):
        epoch_loss = 0
        batches_A = generator_A.flow_from_dataframe(retrain_df, 
                                        directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                        # subset="training",
                                        seed=42,
                                        x_col='file_path', y_col="label", 
                                        target_size=target_size_A, class_mode='categorical', # one-hot
                                        color_mode='rgb', classes=None,
                                        shuffle=False, batch_size=batch_size,
                                        validate_filenames=False)

        batches_B = generator_B.flow_from_dataframe(retrain_df, 
                                        directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                        # subset="training",
                                        seed=42,
                                        x_col='file_path', y_col="label", 
                                        target_size=target_size_B, class_mode='categorical', # one-hot
                                        color_mode='rgb', classes=None,
                                        shuffle=False, batch_size=batch_size,
                                        validate_filenames=False)
        
        batches_stu = generator_stu_train.flow_from_dataframe(retrain_df, 
                                        directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                        # subset="training",
                                        seed=42,
                                        x_col='file_path', y_col="label", 
                                        target_size=target_size_stu, class_mode='categorical', # one-hot
                                        color_mode='rgb', classes=None,
                                        shuffle=False, batch_size=batch_size,
                                        validate_filenames=False)
        
        # batches = get_batches(batches_A, batches_B)
        

        # model_A.layers[-1].activation = None 
        # model_B.layers[-1].activation= None 
        # stu_model.layers[-1].activation = None 
        for i in range(len(batches_A)):
            print(f"训练轮次:{epoch} 训练批次:{i}")
            batch_A = next(batches_A)
            batch_B = next(batches_B)
            batch_stu = next(batches_stu)
            X_a = batch_A[0]
            X_b = batch_B[0]
            X_stu = batch_stu[0]

            out_a = model_A(X_a, training=False)
            out_b = model_B(X_b, training=False)
            
            out_a_global = local_to_global(local_to_global_party_A,out_a)
            out_b_global = local_to_global(local_to_global_party_B,out_b)
            out_ab = np.max(np.array([out_a_global,out_b_global]), axis=0)
            with tf.GradientTape() as tape:
                out_stu = stu_model(X_stu, training=True)
                loss = categorical_crossentropy(out_ab, out_stu)
                epoch_loss += tf.reduce_sum(loss)
            # with tf.GradientTape() as tape:
            #     tape.watch(stu_model.trainable_variables)
            #     stu_out = get_stu_out(stu_model, retrain_df)
            #     loss = soft_cross_entropy(stu_out,integ_out)
            gradients = tape.gradient(loss, stu_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, stu_model.trainable_weights))
        epoch_acc = eval_stu_model(stu_model, test_df)
        epoch_improve_acc = round(epoch_acc-base_acc,4)
        print(f"训练轮次:{epoch}, 轮次精度:{epoch_acc}, 轮次提升:{epoch_improve_acc}, 训练轮次损失:{epoch_loss.numpy()}, 训练批次损失:{round(epoch_loss.numpy()/len(batches_A),4)}")
    acc = eval_stu_model(stu_model, test_df)
    # print(f"epoch:{epoch},acc:{acc}")
    return acc
    

def retrain():
    '''
    简化版本的CFL,其实就是CFL论文中的kd
    '''
    # 加载模型
    model_pool = load_models_pool(config, lr)
    # 加载混合评估集
    df_test = pd.read_csv(config["merged_df_path"])
    # 评估模型集成结果
    # base_acc = evaluate_integ(model_pool,df_test)
    # print(f"集成输出在混合测试集的acc:{round(base_acc,4)}")
    stu_model = load_model(config["stu_model_path"])
    base_acc = eval_stu_model(stu_model, df_test)
    print(f"集成stu model在混合测试集的acc:{round(base_acc,4)}")
    common_dir = config["sampled_common_path"]
    sample_num_list = deleteIgnoreFile(os.listdir(common_dir))
    sample_num_list = sorted(sample_num_list, key=lambda e: int(e))
    sample_num_list = [int(sample_num) for sample_num in sample_num_list]
    repeat_num = 5  # 先 统计 5 次随机采样 importent
    ans = {}
    ans["base_acc"] = base_acc
    ans["base_A_acc"] = Base_acc_config["A_acc"]
    ans["base_B_acc"] = Base_acc_config["B_acc"]
    ans["retrained_acc"] = {}
    # epoch_dic = {1:5,3:5,5:5,10:10,15:15,20:20,50:30,80:30,100:30}
    epoches = 5
    for sample_num in sample_num_list:
        # epoches = epoch_dic.get(sample_num)
        ans["retrained_acc"][sample_num] = []
        cur_dir = os.path.join(common_dir, str(sample_num))
        for repeat in range(repeat_num):
            # 新的重复实验，得将模型池重新加载初始各方预训练模型!
            model_pool = load_models_pool(config, lr)
            # 加载stu_model
            stu_model = load_model(config["stu_model_path"])
            stu_model = froze_model(stu_model) # importent !!!
            csv_file_name = "sampled_"+str(repeat)+".csv"
            retrain_csv_path = os.path.join(cur_dir, csv_file_name)
            retrain_df = pd.read_csv(retrain_csv_path)
            # 加载标记代价采样集        
            acc= kd(model_pool, stu_model, retrain_df, merged_test_df, base_acc, epoches = epoches)    
            improve_acc = round(acc-base_acc,4)
            obj = {}
            obj["improve_acc"] = improve_acc
            obj["improve_A_acc"] = None
            obj["improve_B_acc"] = None
            print(f"目标采样比例:{sample_num}%, 实际采样数量:{retrain_df.shape[0]}, 训练轮次:{epoches}, 重复次数:{repeat}, 混合评估精度提高:{improve_acc}")
            ans["retrained_acc"][sample_num].append(obj)   
    print(ans)
    # 保存ans
    save_dir = config["save_retrainResult_path"]
    file_name = "reTrain_acc.data"
    file_path = os.path.join(save_dir, file_name)
    saveData(ans, file_path)
    print("save success finally")                                                                                             




#全局变量区
# 设置训练显卡
os.environ['CUDA_VISIBLE_DEVICES']='4'
config = animal_3_config
Base_acc_config = Base_acc.animal_3
merged_test_df = pd.read_csv(config["merged_df_path"])
lr = 1e-3
# 加载各方的评估集
df_eval_party_A = pd.read_csv(config["df_eval_party_A_path"])
df_eval_party_B = pd.read_csv(config["df_eval_party_B_path"])
df_eval_party_list = [df_eval_party_A, df_eval_party_B]
# 加载数据生成
generator_A = config["generator_A"]
generator_B = config["generator_B"]
generator_stu_train = ImageDataGenerator(rescale=1/255.)
generator_A_test = config["generator_A_test"]
generator_B_test = config["generator_B_test"]
generator_stu_test = ImageDataGenerator(rescale=1/255.)
target_size_A = config["target_size_A"]
target_size_B = config["target_size_B"]
target_size_stu = (224,224)
generator_train_list = [generator_A, generator_B]
generator_test_list = [generator_A_test, generator_B_test]
target_size_list = [target_size_A, target_size_B] 
# 双方的class_name_list
class_name_list_A = getClasses(config["dataset_A_train_path"]) # sorted
class_name_list_B = getClasses(config["dataset_B_train_path"]) # sorted
# local_class_name_list_list
class_name_list_list = [class_name_list_A, class_name_list_B]
# all_class_name_list
all_class_name_list = list(set(class_name_list_A+class_name_list_B))
all_class_name_list.sort()
# 总分类数
all_class_nums = len(all_class_name_list)
# 双方的mapping
# 双方的local to global
local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
localToGlobal_mapping = [local_to_global_party_A, local_to_global_party_B]


if __name__ == "__main__":
    retrain()

    

