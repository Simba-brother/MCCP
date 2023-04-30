# github: https://github.com/YuriWu/HMR

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

from sklearn.utils import shuffle
import math
import os
import random
import joblib
from utils import deleteIgnoreFile,saveData
# 加载数据集 config
from DataSetConfig import food_config, fruit_config, sport_config, weather_config, flower_2_config, car_body_style_config, animal_config, animal_2_config, animal_3_config

# 设置训练显卡
os.environ['CUDA_VISIBLE_DEVICES']='4'

# 配置变量
config = animal_3_config

def add_reserve_class(model):
    '''
    给模型增加一个保留class神经元
    '''
    # copy the original weights, to keep the predicting function same
    weights_bak = model.layers[-1].get_weights()
    num_classes = model.layers[-1].output_shape[-1]
    # model.pop()
    # model.add(Dense(num_classes + 1, activation='softmax'))
    # cut 最后输出
    model_cut = keras.Model(inputs = model.input, outputs = model.get_layer(index = -2).output)
    # 声明出一个新输出层
    new_output_layer = Dense(num_classes + 1, activation='softmax', name="new_output_layer")(model_cut.output)
    # 重新连上新输出层
    model = Model(inputs = model_cut.input, outputs = new_output_layer)
    # model.summary()
    weights_new = model.layers[-1].get_weights()
    weights_new[0][:, :-1] = weights_bak[0]
    weights_new[1][:-1] = weights_bak[1]

    # use the average weight to init the last. This suppress its output, while keeping performance.
    weights_new[0][:, -1] = np.mean(weights_bak[0], axis=1)
    weights_new[1][-1] = np.mean(weights_bak[1])

    model.layers[-1].set_weights(weights_new)

    # model.compile(loss=categorical_crossentropy,
    #             optimizer=Adam(learning_rate=1e-4),
    #             metrics=['accuracy'])
    return model

def getClasses(dir_path):
    '''
    得到数据集目录的class_name_list
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()
    return classes_name_list

class Tunnel():

    def __init__(self, names):
        self.names = names # [0,1]
        self.data = {}
        for sender in names:
            self.data[sender] = {}
            for receiver in names:
                self.data[sender][receiver] = []

    def send(self, sender, receiver, row):
        '''
        row:比如是df的一行
        注意这是append
        '''
        self.data[sender][receiver].append(row)

    def receive(self, receiver):
        '''
         该receiver接收到的所有
        '''
        df = None
        row_list = []
        for sender in self.names:
            row_list.extend(self.data[sender][receiver])
        df = pd.DataFrame(row_list)
        return df

def load_models_pool(config, lr=1e-3):
    # 加载模型
    model_A = load_model(config["model_A_struct_path"])
    if not config["model_A_weight_path"] is None:
        model_A.load_weights(config["model_A_weight_path"])

    model_B = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_B.load_weights(config["model_B_weight_path"])
    model_A = add_reserve_class(model_A)
    model_B = add_reserve_class(model_B)
    model_A.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    model_B.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=lr),
                metrics=['accuracy'])
    # 各方模型池
    models_pool = [model_A, model_B]
    # 模型池中的模型是否具备保留类功能
    hasReceived_pool = [True, True]
    return models_pool, hasReceived_pool

def load_extended_models(config):
    # 加载模型
    model_A = load_model(config["model_A_struct_path"])
    if not config["model_A_weight_path"] is None:
        model_A.load_weights(config["model_A_weight_path"])

    model_B = load_model(config["model_B_struct_path"])
    if not config["model_B_weight_path"] is None:
        model_B.load_weights(config["model_B_weight_path"])
    extended_model_A = add_reserve_class(model_A)
    extended_model_B = add_reserve_class(model_B)
    model_A.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=1e-5),
                metrics=['accuracy'])
    model_B.compile(loss=categorical_crossentropy,
                optimizer=Adamax(learning_rate=1e-5),
                metrics=['accuracy'])
    return extended_model_A, extended_model_B

# 文件全局变量区
# 加载混合评估集
merged_df = pd.read_csv(config["merged_df_path"])
# 加载各方的评估集
df_eval_party_A = pd.read_csv(config["df_eval_party_A_path"])
df_eval_party_B = pd.read_csv(config["df_eval_party_B_path"])
df_eval_party_list = [df_eval_party_A, df_eval_party_B]
# 加载数据生成
generator_A = config["generator_A"]
generator_B = config["generator_B"]
generator_A_test = config["generator_A_test"]
generator_B_test = config["generator_B_test"]
target_size_A = config["target_size_A"]
target_size_B = config["target_size_B"]
generator_train_list = [generator_A, generator_B]
generator_test_list = [generator_A_test, generator_B_test]
target_size_list = [target_size_A, target_size_B] 
# 加载模型池
models_pool, hasReceived_pool = load_models_pool(config)
# 双方的local to global
local_to_global_party_A = joblib.load(config["local_to_global_party_A_path"])
local_to_global_party_B = joblib.load(config["local_to_global_party_B_path"])
# 双方的训练集目录
dataset_A_dir = config["dataset_A_train_path"]
dataset_B_dir = config["dataset_B_train_path"]
# 双方的class_name_list
class_name_list_A = getClasses(dataset_A_dir) # sorted
class_name_list_B = getClasses(dataset_B_dir) # sorted
# local_class_name_list_list
class_name_list_list = [class_name_list_A, class_name_list_B]
# all_class_name_list
all_class_name_list = list(set(class_name_list_A+class_name_list_B))
all_class_name_list.sort()
# 总分类数
all_class_nums = len(all_class_name_list)
# 双方的mapping
localToGlobal_mapping = [local_to_global_party_A, local_to_global_party_B]



def global_class_to_local_class(party, global_label_index):
    '''
    global_label_index => local_label_index
    args:
        party:某一方索引
        global_label_index:全局上的label_index
    return:
        local_label_index:对应该方的local_label_index
    '''
    # 得到某一方local_label_index:global_label_index
    mapping = localToGlobal_mapping[party]
    # 对该方来说是否是保留flag
    reserved_flag = False
    local_label_index = 0
    if global_label_index not in list(mapping.values()):
        # 是保留类
        reserved_flag = True
        # local_label_index设置为+1
        local_label_index = max(list(mapping.keys()))+1
    else: # 不是保留类    
        for key, value in mapping.items():
            value = mapping[key]
            if value == global_label_index:
                local_label_index = key
                break
    return local_label_index, reserved_flag



def adaptModel(i, received_flag, hasReceived_pool):
    '''
    得到某一方模型,并根据是否需要保留类flag和是否已经具备保留类flag来判断是否改造model
    args:
        i:party index
        received_flag:需要保留类flag
        hasReceived_pool][i]:i方模型是否已经具备保留类功能
    return:
        model:模型
        hasRecevied_flag:该方模型是否已经具备保留类功能flag
    '''
    model = models_pool[i]
    hasRecevied = hasReceived_pool[i]
    if hasRecevied is False:
        if received_flag is True:
            model = add_reserve_class(model)
            hasRecevied = True
    return model, hasRecevied

def get_need_classes(df, i):
    '''
    根据reTrain所用数据集df,确定出i方模型需要分的class_name_list
    args:
        df:reTrain dataset
        i:party_index
    return:
        class_name_list:该方模型需要完成的class_name_list
        received_flag:是否需要改造该方模型输出
    '''
    need_class_name_list = []
    class_name_list = class_name_list_list[i]
    need_class_name_list.extend(class_name_list)
    need_classes = list(df["label"].unique())
    received_flag = False
    dif_classes = set(class_name_list)^set(need_classes)
    if len(dif_classes) != 0:
        need_class_name_list.append("zzz")
        received_flag = True
    return need_class_name_list, received_flag


def calibrate(tunnel, receiver, dataset):
    """ 
    校准receiver的prob_output_vector
    args:
        tunnel:数据传递通道类
        receiver:事件接收方party_idx
        dataset:分配给各方原始数据
    output:
        model:receiver方校准后模型
        hasRecevied_flag:该方模型是否具备分类保留类功能
    """
    # 获得该方收到的数据集
    df_received = tunnel.receive(receiver)
    # if reserved_flag:
    #     received_y = to_categorical(received_y, m + 1)
    #     # pad zeros at last，扩展特征维度
    #     y = np.concatenate((y, np.zeros((n, 1))), axis=1)
    # else:
    #     # 如果该样本的global label 在 local label里
    #     received_y = to_categorical(received_y, m)
    # 获得该方原有的分配的数据集
    df_test = dataset[receiver]
    # concat起来
    df = pd.concat([df_test, df_received], ignore_index=True)
    # 把参与校准数据集中,不属于receiver方数据的label 设置为zzz,为保留标签
    df.loc[(df["source"]!=(receiver+1))&(df["is_overlap"]==0), "label"] = "zzz"
    # 获得需要的分类,和是否含有保留类
    classes, received_flag = get_need_classes(df, receiver)
    # 获得receiver方的generator
    generator = generator_train_list[receiver]
    # 获得receiver方的target_size
    target_size = target_size_list[receiver]
    # 生成batches
    batch_size = 8
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            # subset="training",
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=classes, shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)

    # 进一步获得模型
    model, hasRecevied = adaptModel(receiver, received_flag, hasReceived_pool)
    model.compile(loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy'])
    # 开始训练
    history = model.fit(batches,
                    steps_per_epoch=df.shape[0]//batch_size,
                    epochs = 1,
                    # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                    # validation_data=batches_test,
                    # validation_steps = merged_df.shape[0]//batch_size,
                    verbose = 1,
                    shuffle=False)
    return model, hasRecevied

def output_localToGlobal(output_local, i, all_class_nums, localToGlobal_mapping):
    '''
    本地输出概率 映射 到全局输出概率
    args:
        output_local:本地输出概率
        i:某方
        all_class_nums:全分类数量
        localToGlobal_mapping[i]:第方local_label_index => global_label_index
    return:
        predict_value: global_predict_prob
    '''
    n = output_local.shape[0]
    predict_value = np.zeros((n,all_class_nums))
    localToGlobal_dic = localToGlobal_mapping[i]
    mapping = []
    for key, value in localToGlobal_dic.items():
        mapping.append(value)
    predict_value[:, mapping] = output_local
    return predict_value

def MPMC_margin(row, models_pool):
    '''
    获得样本x的margin
    args:
        row:dataFrame随机选的的一行
        models_pool:各方模型池
    return:
        margin: correct_max - incorrect_max 如果小于0 该样本x就需被用于去校准,即消耗一个budget
        i_pos:correct_max对应的party_idx。一会要校准该方 增大correct_max
        i_neg:incorrect_max对应的party_idx 一会要校准该方 降低incorrect_max
    '''
    label_globalIndex = row["label_globalIndex"]
    row_df = row.to_frame()
    df_instance = pd.DataFrame(row_df.values.T, columns=row_df.index)
    # ground_truth label probability 最大 正方
    i_pos = None
    correct_max = 0
    # error label probability 最小 反方
    i_neg = None
    incorrect_max = 0
    # 遍历各方的model
    for i, model in enumerate(models_pool):
        # 此方local概率输出, 该模型可能已经适配了保留类
        output_proba_local = predict_proba(i, df_instance, batch_size=1, remove_reserved_class=True)
        # 全局概率输出
        output_proba_global = output_localToGlobal(output_proba_local,i, all_class_nums, localToGlobal_mapping)
        output_proba_global = np.ravel(output_proba_global)
        if output_proba_global[label_globalIndex] > correct_max:
            # 如果该样本ground_true 的 那个槽位的global label 的概率
            # 也就是说i_pos是true_label 最大概率那一方
            i_pos = i
            # 更新true_label 最大概率
            correct_max = output_proba_global[label_globalIndex]
        # 把该实例在此方的true_label预测的概率
        output_proba_global[label_globalIndex] = 0
        # 找到该实例在此方除了true_label的最大概率
        max_proba = np.max(output_proba_global)
        # 更新不正确最大方
        if max_proba > incorrect_max:
            i_neg = i
            incorrect_max = max_proba
    # 该实例边缘就是，正确分类的最大概率 - 不正确分类中的最大概率
    margin = correct_max - incorrect_max
    return (margin, i_pos, i_neg)

# 数据传输隧道
tunnel = Tunnel([0,1])


def predict_proba(i, df, batch_size, remove_reserved_class=True):
    '''
    i方的对df的预测概率
    '''
    model = models_pool[i]
    hasReserved = hasReceived_pool[i] 
    generator = generator_test_list[i]
    target_size = target_size_list[i]
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            # subset="training",
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=None, shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)

    proba = model.predict(batches,    
                    batch_size=None,
                    verbose="auto",
                    steps=None,
                    callbacks=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False)

    if hasReserved and remove_reserved_class:
        # 如果i方已经具备保留类功能 and 同意移除保留类proba
        return proba[:, :-1]
    return proba

def local_to_global(i,proba_local):
    '''
    i方的proba => globa proba
    '''
    predict_value = np.zeros((proba_local.shape[0], all_class_nums))
    localToGlobal_dic = localToGlobal_mapping[i]
    mapping = []
    for key, value in localToGlobal_dic.items():
        mapping.append(value)
    predict_value[:, mapping] = proba_local
    return predict_value

def evaluate_on(models_pool,df):
    '''
    集成评估
    args:
        models_pool:模型池
        df:评估数据集
    return:
        accuracy
    '''
    n = df.shape[0]
    predict_value = np.zeros((len(models_pool), n, all_class_nums))
    for i,model in enumerate(models_pool):
        proba_local = predict_proba(i, df, batch_size=8, remove_reserved_class=True)
        proba_global = local_to_global(i,proba_local)
        predict_value[i, :, :] = proba_global
    proba_vector = np.max(predict_value, axis=0)
    predict_global_idx_array = np.argmax(proba_vector,axis=1)
    true_global_idx_array = np.array(df["label_globalIndex"])
    accuracy = np.sum(predict_global_idx_array == true_global_idx_array) / df.shape[0]
    return accuracy

def HMR(budget, dataset_pool, models_pool, df_test_mix):
    accuracy_integrate = evaluate_on(models_pool,df_test_mix)
    for step in range(0, 1000):
        if budget <= 0:
            break
        # 随机从模型池子选择一方模型,作为发送方
        random_sender = random.randrange(len(models_pool))
        # random_sender = random.choice(models_pool)
        df_sender = dataset_pool[random_sender]
        # 从混合数据集中随机选择一个即消耗一个budget
        sampled_df = df_sender.sample(n=1, axis=0) # random_state=123
        row = sampled_df.iloc[0]
        # prefix_path = "/data/mml/overlap_v2_datasets/"
        # file_path = os.path.join(prefix_path, row["file_path"])
        # # 该样本标签
        # label = row["label"]
        # # 该样本标签global_index
        # label_globalIndex = row["label_globalIndex"]
        # # file_path => PIL.Image
        # image = load_img(file_path, color_mode="rgb", target_size=None, interpolation="nearest")
        # # 样本
        # x = img_to_array(image) 
        margin, pos_receiver, neg_receiver = MPMC_margin(row, models_pool)
        if margin <= 0:  # violated， 不正确分类的最大概率 > 正确分类的最大概率了。需要将该实例发送到i_pos,i_neg
            print('sender: %d, pos: %d, neg: %d, class: %-6s, margin: %.6f, budget:%d, step:%d' %
                    (random_sender, pos_receiver, neg_receiver, row["label"], margin, budget, step))
            tunnel.send(random_sender, pos_receiver, row)
            tunnel.send(random_sender, neg_receiver, row)
            model_positive, hasRecevied_flag_pos = calibrate(tunnel, pos_receiver, dataset_pool)
            model_negtive, hasRecevied_flag_neg = calibrate(tunnel, neg_receiver, dataset_pool)
            models_pool[pos_receiver] = model_positive
            models_pool[neg_receiver] = model_negtive
            hasReceived_pool[pos_receiver] = hasRecevied_flag_pos
            hasReceived_pool[neg_receiver] = hasRecevied_flag_neg
            accuracy_integrate = evaluate_on(models_pool,df_test_mix)
            budget -= 1
        else:
            print('sender: %d, pos: %d, neg: %d, margin: %.6f, step:%d' %
                    (random_sender, pos_receiver, neg_receiver, margin, step))
            continue

    return accuracy_integrate

def HRM_improve(models_pool, df_sampled, df_test):
    '''
    对HRM的改进。人工标记df 中 margin < 0的 统统拿去 给双发 calibrate 
    args:
        models_pool: 各方模型池
        df_sampled: 标注代价采样集
        df_test: 集成模型评估集
    return:
        accuracy_integrate:集成模型在混合测试集上的评估
    '''
    # 存分别用于各方calibrate的样本idx
    group = {0:[],1:[]}
    for row_index, row in df_sampled.iterrows():
        margin, pos_receiver, neg_receiver = MPMC_margin(row, models_pool)
        if margin < 0:
            group[pos_receiver].append(row["merged_idx"])
            group[neg_receiver].append(row["merged_idx"])
    trueOrFalse_A_list = [ True if merged_idx in group[0] else False for merged_idx in df_sampled["merged_idx"] ]
    trueOrFalse_B_list = [ True if merged_idx in group[1] else False for merged_idx in df_sampled["merged_idx"] ]
    # 获得要用于校准各方的数据
    df_calibrate_A = df_sampled[trueOrFalse_A_list]
    df_calibrate_B = df_sampled[trueOrFalse_B_list]
    print("用于校准样本数量:{}".format(df_calibrate_A.shape[0]+df_calibrate_B.shape[0]))
    # 开始校准校准
    models_pool[0], hasReceived_pool[0] = calibrate_improve(df_calibrate_A, 0)
    models_pool[1], hasReceived_pool[1] = calibrate_improve(df_calibrate_B, 1)
    # 评估
    accuracy_integrate = evaluate_on(models_pool,df_test)
    return accuracy_integrate

def calibrate_improve_v2(df, models_pool, i):
    '''
    功能:对[i]方extended_model进行retrain
    args:
        df:retrain_df
        models_pool:各方extended_model
        i:[i]方
    return: retrained_model
    '''
    # 获得extended model
    epochs = 5
    extended_model = models_pool[i]
    classes = class_name_list_list[i]
    ex_classes = []
    ex_classes.extend(classes)
    ex_classes.append("zzz")  # importent!!
    # 获得receiver方的generator
    generator = generator_train_list[i]
    # 获得receiver方的target_size
    target_size = target_size_list[i]
    # 生成batches
    batch_size = 5 # min(math.ceil(df.shape[0]/4), 32)
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, 
                                            class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=ex_classes, 
                                            shuffle=True, batch_size=batch_size,
                                            validate_filenames=False)
    # 开始训练
    history = extended_model.fit(batches,
                    steps_per_epoch=df.shape[0]//batch_size,
                    epochs = epochs,
                    # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                    # validation_data=batches_test,
                    # validation_steps = merged_df.shape[0]//batch_size,
                    verbose = 1,
                    shuffle=True)
    return extended_model

def HRM_improve_2(extended_model_A, extended_model_B, df_sampled, df_test):
    # 开始校准校准
    models_pool[0], hasReceived_pool[0] = calibrate_improve(df_sampled, 0)
    # HRM后A方model在自家评估集acc
    acc_local_A = eval_singleModel(0)
    models_pool[1], hasReceived_pool[1] = calibrate_improve(df_sampled, 1)
    # HRM后B方model
    acc_local_B = eval_singleModel(1)
    # 评估
    accuracy_integrate = evaluate_on(models_pool,df_test)
    return accuracy_integrate, acc_local_A, acc_local_B

def get_retrain_df(df, flag):
    if flag == "retrain_A":
        df.loc[(df["source"]!=1)&(df["is_overlap"]==0), "label"] = "zzz"    
        return df
    if flag == "retrain_B":
        df.loc[(df["source"]!=2)&(df["is_overlap"]==0), "label"] = "zzz"    
        return df
    else:
        raise Exception("flag指定错误")
    
def HRM_improve_v3(models_pool, retrain_csv_path, df_test):
    # 开始校准extended_model_A
    retrain_df = pd.read_csv(retrain_csv_path)
    retrain_A_df = get_retrain_df(retrain_df, "retrain_A") # retrain_df被写了
    models_pool[0] = calibrate_improve_v2(retrain_A_df, models_pool, 0)
    # HRM后A方model在自家评估集acc
    acc_local_A = eval_singleModel_v2(models_pool,0)
    # 开始校准extended_model_B
    retrain_df = pd.read_csv(retrain_csv_path)
    retrain_B_df = get_retrain_df(retrain_df, "retrain_B")
    models_pool[1] = calibrate_improve_v2(retrain_B_df, models_pool, 1)
    # HRM后B方model
    acc_local_B = eval_singleModel_v2(models_pool,1)
    # 评估
    accuracy_integrate = evaluate_on(models_pool,df_test)
    return accuracy_integrate, acc_local_A, acc_local_B

def HRM_improve(models_pool, df_sampled, df_test):
    '''
    对HRM的改进。人工标记df 中 margin < 0的 统统拿去 给双发 calibrate 
    args:
        models_pool: 各方模型池
        df_sampled: 标注代价采样集
        df_test: 集成模型评估集
    return:
        accuracy_integrate:集成模型在混合测试集上的评估
    '''
    # 存分别用于各方calibrate的样本idx
    group = {0:[],1:[]}
    for row_index, row in df_sampled.iterrows():
        margin, pos_receiver, neg_receiver = MPMC_margin(row, models_pool)
        if margin < 0:
            group[pos_receiver].append(row["merged_idx"])
            group[neg_receiver].append(row["merged_idx"])
    trueOrFalse_A_list = [ True if merged_idx in group[0] else False for merged_idx in df_sampled["merged_idx"] ]
    trueOrFalse_B_list = [ True if merged_idx in group[1] else False for merged_idx in df_sampled["merged_idx"] ]
    # 获得要用于校准各方的数据
    df_calibrate_A = df_sampled[trueOrFalse_A_list]
    df_calibrate_B = df_sampled[trueOrFalse_B_list]
    print("用于校准样本数量:{}".format(df_calibrate_A.shape[0]+df_calibrate_B.shape[0]))
    # 开始校准校准
    models_pool[0], hasReceived_pool[0] = calibrate_improve(df_calibrate_A, 0)
    models_pool[1], hasReceived_pool[1] = calibrate_improve(df_calibrate_B, 1)
    # 评估
    accuracy_integrate = evaluate_on(models_pool,df_test)
    return accuracy_integrate


def calibrate_improve(df, i):
    '''
    用df集对模型池calibrate
    args:
        df:用于校准modes_pool的数据集
        i: party_i
    return:
        model:校准后的model
        hasRecevied:该模型是否具备保留类功能
    notes:epochs = 1
    '''
    epochs = 5
    # 把参与校准数据集中,不属于i方数据的label 设置为other,为保留标签
    df.loc[(df["source"]!=(i+1))&(df["is_overlap"]==0), "label"] = "zzz"
    # 获得需要的分类,和是否含有保留类
    classes, received_flag = get_need_classes(df, i)

    # 获得receiver方的generator
    generator = generator_train_list[i]
    # 获得receiver方的target_size
    target_size = target_size_list[i]
    # 生成batches
    batch_size = min(math.ceil(df.shape[0]/4), 32)
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            # subset="training",
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, 
                                            class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=classes, 
                                            shuffle=False, batch_size=batch_size,
                                            validate_filenames=False)

    # 进一步获得模型
    model, hasRecevied = adaptModel(i, received_flag, hasReceived_pool)
    # 开始训练
    history = model.fit(batches,
                    steps_per_epoch=df.shape[0]//batch_size,
                    epochs = epochs,
                    # callbacks=[checkpointer, learning_rate_reduction], #[lr_scheduler, early_stopping], 
                    # validation_data=batches_test,
                    # validation_steps = merged_df.shape[0]//batch_size,
                    verbose = 1,
                    shuffle=False)
    return model, hasRecevied

def data_alloc(df):
    # 找到budget集中的overlap
    df_overlap  = df[df["is_overlap"]==1]
    # 打乱overlap标签
    df_overlap_shuffle = shuffle(df_overlap)
    # 切一半分给AB双方
    cut_off = round(df_overlap_shuffle.shape[0]/2)
    df_overlap_shuffle_1 = df_overlap_shuffle[0:cut_off]
    df_overlap_shuffle_2 = df_overlap_shuffle[cut_off:df_overlap_shuffle.shape[0]]
    # A方unique数据
    df_unique_A = df[(df["is_overlap"]==0) & (df["source"]==1)]
    # B方unique数据
    df_unique_B = df[(df["is_overlap"]==0) & (df["source"]==2)]
    # 构建出AB方数据集
    df_A = pd.concat([df_unique_A, df_overlap_shuffle_1], ignore_index=True)
    df_B = pd.concat([df_unique_B, df_overlap_shuffle_2], ignore_index=True)
    dataset_pool = [df_A, df_B]
    return dataset_pool


def eval_singleModel(i):
    '''
    集成评估
    args:
        models_pool:模型池
        df:评估数据集
    return:
        accuracy
    '''
    # 过得该方的评估集
    df = df_eval_party_list[i]
    # 获得该方校准后的model
    model = models_pool[i]
    hasReceived = hasReceived_pool[i]
    # 获得该方原始分类数
    class_name_list = class_name_list_list[i]
    classes = []
    classes.extend(class_name_list)
    if hasReceived is True:
        classes.append("zzz")
    generator = generator_test_list[i]
    batch_size = 8
    target_size = target_size_list[i]
    batches = generator.flow_from_dataframe(df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            # subset="training",
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=classes, shuffle=False, 
                                            batch_size=batch_size,
                                            validate_filenames=False)
        # 开始评估
    eval_matric = model.evaluate(batches,steps=batches.n / batch_size, verbose = 1)
    acc = eval_matric[1]
    return acc

def eval_singleModel_v2(models_pool, i):
    '''
    功能: 评估一下[i]方retrained_extended model在local_i的acc
    '''
    correct_num = 0
    # 获得extended_model
    extended_model = models_pool[i]
    # 获得local_i test dataset
    test_df = df_eval_party_list[i]
    total  =test_df.shape[0]
    # 获得classes
    class_name_list = class_name_list_list[i]
    classes = []
    classes.extend(class_name_list)
    classes.append("zzz")
    batch_size = 5
    # 获得test生成器
    generator = generator_test_list[i]
    # 获得target_size
    target_size = target_size_list[i]
    # 获得batches
    batches = generator.flow_from_dataframe(test_df, 
                                            directory = "/data/mml/overlap_v2_datasets/", # 添加绝对路径前缀
                                            seed=42,
                                            x_col='file_path', y_col="label", 
                                            target_size=target_size, class_mode='categorical', # one-hot
                                            color_mode='rgb', classes=classes, shuffle=False, 
                                            batch_size=batch_size,
                                            validate_filenames=False)
    for i in range(len(batches)):
        batch = next(batches)
        X = batch[0]
        Y = batch[1] # one hot

        out = extended_model.predict(X, batch_size=None, verbose=0)
        out_cut = out[:,:-1]
        p_label = np.argmax(out_cut, axis = 1)
        ground_label = np.argmax(Y, axis = 1)
        correct_num += np.sum(p_label==ground_label)
    acc = round(correct_num/total, 4)
    # loss, acc = extended_model.evaluate_generator(batches,steps=batches.n / batch_size, verbose = 1)
    return acc
    
if __name__ == "__main__":
    common_dir = config["sampled_common_path"]
    sample_num_list = deleteIgnoreFile(os.listdir(common_dir))
    sample_num_list = sorted(sample_num_list, key=lambda e: int(e))
    sample_num_list = [int(sample_num) for sample_num in sample_num_list]
    repeat_num = 5  # 先 统计 5 次随机采样 importent
    lr = 1e-5
    ans = {}
    models_pool, _ = load_models_pool(config, lr) # 输出层添加了"zzz"类别
    # extended_model_A, extended_model_B = load_extended_models(config)
    # 获得init acc
    accuracy_integrate_init = evaluate_on(models_pool,merged_df)
    accuracy_integrate_init = round(accuracy_integrate_init,4)
    acc_init_A =  round(eval_singleModel(0),4)
    acc_init_B =  round(eval_singleModel(1),4)
    print(f"集成模型混合集初始精度:{accuracy_integrate_init}, extended_model_A初始精度:{acc_init_A}, extended_model_B初始精度:{acc_init_B}")
    for sample_num in sample_num_list:
        ans[sample_num] = []
        cur_dir = os.path.join(common_dir, str(sample_num))
        for repeat in range(repeat_num):
            # 新的重复实验，得将模型池重新加载初始各方预训练模型！！！！！并且添加了"zzz"
            models_pool, _ = load_models_pool(config, lr)
            csv_file_name = "sampled_"+str(repeat)+".csv"
            retrain_csv_path = os.path.join(cur_dir, csv_file_name)
            retrain_df = pd.read_csv(retrain_csv_path)
            # 加载标记代价采样集        
            accuracy_integrate, acc_local_A, acc_local_B= HRM_improve_v3(models_pool, retrain_csv_path, merged_df)           
            accuracy_integrate = round(accuracy_integrate,4)
            acc_local_A = round(acc_local_A,4)
            acc_local_B = round(acc_local_B,4)
            acc_improve_integrate = round(accuracy_integrate - accuracy_integrate_init,4)
            acc_improve_local_A = round(acc_local_A - acc_init_A,4)
            acc_improve_local_B = round(acc_local_B - acc_init_B,4)
            print("目标采样数量:{}, 实际采样数量:{}, 重复次数:{}, 混合评估精度提高:{}, local_A评估精度提高:{}, local_B评估精度提高:{}".format(sample_num, 
                                                                                                            retrain_df.shape[0],
                                                                                                            repeat, 
                                                                                                            acc_improve_integrate, 
                                                                                                            acc_improve_local_A,
                                                                                                            acc_improve_local_B
                                                                                                            ))
            ans[sample_num].append({"acc_improve_integrate":acc_improve_integrate, "acc_improve_local_A":acc_improve_local_A, "acc_improve_local_B":acc_improve_local_B})
    print(ans)
    # 保存ans
    save_dir = config["save_retrainResult_path"]
    file_name = "reTrain_acc_improve_accords_alluse_new.data"
    file_path = os.path.join(save_dir, file_name)
    saveData(ans, file_path)
    print("save success finally")